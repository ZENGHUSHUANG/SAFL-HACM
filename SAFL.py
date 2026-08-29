"""
SAFL-HACM refactor (reference implementation)
Based on the paper:
"Semi-Asynchronous Federated Learning with Heterogeneity-Aware Coordination Mechanism
 in Mobile Edge Computing Networks"

This file focuses on the key components and is designed to be integrated with the
existing Edge_Device / dataset / model code.

Key paper-aligned components:
1) Age-aware model and learning rate
2) Local Drift Control (LDC)
3) Cluster-balanced / participation-aware scheduling
4) Latency-aware bandwidth allocation
5) Data-size-weighted pseudo-gradient aggregation
6) Buffered semi-asynchronous update lifecycle
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import torch

TensorDict = Dict[str, torch.Tensor]


# ---------------------------------------------------------------------
# Tensor-dict helpers
# ---------------------------------------------------------------------

def zeros_like_state(state: Mapping[str, torch.Tensor]) -> TensorDict:
    return {k: torch.zeros_like(v) for k, v in state.items()}


def clone_state(state: Mapping[str, torch.Tensor]) -> TensorDict:
    return {k: v.detach().clone() for k, v in state.items()}


def state_sub(a: Mapping[str, torch.Tensor],
              b: Mapping[str, torch.Tensor]) -> TensorDict:
    """a - b."""
    return {k: a[k] - b[k] for k in a.keys()}


def state_add_scaled_(dst: MutableMapping[str, torch.Tensor],
                      src: Mapping[str, torch.Tensor],
                      scale: float) -> None:
    with torch.no_grad():
        for k in dst.keys():
            dst[k].add_(src[k], alpha=scale)


# ---------------------------------------------------------------------
# Buffered update used by semi-asynchronous server
# ---------------------------------------------------------------------

@dataclass
class BufferedUpdate:
    client_id: int
    start_round: int              # s_k(t): global model index used to start local training
    pseudo_grad: TensorDict       # g_k^t = w_start - w_end
    local_steps: int              # K_k / E_k
    local_lr: float               # eta_k
    local_loss: float             # F_{k,t}
    num_samples: int              # |D_k|

    @property
    def local_avg_grad(self) -> TensorDict:
        denom = max(self.local_steps * self.local_lr, 1e-12)
        return {k: v / denom for k, v in self.pseudo_grad.items()}


# ---------------------------------------------------------------------
# 1. Age-aware model, paper Eq. (6)
# ---------------------------------------------------------------------

def update_age(current_round: int, start_round: int) -> int:
    """a_{k,t} = t - s_k(t)."""
    return max(0, current_round - start_round)


def age_aware_global_lr(
    base_lr: float,
    ages: Sequence[int],
    age_threshold: float,
    decay: float,
) -> float:
    """
    Paper Eq. (6):
        eta_t = eta,                         if avg_age <= a_c
              = eta * epsilon^(avg_age-a_c), otherwise

    decay must be in (0, 1).
    """
    if not ages:
        return base_lr
    if not (0.0 < decay < 1.0):
        raise ValueError("decay must be in (0, 1).")

    avg_age = sum(ages) / len(ages)
    if avg_age <= age_threshold:
        return base_lr
    return base_lr * (decay ** (avg_age - age_threshold))


# ---------------------------------------------------------------------
# 2. Local Drift Control (LDC), paper Eq. (13)-(19)
# ---------------------------------------------------------------------

def ldc_objective(
    model: torch.nn.Module,
    empirical_loss: torch.Tensor,
    stale_global_state: Mapping[str, torch.Tensor],
    drift_state: Mapping[str, torch.Tensor],
    global_grad_ref: Mapping[str, torch.Tensor],
    local_grad_ref: Mapping[str, torch.Tensor],
    alpha: float,
    beta: float,
) -> torch.Tensor:
    """
    Paper Eq. (13), (15), (16).

    L_k = ||v_k - (w - w_k)||^2

    G_k is represented as the parameter inner product with the difference between
    the global gradient reference and the local gradient reference.

    IMPORTANT:
    global_grad_ref / local_grad_ref are treated as fixed references for this
    local optimization step. The caller should pass the references corresponding
    to the model version used by the device.
    """
    named_params = dict(model.named_parameters())

    drift_penalty = empirical_loss.new_zeros(())
    direction_term = empirical_loss.new_zeros(())

    for name, p in named_params.items():
        if name not in stale_global_state:
            continue

        v = drift_state[name].detach().to(device=p.device, dtype=p.dtype)
        w_ref = stale_global_state[name].detach().to(device=p.device, dtype=p.dtype)

        drift_penalty = drift_penalty + torch.sum((v - (w_ref - p)) ** 2)

        if name in global_grad_ref and name in local_grad_ref:
            g_global = global_grad_ref[name].detach().to(device=p.device, dtype=p.dtype)
            g_local = local_grad_ref[name].detach().to(device=p.device, dtype=p.dtype)
            direction_term = direction_term + torch.sum(p * (g_global - g_local))

    return empirical_loss + 0.5 * alpha * direction_term + beta * drift_penalty


def update_ldv(
    old_v: Mapping[str, torch.Tensor],
    start_state: Mapping[str, torch.Tensor],
    end_state: Mapping[str, torch.Tensor],
    phi: float,
) -> TensorDict:
    """
    Paper Eq. (18):
        v_k <- (1-phi) v_k + phi (w_end - w_start)
    """
    if not (0.0 < phi < 1.0):
        raise ValueError("phi must be in (0, 1).")

    out = {}
    for k in old_v.keys():
        out[k] = (1.0 - phi) * old_v[k] + phi * (end_state[k] - start_state[k])
    return out


def stale_drift_correction(
    pseudo_grad: Mapping[str, torch.Tensor],
    drift_state: Mapping[str, torch.Tensor],
    age: int,
    rho: float,
) -> TensorDict:
    """
    Paper Eq. (19):
        g_k^t <- g_k^t - rho * a_{k,t} * v_k
    """
    return {
        k: pseudo_grad[k] - rho * float(age) * drift_state[k]
        for k in pseudo_grad.keys()
    }


def update_global_average_gradient(
    previous: Mapping[str, torch.Tensor],
    selected_updates: Sequence[BufferedUpdate],
    mu: float,
) -> TensorDict:
    """
    Paper Eq. (14):
        gbar^t = mu*gbar^(t-1)
               + (1-mu) * (1/R) * sum_k [g_k^t / (K_k * eta_k)]
    """
    if not selected_updates:
        return clone_state(previous)
    if not (0.0 <= mu < 1.0):
        raise ValueError("mu must be in [0, 1).")

    avg = zeros_like_state(previous)
    R = len(selected_updates)

    for upd in selected_updates:
        denom = max(upd.local_steps * upd.local_lr, 1e-12)
        for name in avg.keys():
            avg[name].add_(upd.pseudo_grad[name] / denom, alpha=1.0 / R)

    return {
        name: mu * previous[name] + (1.0 - mu) * avg[name]
        for name in previous.keys()
    }


# ---------------------------------------------------------------------
# 3. Cluster-balanced scheduling, paper Eq. (25)-(26), Algorithm 3
# ---------------------------------------------------------------------

def contribution_score(local_loss: float, participation_count: int) -> float:
    """
    Paper Eq. (25):
        gamma_k = F_{k,t} / (vartheta_{k,t} + 1)
    """
    return float(local_loss) / (float(participation_count) + 1.0)


def cluster_balanced_schedule(
    cluster_of: Mapping[int, int],
    ready_ids: Iterable[int],
    nonready_ids: Iterable[int],
    local_losses: Mapping[int, float],
    participation_count: Mapping[int, int],
    remaining_compute: Mapping[int, float],
    estimated_comm_latency: Mapping[int, float],
    R: int,
) -> List[int]:
    """
    Paper Algorithm 3:
    - Prefer eligible ready updates in every cluster.
    - Within a ready cluster, choose max gamma_k.
    - If a cluster has no ready update, choose the non-ready device with minimum
      remaining-computation + estimated-communication latency.
    - Cycle over clusters until R devices have been selected.

    Stale ready updates should be filtered BEFORE calling this function.
    """
    ready = set(ready_ids)
    nonready = set(nonready_ids)

    cluster_ids = sorted(set(cluster_of.values()))
    ready_by_cluster = {
        c: [k for k in ready if cluster_of[k] == c]
        for c in cluster_ids
    }
    waiting_by_cluster = {
        c: [k for k in nonready if cluster_of[k] == c]
        for c in cluster_ids
    }

    selected: List[int] = []
    selected_set = set()

    while len(selected) < R:
        progress = False

        for c in cluster_ids:
            if len(selected) >= R:
                break

            candidates = [k for k in ready_by_cluster[c] if k not in selected_set]
            if candidates:
                k_star = max(
                    candidates,
                    key=lambda k: contribution_score(
                        local_losses.get(k, 0.0),
                        participation_count.get(k, 0),
                    ),
                )
            else:
                candidates = [k for k in waiting_by_cluster[c] if k not in selected_set]
                if not candidates:
                    continue
                k_star = min(
                    candidates,
                    key=lambda k: (
                        remaining_compute.get(k, math.inf)
                        + estimated_comm_latency.get(k, math.inf)
                    ),
                )

            selected.append(k_star)
            selected_set.add(k_star)
            progress = True

        if not progress:
            break

    return selected


# ---------------------------------------------------------------------
# 4. Latency-aware bandwidth allocation, paper Eq. (27)-(32)
# ---------------------------------------------------------------------

def communication_coefficient(
    model_size: float,
    channel_gain: float,
    tx_power: float,
    noise_psd: float,
) -> float:
    """
    alpha_k = S / log2(1 + G_k P_k / N0)

    With this definition:
        H_cm = alpha_k / b_k
    """
    snr = channel_gain * tx_power / max(noise_psd, 1e-30)
    spectral_eff = math.log2(1.0 + snr)
    if spectral_eff <= 0:
        return math.inf
    return model_size / spectral_eff


def latency_aware_bandwidth_allocation(
    selected: Sequence[int],
    total_bandwidth: float,
    alpha: Mapping[int, float],
    remaining_compute: Mapping[int, float],
    tol: float = 1e-7,
    max_iter: int = 200,
) -> Tuple[Dict[int, float], float]:
    """
    Solve:
        min max_k { H_cp[k] + alpha[k] / b[k] }
        s.t. sum b[k] <= B, b[k] >= 0

    Not-waiting case:
        H_cp[k] = 0 for all selected k
        b_k = alpha_k / sum(alpha) * B        (paper Eq. 32)

    Waiting case:
        b_k = alpha_k / (H - H_cp[k])
    and H is found by binary search so that sum b_k = B.
    """
    if not selected:
        return {}, 0.0
    if total_bandwidth <= 0:
        raise ValueError("total_bandwidth must be positive.")

    selected = list(selected)
    hcp = {k: max(0.0, float(remaining_compute.get(k, 0.0))) for k in selected}

    # Closed-form case (paper Sec. 4.4.1 / Eq. 32)
    if all(v <= tol for v in hcp.values()):
        denom = sum(alpha[k] for k in selected)
        if denom <= 0:
            raise ValueError("sum(alpha) must be positive.")
        bw = {k: total_bandwidth * alpha[k] / denom for k in selected}
        H = max(alpha[k] / max(bw[k], 1e-30) for k in selected)
        return bw, H

    # Waiting case: binary search on equalized completion time H.
    low = max(hcp.values()) + tol

    def required_bw(H: float) -> float:
        total = 0.0
        for k in selected:
            denom = H - hcp[k]
            if denom <= 0:
                return math.inf
            total += alpha[k] / denom
        return total

    # Find a feasible upper bound.
    high = low + max(sum(alpha[k] for k in selected) / total_bandwidth, 1.0)
    for _ in range(100):
        if required_bw(high) <= total_bandwidth:
            break
        high = low + 2.0 * (high - low)
    else:
        raise RuntimeError("Failed to bracket bandwidth-allocation solution.")

    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        need = required_bw(mid)

        if abs(need - total_bandwidth) <= tol * max(total_bandwidth, 1.0):
            low = high = mid
            break

        if need > total_bandwidth:
            low = mid
        else:
            high = mid

    H = 0.5 * (low + high)
    bw = {k: alpha[k] / max(H - hcp[k], 1e-30) for k in selected}

    # Small numerical normalization to use exactly the bandwidth budget.
    s = sum(bw.values())
    if s > 0:
        scale = total_bandwidth / s
        bw = {k: v * scale for k, v in bw.items()}

    H_actual = max(hcp[k] + alpha[k] / max(bw[k], 1e-30) for k in selected)
    return bw, H_actual


# ---------------------------------------------------------------------
# 5. Server aggregation, paper Eq. (4)-(5)
# ---------------------------------------------------------------------

def data_size_weights(
    selected_updates: Sequence[BufferedUpdate],
) -> Dict[int, float]:
    total = sum(max(0, u.num_samples) for u in selected_updates)
    if total <= 0:
        # Safe fallback only when sample counts are unavailable.
        w = 1.0 / max(len(selected_updates), 1)
        return {u.client_id: w for u in selected_updates}
    return {u.client_id: u.num_samples / total for u in selected_updates}


def aggregate_pseudogradient(
    selected_updates: Sequence[BufferedUpdate],
) -> TensorDict:
    """
    Delta_t = sum theta_k(t) * g_k^t
    """
    if not selected_updates:
        raise ValueError("selected_updates must be non-empty.")

    weights = data_size_weights(selected_updates)
    delta = zeros_like_state(selected_updates[0].pseudo_grad)

    for upd in selected_updates:
        theta = weights[upd.client_id]
        for name in delta.keys():
            delta[name].add_(upd.pseudo_grad[name], alpha=theta)

    return delta


def global_model_update(
    global_state: Mapping[str, torch.Tensor],
    delta_t: Mapping[str, torch.Tensor],
    eta_t: float,
) -> TensorDict:
    """
    Paper Eq. (4):
        w^{t+1} = w^t - eta_t * Delta_t

    This sign assumes pseudo_grad = w_start - w_end exactly as defined in the paper.
    """
    return {
        name: global_state[name] - eta_t * delta_t[name]
        for name in global_state.keys()
    }


# ---------------------------------------------------------------------
# 6. Buffer lifecycle helpers
# ---------------------------------------------------------------------

def discard_stale_buffered_updates(
    buffer: MutableMapping[int, BufferedUpdate],
    current_round: int,
    staleness_tolerance: int,
) -> List[int]:
    """
    Paper behavior:
    - Unscheduled completed updates remain buffered.
    - Discard only when age exceeds kappa.
    """
    removed = []
    for k, upd in list(buffer.items()):
        if update_age(current_round, upd.start_round) > staleness_tolerance:
            removed.append(k)
            del buffer[k]
    return removed


def replace_with_fresher_update(
    buffer: MutableMapping[int, BufferedUpdate],
    new_update: BufferedUpdate,
) -> None:
    """
    If the same ED generates a fresher update before its old update is scheduled,
    keep the fresher one.
    """
    old = buffer.get(new_update.client_id)
    if old is None or new_update.start_round >= old.start_round:
        buffer[new_update.client_id] = new_update




# ---------------------------------------------------------------------
# Full paper-aligned two-stage aggregation cycle
# ---------------------------------------------------------------------

def plan_aggregation_cycle(
    *,
    t: int,
    send_buffer: MutableMapping[int, BufferedUpdate],
    cluster_of: Mapping[int, int],
    all_client_ids: Sequence[int],
    participation_count: Mapping[int, int],
    running_local_losses: Mapping[int, float],
    remaining_compute: Mapping[int, float],
    estimated_comm_latency: Mapping[int, float],
    alpha_comm: Mapping[int, float],
    R: int,
    kappa: int,
    total_bandwidth: float,
) -> Tuple[List[int], Dict[int, float], float]:
    """
    Stage A of one SAFL-HACM aggregation cycle.

    1. Remove only expired buffered updates.
    2. Build Omega(t) (ready) and Gamma(t) (not ready).
    3. Schedule R EDs with Algorithm 3.
    4. Allocate bandwidth by Eq. (27)-(32), INCLUDING remaining local
       computation time for selected EDs in Gamma(t).

    Returns:
        selected, bandwidth, H_t

    The simulator should then advance virtual time by H_t. During this interval,
    ongoing local training is NOT interrupted. Once every selected ED's update
    has become available, call finalize_aggregation_cycle().
    """
    discard_stale_buffered_updates(send_buffer, t, kappa)

    ready_ids = set(send_buffer.keys())
    nonready_ids = [k for k in all_client_ids if k not in ready_ids]

    losses = dict(running_local_losses)
    for k in ready_ids:
        losses[k] = send_buffer[k].local_loss

    selected = cluster_balanced_schedule(
        cluster_of=cluster_of,
        ready_ids=ready_ids,
        nonready_ids=nonready_ids,
        local_losses=losses,
        participation_count=participation_count,
        remaining_compute=remaining_compute,
        estimated_comm_latency=estimated_comm_latency,
        R=R,
    )

    if not selected:
        return [], {}, 0.0

    bw, H_t = latency_aware_bandwidth_allocation(
        selected=selected,
        total_bandwidth=total_bandwidth,
        alpha=alpha_comm,
        remaining_compute=remaining_compute,
    )
    return selected, bw, H_t


def finalize_aggregation_cycle(
    *,
    t: int,
    selected: Sequence[int],
    global_state: Mapping[str, torch.Tensor],
    global_avg_grad: Mapping[str, torch.Tensor],
    send_buffer: MutableMapping[int, BufferedUpdate],
    drift_states: Mapping[int, Mapping[str, torch.Tensor]],
    participation_count: MutableMapping[int, int],
    base_global_lr: float,
    age_threshold: float,
    age_decay: float,
    rho: float,
    global_grad_momentum: float,
) -> Tuple[TensorDict, TensorDict, float, List[int]]:
    """
    Stage B of one SAFL-HACM aggregation cycle.

    Call this AFTER the event simulator has advanced time enough that every
    scheduled ED has produced an update and placed it in send_buffer.

    Implements:
      Eq. (19) stale-drift correction
      Eq. (6)  age-aware global learning rate
      Eq. (4)-(5) global pseudo-gradient aggregation
      Eq. (14) global average-gradient update
    """
    missing = [k for k in selected if k not in send_buffer]
    if missing:
        raise RuntimeError(
            "Selected EDs are not ready after advancing the aggregation cycle: "
            f"{missing}"
        )

    corrected_updates: List[BufferedUpdate] = []
    ages: List[int] = []

    for k in selected:
        upd = send_buffer[k]
        age = update_age(t, upd.start_round)
        ages.append(age)

        corrected_grad = stale_drift_correction(
            pseudo_grad=upd.pseudo_grad,
            drift_state=drift_states[k],
            age=age,
            rho=rho,
        )
        corrected_updates.append(
            BufferedUpdate(
                client_id=upd.client_id,
                start_round=upd.start_round,
                pseudo_grad=corrected_grad,
                local_steps=upd.local_steps,
                local_lr=upd.local_lr,
                local_loss=upd.local_loss,
                num_samples=upd.num_samples,
            )
        )

    eta_t = age_aware_global_lr(
        base_lr=base_global_lr,
        ages=ages,
        age_threshold=age_threshold,
        decay=age_decay,
    )

    delta_t = aggregate_pseudogradient(corrected_updates)
    next_global = global_model_update(global_state, delta_t, eta_t)
    next_global_avg_grad = update_global_average_gradient(
        previous=global_avg_grad,
        selected_updates=corrected_updates,
        mu=global_grad_momentum,
    )

    for k in selected:
        participation_count[k] = participation_count.get(k, 0) + 1
        del send_buffer[k]

    return next_global, next_global_avg_grad, eta_t, ages


# ---------------------------------------------------------------------
# Recommended server-round skeleton
# ---------------------------------------------------------------------

def run_server_round(
    *,
    t: int,
    global_state: Mapping[str, torch.Tensor],
    global_avg_grad: Mapping[str, torch.Tensor],
    send_buffer: MutableMapping[int, BufferedUpdate],
    cluster_of: Mapping[int, int],
    all_client_ids: Sequence[int],
    participation_count: MutableMapping[int, int],
    remaining_compute: Mapping[int, float],
    estimated_comm_latency: Mapping[int, float],
    alpha_comm: Mapping[int, float],
    drift_states: Mapping[int, Mapping[str, torch.Tensor]],
    R: int,
    kappa: int,
    base_global_lr: float,
    age_threshold: float,
    age_decay: float,
    rho: float,
    global_grad_momentum: float,
    total_bandwidth: float,
) -> Tuple[TensorDict, TensorDict, List[int], Dict[int, float], float]:
    """
    One paper-aligned ES iteration.

    Notes:
    - If selected non-ready clients exist, their updates must be made available by
      the event simulator after their remaining local computation finishes.
      This function therefore expects send_buffer to contain the actual updates
      before aggregation. In a full simulator, schedule -> advance time -> collect
      finished selected updates -> call aggregation.
    """

    # 1) Keep valid buffered updates; do NOT clear all unselected updates.
    discard_stale_buffered_updates(send_buffer, t, kappa)

    ready_ids = set(send_buffer.keys())
    nonready_ids = [k for k in all_client_ids if k not in ready_ids]

    local_losses = {k: send_buffer[k].local_loss for k in ready_ids}

    # 2) Cluster-balanced / participation-aware scheduling.
    selected = cluster_balanced_schedule(
        cluster_of=cluster_of,
        ready_ids=ready_ids,
        nonready_ids=nonready_ids,
        local_losses=local_losses,
        participation_count=participation_count,
        remaining_compute=remaining_compute,
        estimated_comm_latency=estimated_comm_latency,
        R=R,
    )

    # If this simple round function is called before an event simulator has made
    # waiting-client updates available, aggregate the ready subset only.
    selected_ready = [k for k in selected if k in send_buffer]
    if not selected_ready:
        return clone_state(global_state), clone_state(global_avg_grad), [], {}, 0.0

    # 3) Bandwidth allocation. For selected-ready only, H_cp is usually zero.
    bw, H_t = latency_aware_bandwidth_allocation(
        selected=selected_ready,
        total_bandwidth=total_bandwidth,
        alpha=alpha_comm,
        remaining_compute={k: 0.0 for k in selected_ready},
    )

    # 4) Age-dependent correction before aggregation.
    corrected_updates: List[BufferedUpdate] = []
    ages: List[int] = []

    for k in selected_ready:
        upd = send_buffer[k]
        age = update_age(t, upd.start_round)
        ages.append(age)

        corrected = stale_drift_correction(
            pseudo_grad=upd.pseudo_grad,
            drift_state=drift_states[k],
            age=age,
            rho=rho,
        )

        corrected_updates.append(
            BufferedUpdate(
                client_id=upd.client_id,
                start_round=upd.start_round,
                pseudo_grad=corrected,
                local_steps=upd.local_steps,
                local_lr=upd.local_lr,
                local_loss=upd.local_loss,
                num_samples=upd.num_samples,
            )
        )

    # 5) Paper Eq. (6): age-aware global learning rate.
    eta_t = age_aware_global_lr(
        base_lr=base_global_lr,
        ages=ages,
        age_threshold=age_threshold,
        decay=age_decay,
    )

    # 6) Paper Eq. (4)-(5): weighted pseudo-gradient aggregation.
    delta_t = aggregate_pseudogradient(corrected_updates)
    next_global = global_model_update(global_state, delta_t, eta_t)

    # 7) Paper Eq. (14): global average gradient.
    next_global_avg_grad = update_global_average_gradient(
        previous=global_avg_grad,
        selected_updates=corrected_updates,
        mu=global_grad_momentum,
    )

    # 8) Participation counts and send-buffer maintenance.
    for k in selected_ready:
        participation_count[k] = participation_count.get(k, 0) + 1
        del send_buffer[k]

    return next_global, next_global_avg_grad, selected_ready, bw, H_t
