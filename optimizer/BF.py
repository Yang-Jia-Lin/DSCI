# optimizer/BF.py
import numpy as np
import math
from typing import Tuple, Dict, Any, List

from objective.compute_P import compute_layer_exit_probs
from objective.compute_accuracy import compute_expected_accuracy
from objective.objective import objective


def _generate_all_valid_X_rows(m: int) -> np.ndarray:
    """
    Generate all valid X rows with exactly two 1s (C(m,2)).
    Return: (K, m) int8
    """
    rows = []
    for a in range(m):
        for b in range(a + 1, m):
            x = np.zeros(m, dtype=np.int8)
            x[a] = 1
            x[b] = 1
            rows.append(x)
    return np.stack(rows, axis=0)


def _get_cut_points_from_xrow(x_row: np.ndarray) -> Tuple[int, int]:
    """Return sorted indices of ones. Assumes exactly two ones."""
    idx = np.where(x_row > 0.5)[0]
    if len(idx) != 2:
        raise ValueError(f"X row must contain exactly two 1s, got {len(idx)}")
    return int(idx[0]), int(idx[1])


def _tau_grid(step: float = 0.01) -> np.ndarray:
    """0.00~1.00 with fixed 2 decimals."""
    k = int(round(1.0 / step))
    return np.array([round(i * step, 2) for i in range(k + 1)], dtype=np.float64)


def _build_y_row(m: int, exit_layers: Tuple[int, int], t1: float, t2: float) -> np.ndarray:
    """Other layers fixed to 1; only two exit layers are adjustable."""
    y = np.ones(m, dtype=np.float64)
    e1, e2 = exit_layers
    y[e1] = t1
    y[e2] = t2
    return y


def _precompute_P_acc_cache(paras, step: float = 0.01) -> Dict[Tuple[int, int], Dict[str, Any]]:
    """
    Cache mapping (i1,i2) -> {P_row, acc_scalar, PS, PCS}
    Where:
      P_row: (m,)
      acc_scalar: float
      PS[j] = sum_{k<j} P[k]  (j=0..m)
      PCS[j] = sum_{k<j} P[k] * prefC[k] (j=0..m)
    """
    m = paras.m
    exit_layers = tuple(paras.E)
    assert len(exit_layers) == 2, "This BF expects exactly 2 early-exit layers."

    C = np.asarray(paras.C, dtype=np.float64)
    prefC = np.zeros(m + 1, dtype=np.float64)
    prefC[1:] = np.cumsum(C[:m], dtype=np.float64)

    grid = _tau_grid(step)
    cache: Dict[Tuple[int, int], Dict[str, Any]] = {}
    cache[("prefC", "prefC")] = {"prefC": prefC, "grid": grid}

    # Compute using shape (1,m) if possible; fallback to (n,m).
    def _compute_for_yrow(yrow: np.ndarray) -> Tuple[np.ndarray, float]:
        try:
            Y1 = yrow.reshape(1, m)
            P = compute_layer_exit_probs(Y1, paras)
            acc_vec = compute_expected_accuracy(Y1, P, paras)
            return np.asarray(P[0], dtype=np.float64), float(acc_vec[0])
        except Exception:
            YN = np.tile(yrow.reshape(1, m), (paras.n, 1))
            P = compute_layer_exit_probs(YN, paras)
            acc_vec = compute_expected_accuracy(YN, P, paras)
            return np.asarray(P[0], dtype=np.float64), float(acc_vec[0])

    # Full cache: 101*101 = 10201 entries (one-time cost)
    for i1, t1 in enumerate(grid):
        for i2, t2 in enumerate(grid):
            yrow = _build_y_row(m, exit_layers, t1, t2)
            P_row, acc_scalar = _compute_for_yrow(yrow)

            PS = np.zeros(m + 1, dtype=np.float64)
            PS[1:] = np.cumsum(P_row, dtype=np.float64)

            PCS = np.zeros(m + 1, dtype=np.float64)
            PCS[1:] = np.cumsum(P_row * prefC[:m], dtype=np.float64)

            cache[(i1, i2)] = {
                "P": P_row,
                "acc": acc_scalar,
                "PS": PS,
                "PCS": PCS,
            }

    return cache


def _compute_user_latency_consts_and_work(
    paras,
    i: int,
    cut0: int,
    cut1: int,
    PS: np.ndarray,
    PCS: np.ndarray,
    prefC: np.ndarray,
) -> Tuple[float, float, float]:
    """
    Return (const_i, b_i, c_i) s.t.
      latency_i = const_i + b_i / f_e_i + c_i / f_c_i
    with f_e_i, f_c_i in GHz (matching your code which multiplies by 1e9).
    """
    m = paras.m

    # local part: sum_{j=0..cut0} P[j]*prefC[j] / (f_u*1e9)
    f_u = float(np.asarray(paras.F_u).reshape(-1)[i])  # GHz
    local_work = PCS[cut0 + 1]  # sum_{j<cut0+1} P[j]*prefC[j]
    local_delay = local_work / (f_u * 1e9)

    # edge work: sum_{j=cut0..cut1-1} P[j]*(prefC[j]-prefC[cut0]) / 1e9
    edge_work = (PCS[cut1] - PCS[cut0]) - prefC[cut0] * (PS[cut1] - PS[cut0])
    b_i = max(0.0, edge_work / 1e9)

    # cloud work: sum_{j=cut1..m-1} P[j]*(prefC[j]-prefC[cut1]) / 1e9
    cloud_work = (PCS[m] - PCS[cut1]) - prefC[cut1] * (PS[m] - PS[cut1])
    c_i = max(0.0, cloud_work / 1e9)

    # u2e delay: d1 / R_i
    D = np.asarray(paras.D, dtype=np.float64).reshape(-1)
    d1 = float(D[cut0])
    h_i = float(np.asarray(paras.H_u).reshape(-1)[i])
    R_i = (float(paras.b_e) * 1e6) * math.log2(1.0 + (h_i * float(paras.G)) / float(paras.delta))
    u2e = d1 / R_i

    # e2c delay: d2 / (b_c*1e6)
    d2 = float(D[cut1])
    e2c = d2 / (float(paras.b_c) * 1e6)

    const_i = local_delay + u2e + e2c
    return const_i, b_i, c_i


def _solve_F_sqrt_allocation(b_vec: np.ndarray, c_vec: np.ndarray, paras) -> Tuple[np.ndarray, np.ndarray]:
    """
    Closed-form convex optimum for:
      minimize sum b_i/f_e_i s.t. sum f_e_i <= f_e_max, f_e_i>=0
    and similarly for cloud.
    """
    b = np.clip(b_vec.astype(np.float64), 0.0, None)
    c = np.clip(c_vec.astype(np.float64), 0.0, None)

    sb = np.sqrt(b)
    sc = np.sqrt(c)

    denom_b = float(np.sum(sb))
    denom_c = float(np.sum(sc))

    if denom_b > 0:
        F_e = float(paras.f_e_max) * sb / denom_b
    else:
        F_e = np.zeros_like(sb)

    if denom_c > 0:
        F_c = float(paras.f_c_max) * sc / denom_c
    else:
        F_c = np.zeros_like(sc)

    return F_e.reshape(-1, 1), F_c.reshape(-1, 1)


def _safe_latency_sum(const_vec: np.ndarray, b_vec: np.ndarray, c_vec: np.ndarray, F_e: np.ndarray, F_c: np.ndarray) -> float:
    """
    latency = sum const + sum b_i/f_e_i + sum c_i/f_c_i
    with safe division (no divide-by-zero warnings, no NaNs).
    """
    fe = F_e.reshape(-1).astype(np.float64)
    fc = F_c.reshape(-1).astype(np.float64)

    b = b_vec.astype(np.float64)
    c = c_vec.astype(np.float64)

    edge_term = np.divide(b, fe, out=np.zeros_like(b), where=fe > 0)
    cloud_term = np.divide(c, fc, out=np.zeros_like(c), where=fc > 0)

    lat = float(np.sum(const_vec) + np.sum(edge_term) + np.sum(cloud_term))
    return lat


def optimize_BF(
    paras=None,
    max_iter: int = 10,
    restarts: int = 3,
    threshold_step: float = 0.01,
    tol: float = 1e-6,
    verbose: bool = True,
):
    """
    Simple BCD + local brute force baseline.
    - No dual variables, no step-size tuning.
    - Each user update: brute force over all X rows (C(m,2)) and two-exit thresholds (10201).
    - Evaluate global objective using the SAME form: alpha*sum(acc) - beta*sum(latency)
      where latency is computed from (const + b/f_e + c/f_c) to avoid calling objective() 10^8 times.

    Returns:
      best_val, best_sol=(X, Y, F_e, F_c), history(list of best values per outer iter)
    """
    if paras is None:
        from __main__ import paras as _paras
        paras = _paras

    n, m = paras.n, paras.m
    exit_layers = tuple(paras.E)
    assert len(exit_layers) == 2, "This BF implementation assumes exactly 2 early-exit layers."
    e1, e2 = exit_layers

    # Precompute threshold cache: P, acc, PS, PCS for every (t1,t2)
    cache = _precompute_P_acc_cache(paras, step=threshold_step)
    prefC = cache[("prefC", "prefC")]["prefC"]
    grid = cache[("prefC", "prefC")]["grid"]

    # Precompute all X candidates
    X_candidates = _generate_all_valid_X_rows(m)  # (K,m)
    K = X_candidates.shape[0]

    best_overall_val = -float("inf")
    best_overall_sol = None
    best_overall_hist: List[float] = []

    rng = np.random.default_rng()

    for r in range(restarts):
        # ---- 1) Random init ----
        X = np.zeros((n, m), dtype=np.int8)
        Y = np.ones((n, m), dtype=np.float64)

        # Random X per user
        idxs = rng.integers(low=0, high=K, size=n)
        for i in range(n):
            X[i] = X_candidates[idxs[i]]

        # Random thresholds per user (two exits)
        t_idx = rng.integers(low=0, high=len(grid), size=(n, 2))
        for i in range(n):
            Y[i, e1] = grid[t_idx[i, 0]]
            Y[i, e2] = grid[t_idx[i, 1]]

        # Precompute per-user (const, b, c, acc) for current state
        const_vec = np.zeros(n, dtype=np.float64)
        b_vec = np.zeros(n, dtype=np.float64)
        c_vec = np.zeros(n, dtype=np.float64)
        acc_vec = np.zeros(n, dtype=np.float64)

        # Also store current tau indices per user for quick cache lookup
        tau_idx_cur = [(int(t_idx[i, 0]), int(t_idx[i, 1])) for i in range(n)]

        for i in range(n):
            i1, i2 = tau_idx_cur[i]
            PS = cache[(i1, i2)]["PS"]
            PCS = cache[(i1, i2)]["PCS"]
            acc_vec[i] = cache[(i1, i2)]["acc"]

            cut0, cut1 = _get_cut_points_from_xrow(X[i])
            const_i, b_i, c_i = _compute_user_latency_consts_and_work(paras, i, cut0, cut1, PS, PCS, prefC)
            const_vec[i], b_vec[i], c_vec[i] = const_i, b_i, c_i

        # Evaluate initial objective
        F_e, F_c = _solve_F_sqrt_allocation(b_vec, c_vec, paras)
        lat = _safe_latency_sum(const_vec, b_vec, c_vec, F_e, F_c)
        val = float(paras.alpha * np.sum(acc_vec) - paras.beta * lat)

        # safety: avoid NaN poisoning
        if not np.isfinite(val):
            # restart this run
            if verbose:
                print(f"[BF-BCD] restart={r+1}/{restarts} init_obj is not finite, restarting.")
            continue

        hist = [val]

        if verbose:
            print(f"[BF-BCD] restart={r+1}/{restarts} init_obj={val:.6f}")

        # ---- 2) BCD iterations ----
        for it in range(max_iter):
            improved_any = False
            order = rng.permutation(n)

            for u in order:
                # Save current user state
                x_old = X[u].copy()
                y_old = Y[u].copy()
                const_old, b_old, c_old, acc_old = const_vec[u], b_vec[u], c_vec[u], acc_vec[u]
                i1_old, i2_old = tau_idx_cur[u]

                best_local_val = val
                best_local_x = x_old
                best_local_y = y_old
                best_local_const = const_old
                best_local_b = b_old
                best_local_c = c_old
                best_local_acc = acc_old
                best_local_tau = (i1_old, i2_old)

                # Brute force this user's X and two thresholds
                for k in range(K):
                    x_cand = X_candidates[k]
                    cut0, cut1 = _get_cut_points_from_xrow(x_cand)

                    for i1 in range(len(grid)):
                        for i2 in range(len(grid)):
                            PS = cache[(i1, i2)]["PS"]
                            PCS = cache[(i1, i2)]["PCS"]
                            acc_u = cache[(i1, i2)]["acc"]

                            const_u, b_u, c_u = _compute_user_latency_consts_and_work(
                                paras, u, cut0, cut1, PS, PCS, prefC
                            )

                            # Update vectors (only user u changes)
                            const_tmp = const_vec.copy()
                            b_tmp = b_vec.copy()
                            c_tmp = c_vec.copy()
                            acc_tmp = acc_vec.copy()

                            const_tmp[u] = const_u
                            b_tmp[u] = b_u
                            c_tmp[u] = c_u
                            acc_tmp[u] = acc_u

                            # Solve optimal F analytically
                            F_e_tmp, F_c_tmp = _solve_F_sqrt_allocation(b_tmp, c_tmp, paras)
                            lat_tmp = _safe_latency_sum(const_tmp, b_tmp, c_tmp, F_e_tmp, F_c_tmp)

                            val_tmp = float(paras.alpha * np.sum(acc_tmp) - paras.beta * lat_tmp)

                            # Skip bad numerical cases
                            if not np.isfinite(val_tmp):
                                continue

                            if val_tmp > best_local_val + tol:
                                best_local_val = val_tmp
                                best_local_x = x_cand.copy()

                                y_cand = np.ones(m, dtype=np.float64)
                                y_cand[e1] = grid[i1]
                                y_cand[e2] = grid[i2]
                                best_local_y = y_cand

                                best_local_const = const_u
                                best_local_b = b_u
                                best_local_c = c_u
                                best_local_acc = acc_u
                                best_local_tau = (i1, i2)

                # Apply best update for user u
                if best_local_val > val + tol:
                    improved_any = True
                    val = best_local_val

                    X[u] = best_local_x
                    Y[u] = best_local_y
                    const_vec[u] = best_local_const
                    b_vec[u] = best_local_b
                    c_vec[u] = best_local_c
                    acc_vec[u] = best_local_acc
                    tau_idx_cur[u] = best_local_tau

                else:
                    # keep old
                    X[u] = x_old
                    Y[u] = y_old
                    const_vec[u], b_vec[u], c_vec[u], acc_vec[u] = const_old, b_old, c_old, acc_old
                    tau_idx_cur[u] = (i1_old, i2_old)

            hist.append(val)
            if verbose:
                print(f"[BF-BCD] restart={r+1} iter={it+1}/{max_iter} obj={val:.6f}")

            if not improved_any:
                if verbose:
                    print(f"[BF-BCD] restart={r+1} converged (no improvement).")
                break

        # ---- 3) Final primal evaluation using your original objective() for safety ----
        F_e, F_c = _solve_F_sqrt_allocation(b_vec, c_vec, paras)
        val_check = float(objective(X.astype(float), Y, F_e, F_c, paras))

        if verbose:
            print(f"[BF-BCD] restart={r+1} final_obj(check)={val_check:.6f}")

        if val_check > best_overall_val:
            best_overall_val = val_check
            best_overall_sol = (X.astype(float).copy(), Y.copy(), F_e.copy(), F_c.copy())
            best_overall_hist = hist

    return best_overall_val, best_overall_sol, best_overall_hist
