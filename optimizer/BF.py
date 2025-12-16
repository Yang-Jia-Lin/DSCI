import numpy as np
import itertools
from objective.objective import objective
from objective.compute_P import compute_layer_exit_probs


def optimize_BF(paras, max_iter=5, y_step=0.05):
    """
    Brute Force Solver using Block Coordinate Descent.

    Args:
        paras: Parameters object.
        max_iter: Number of BCD iterations (rounds).
        y_step: Step size for Y grid search (e.g., 0.05 or 0.01).
                0.05 = 20*20 = 400 combinations per X.
                0.01 = 100*100 = 10000 combinations per X.
    """
    n, m = paras.n, paras.m

    # --- 1. Initialization ---
    # Randomly initialize valid X and Y
    # X: 0-1 matrix, we start with all zeros (all local) or random valid
    X = np.zeros((n, m))
    # Y: Continuous 0-1, start with 0.5
    Y = np.ones((n, m)) * 0.5
    # Initialize F with equal allocation
    F_e = np.ones((n, 1)) * (paras.f_e_max / n)
    F_c = np.ones((n, 1)) * (paras.f_c_max / n)

    # Find the indices of the early exit layers (assuming 2 exits as described)
    # If paras.E is a list of booleans or indices, convert to indices
    exit_indices = [i for i, is_exit in enumerate(paras.E) if is_exit]
    # If explicit indices are not in paras, assuming standard 2-exit structure for now:
    if len(exit_indices) < 2:
        # Fallback based on your snippet "RL_Y_opt[:,[57,103]]"
        exit_indices = [57, 103]
        print(f"Warning: Using hardcoded exit indices {exit_indices} for BF.")

    # Pre-calculate all valid X combinations (Cut points)
    # A valid X row has at most 2 ones.
    # Logic: 0 ones (Local only), 1 one (Local->Edge or Local->Cloud?), 2 ones (Local->Edge->Cloud)
    # We represent state by cut locations (k1, k2).
    valid_x_rows = generate_all_valid_x_rows(m)

    # Pre-calculate all valid Y pairs (Grid Search)
    y_values = np.arange(0, 1.0 + y_step / 2, y_step)  # e.g. [0.0, 0.05, ..., 1.0]
    valid_y_pairs = list(itertools.product(y_values, repeat=len(exit_indices)))

    best_global_val = -np.inf
    history = []

    print(f"--- Starting BF Optimization (BCD) ---")
    print(f"Users: {n}, Layers: {m}, X_candidates: {len(valid_x_rows)}, Y_pairs: {len(valid_y_pairs)}")

    # --- 2. BCD Loop ---
    for it in range(max_iter):
        improved_this_round = False

        # Shuffle update order to prevent bias
        user_order = np.random.permutation(n)

        for u in user_order:

            # Current best for this user
            current_user_best_val = -np.inf
            current_user_best_config = None  # (x_row, y_row)

            # Store original state to revert if needed (though we usually update greedily)
            # original_X_row = X[u].copy()
            # original_Y_row = Y[u].copy()

            # --- A. Iterate over all valid X structures ---
            for x_row_cand in valid_x_rows:

                # --- B. Iterate over all Y grid points ---
                for y_pair in valid_y_pairs:
                    # Construct Candidate Y Row
                    y_row_cand = np.ones(m)  # Default 1.0 (pass through)
                    for idx, val in zip(exit_indices, y_pair):
                        y_row_cand[idx] = val

                    # Temporarily update Global Matrices to calculate Resource Allocation
                    X[u] = x_row_cand
                    Y[u] = y_row_cand

                    # --- C. Analytical Solution for F ---
                    # Based on current X and Y, calculate optimal F_e, F_c for ALL users
                    # This is fast because it's a closed-form formula
                    F_e_opt, F_c_opt = solve_optimal_F_analytical(X, Y, paras)

                    # --- D. Evaluate Objective ---
                    val = objective(X, Y, F_e_opt, F_c_opt, paras)

                    if val > current_user_best_val:
                        current_user_best_val = val
                        current_user_best_config = (x_row_cand.copy(), y_row_cand.copy())

            # Update Global State for User u with the best found locally
            if current_user_best_val > best_global_val:  # Check against global best just for tracking
                pass

                # Apply the best local move to the global state
            X[u] = current_user_best_config[0]
            Y[u] = current_user_best_config[1]
            # Update F to match the new X, Y
            F_e, F_c = solve_optimal_F_analytical(X, Y, paras)
            print(f"第{it}轮，第{user_order}个用户：{current_user_best_val}")

        # End of Round Evaluation
        current_global_val = objective(X, Y, F_e, F_c, paras)
        history.append(current_global_val)

        print(f"Round {it + 1}/{max_iter} Best Obj: {current_global_val:.5f}")

        if current_global_val > best_global_val + 1e-6:
            best_global_val = current_global_val
            improved_this_round = True

        if not improved_this_round:
            print("Converged.")
            break

    return best_global_val, (X, Y, F_e, F_c), history


def generate_all_valid_x_rows(m):
    """
    Generates all valid X vectors (size m).
    Constraints: At most 2 ones.
    1 means 'offload to next stage'.
    Sequence: Local -> Edge -> Cloud.
    """
    candidates = []

    # Case 1: All Zeros (All Local Computing)
    candidates.append(np.zeros(m))

    # Case 2: One '1' at layer k (Local [0,k] -> Edge [k+1, m])
    # Or Local -> Cloud? Usually assumes sequential Local->Edge->Cloud
    # Assuming X[k]=1 means cut after layer k.
    for k in range(m):
        x = np.zeros(m)
        x[k] = 1
        candidates.append(x)

    # Case 3: Two '1's at k1, k2 (Local -> Edge -> Cloud)
    for k1 in range(m):
        for k2 in range(k1 + 1, m):
            x = np.zeros(m)
            x[k1] = 1
            x[k2] = 1
            candidates.append(x)

    return candidates


def solve_optimal_F_analytical(X, Y, paras):
    """
    Calculates Optimal F_e and F_c using Closed-form solution (Sqrt Law).
    Objective part related to F: Min sum(W_i / F_i)
    Constraint: sum(F_i) <= F_max
    Solution: F_i propto sqrt(W_i)
    """
    n, m = X.shape

    # 1. Calculate Expected Workload (Compute Cycles) for Edge and Cloud
    # We need to replicate the logic of how much data flows to Edge vs Cloud
    # This depends on X (partition) and Y (early exit probs)

    # Calculate Probabilities of reaching each layer
    # P[u, l] is prob that user u executes layer l
    P = compute_layer_exit_probs(Y, paras)

    W_edge = np.zeros(n)
    W_cloud = np.zeros(n)

    # Parse X to find Edge and Cloud layers for each user
    for u in range(n):
        cuts = np.where(X[u] == 1)[0]

        edge_start, edge_end = m, m  # Default: No Edge
        cloud_start, cloud_end = m, m  # Default: No Cloud

        if len(cuts) == 0:
            # All Local
            pass
        elif len(cuts) == 1:
            # Local -> Edge (Assuming single cut means rest is Edge, or checking paras logic)
            # Common assumption: Cut 1 moves to Edge. No second cut means no Cloud.
            edge_start = cuts[0] + 1
            edge_end = m
        elif len(cuts) == 2:
            # Local -> Edge -> Cloud
            edge_start = cuts[0] + 1
            edge_end = cuts[1] + 1  # Edge ends at second cut
            cloud_start = cuts[1] + 1
            cloud_end = m

        # Accumulate Workload (Expected Cycles)
        # paras.C is vector of cycles per layer
        if edge_start < m:
            # Vectorized sum for this user segment
            # W = Sum( P[layer] * Cost[layer] )
            W_edge[u] = np.sum(P[u, edge_start:edge_end] * paras.C[edge_start:edge_end])

        if cloud_start < m:
            W_cloud[u] = np.sum(P[u, cloud_start:cloud_end] * paras.C[cloud_start:cloud_end])

    # 2. Apply Square Root Allocation Law
    # Avoid division by zero if W is 0
    sqrt_W_edge = np.sqrt(W_edge)
    sqrt_W_cloud = np.sqrt(W_cloud)

    sum_sqrt_edge = np.sum(sqrt_W_edge)
    sum_sqrt_cloud = np.sum(sqrt_W_cloud)

    # Calculate F
    # If sum is 0 (no one uses edge), allocate 0 (or distribute evenly to avoid NaN)
    if sum_sqrt_edge > 1e-9:
        F_e = (sqrt_W_edge / sum_sqrt_edge) * paras.f_e_max
    else:
        F_e = np.zeros(n)

    if sum_sqrt_cloud > 1e-9:
        F_c = (sqrt_W_cloud / sum_sqrt_cloud) * paras.f_c_max
    else:
        F_c = np.zeros(n)

    # Reshape for consistency (n, 1)
    return F_e.reshape(n, 1), F_c.reshape(n, 1)