# functions/Objective.py

from Src.Objective.compute_P import compute_layer_exit_probs
from Src.Objective.compute_latency import compute_total_latency
from Src.Objective.compute_accuracy import compute_expected_accuracy


def objective(X, Y, F_e, F_c, paras):

    # 1.Exit probabilities
    P = compute_layer_exit_probs(Y, paras)

    # 2.Delay
    latency_vec = compute_total_latency(X, P, F_e, F_c, paras)
    latency = sum(latency_vec)

    # 3.Accuracy
    acc_vec = compute_expected_accuracy(Y, P, paras)
    acc = sum(acc_vec)

    return paras.alpha * acc - paras.beta * latency


# ==========================================
# Test Block for Objective
# ==========================================
if __name__ == "__main__":

    from Src.paras import *
    from Src.Utils.parsing_data import parsing_rate_and_acc

    print("\n" + "=" * 20 + " Objective Test " + "=" * 20)

    # -------------------------------------------------
    # 1. 初始化参数
    # -------------------------------------------------
    paras = Paras(
        # Users
        n=NUM_USERS,
        # Model
        m=NUM_LAYERS,
        E=EARLY_EXIT_LAYERS,
        D=DATA_SIZE_LAYERS,
        C=COMPUTE_SIZE_LAYERS,
        # Resource
        F_u=USER_FREQs,
        f_e_max=EDGE_MAX_FREQ,
        f_c_max=CLOUD_MAX_FREQ,
        H_u=CHANNEL_GAINS_USERS,
        b_e=BANDWIDTH_EDGE,
        b_c=BANDWIDTH_CLOUD,
        G=BASE_STATION_POWER,
        delta=NOISE_POWER,
        # Weights
        alpha=1,
        beta=20
    )
    paras.rates, paras.accs = parsing_rate_and_acc(paras)

    n, m = paras.n, paras.m

    # -------------------------------------------------
    # 2. 构造变量
    # -------------------------------------------------
    X = np.zeros((n, m))
    # 给 user 0 一个典型切点，避免全零
    if m > 2:
        X[0, 0] = 1
        X[0, m // 2] = 1

    Y = np.ones((n, m))
    # 给 user 0 两个非 1 阈值，制造早退差异
    if m > 100:
        Y[0, 57] = 0.9
        Y[0, 103] = 0.8

    F_e = np.ones((n, 1)) * (paras.f_e_max / n)
    F_c = np.ones((n, 1)) * (paras.f_c_max / n)

    # -------------------------------------------------
    # 3. Exit Probabilities
    # -------------------------------------------------
    print("\n[1] Exit Probabilities")
    P = compute_layer_exit_probs(Y, paras)
    print("P shape:", P.shape)

    # 打印 user 0 的主要概率质量
    P0 = P[0]
    topk = np.argsort(-P0)[:10]
    print("User 0 top-10 exit probabilities:")
    for j in topk:
        print(f"  layer {j:4d}: P = {P0[j]:.6e}")
    print(f"sum(P[0]) = {np.sum(P0):.6f}")

    # -------------------------------------------------
    # 4. Latency
    # -------------------------------------------------
    print("\n[2] Latency")
    latency_vec = compute_total_latency(X, P, F_e, F_c, paras)
    latency = float(np.sum(latency_vec))

    for i in range(min(3, n)):
        print(f"User {i} latency = {latency_vec[i]:.6e} s")

    print(f"Total latency (sum over users) = {latency:.6e} s")

    # -------------------------------------------------
    # 5. Accuracy
    # -------------------------------------------------
    print("\n[3] Accuracy")
    acc_vec = compute_expected_accuracy(Y, P, paras)
    acc = float(np.sum(acc_vec))

    for i in range(min(3, n)):
        print(f"User {i} expected accuracy = {acc_vec[i]:.6f}")

    print(f"Total expected accuracy (sum over users) = {acc:.6f}")

    # -------------------------------------------------
    # 6. Objective Breakdown
    # -------------------------------------------------
    print("\n[4] Objective Breakdown")
    weighted_acc = paras.alpha * acc
    weighted_latency = paras.beta * latency
    obj = weighted_acc - weighted_latency

    print(f"alpha * acc   = {paras.alpha} * {acc:.6f} = {weighted_acc:.6f}")
    print(f"beta  * delay = {paras.beta} * {latency:.6e} = {weighted_latency:.6e}")
    print("-" * 50)
    print(f"Objective value = {obj:.6e}")

    # -------------------------------------------------
    # 7. 对照 objective() 函数
    # -------------------------------------------------
    print("\n[5] Sanity Check with objective()")
    obj_func = objective(X, Y, F_e, F_c, paras)
    print(f"objective() returns = {obj_func:.6e}")

    diff = abs(obj - obj_func)
    if diff < 1e-9:
        print("✅ Test passed: manual breakdown matches objective().")
    else:
        print(f"❌ Test failed: diff = {diff:.3e}")

    print("\n" + "=" * 60)
