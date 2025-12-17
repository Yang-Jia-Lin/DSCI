# functions/Objective.py

from Src.Objective.compute_P import compute_layer_exit_probs
from Src.Objective.compute_latency import compute_total_latency
from Src.Objective.compute_accuracy import compute_expected_accuracy


def objective(X, Y, F_e, F_c, paras):

    # 1.Exit probabilities
    P = compute_layer_exit_probs(Y, paras)

    # 2.Delay
    latency_vec = compute_total_latency(X, P, F_e, F_c, paras)
    # print("latency is ", latency_vec)
    latency = sum(latency_vec)

    # 3.Accuracy
    acc_vec = compute_expected_accuracy(Y, P, paras)
    # print("accuracy is ", acc_vec)
    acc = sum(acc_vec)

    # return paras.beta * latency_vec - paras.alpha * acc_vec
    return paras.alpha * acc - paras.beta * latency