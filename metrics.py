

def mse(y_real, y_pred):
    score = 0
    for real, pred in zip(y_real, y_pred):
        score += (real - pred) ** 2
    return score / len(y_real)


def precision(TP, FP):
    A = TP + FP
    if A == 0:
        return 100
    return (TP / A) * 100


def recall(TP, FN):
    A = TP + FN
    if A == 0:
        return 100
    return (TP / A) * 100


def accuracy(TP, TN, FP, FN):
    A = TP + TN
    B = TP + TN + FP + FN
    if B == 0:
        return 100
    return (A / B) * 100
