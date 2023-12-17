import numpy as np


def compute_accuracy(HDC_cont_test, Y_test, centroid, bias):
    Acc = 0
    for i in range(Y_test.shape[0]):
        # compute LS-SVM response
        response = (np.inner(centroid, HDC_cont_test[i]) + bias) >= 0

        # Give labels +1 and -1
        response = 1 if response else -1

        if response == Y_test[i]:  # I changed quite some stuff in this function compared to the original
            Acc += 1
    return Acc/Y_test.shape[0]