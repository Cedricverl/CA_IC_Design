"""
Design of a Hyperdimensional Computing Circuit for Bio-signal Classification via Nelder-Mead optimization
and LS-SVM Training.
*HDC library*
Computer-Aided IC Design (B-KUL-H05D7A)
ir. Ali Safa, ir. Sergio Massaioli, Prof. Georges Gielen (MICAS-IMEC-KU Leuven)
(Author: A. Safa)
"""

import numpy as np
import numpy.matlib
from sklearn.utils import shuffle


# Receives the HDC encoded test set "HDC_cont_test" and test labels "Y_test"
# Computes test accuracy w.r.t. the HDC prototypes (centroids) and the biases found at training time
def compute_accuracy(HDC_cont_test, Y_test, centroid, bias):
    # Acc = 0
    # for i in range(Y_test.shape[0]):
    #     # compute LS-SVM response
    #     response = (np.inner(centroid, HDC_cont_test[i]) + bias) >= 0
    #
    #     # Give labels +1 and -1
    #     response = 1 if response else -1
    #
    #     if response == Y_test[i]:  # I changed quite some stuff in this function compared to the original
    #         Acc += 1

    responses = (np.inner(centroid, HDC_cont_test) + bias) >= 0
    responses = np.where(responses, 1, -1)
    Acc = sum(responses[0] == Y_test)
    # np.where(x > alpha_sp, 1, np.where(abs(x) <= alpha_sp, 0, -1))
    return (Acc/Y_test.shape[0]).astype(np.float16)


# Generates random binary matrices of -1 and +1
# when mode == 0, all elements are generated with the same statistics (probability of having +1 always = 0.5)
# when mode == 1, all the probability of having +1 scales with an input "key" i.e., when the inputs to the HDC encoded are coded
# on e.g., 8-bit, we have 256 possible keys
def lookup_generate(dim, n_keys, mode=1):
    table = np.random.rand(n_keys, dim)  # random matrix with values between 0 and 1
    if mode == 0:
        table = np.where(table < 0.5, 1, -1)  # 50/50 to get -1 or 1
    else:
        for i in range(n_keys):
            p = i/(n_keys-1)  # p=probability for 1
            table[i, :] = np.where(table[i, :] < p, 1, -1)
    return table.astype(np.int8)
    

# Performs "part" of the HDC encoding (only input encoding, position encoding and bundling), without the thresholding at the end.
# Returns H = bundle_along_features(P.L)
# img is the input feature vector to be encoded # int between 0 and 255?
# position_table is the random matrix of mode == 0
# grayscale_table is the input encoding LUT of mode == 1
# dim is the HDC dimensionality D
def encode_HDC_RFF(img, position_table, grayscale_table, dim):
    # Get the input-encoding and XOR-ing result: (own, faster parallel implementation)
    hv = grayscale_table[img, :]
    container = hv*position_table  # XOR
    img_hv = np.sum(container, axis=0)  # bundling without the cyclic step yet
    return img_hv.astype(np.int8)


# Train the HDC circuit on the training set : (Y_train, HDC_cont_train)
# n_class: number of classes
# N_train: number of data points in training set
# gamma: LS-SVM regularization
# D_b: number of bit for HDC prototype quantization
def train_HDC_RFF(n_class, N_train, Y_train, HDC_cont_train, gamma, D_b):
    centroids = []
    centroids_q = []
    biases_q = []
    biases = []
    # for cla in range(n_class):
    # The steps below implement the LS-SVM training, check out the course notes, we are just implementing that
    # Beta.alpha = L -> alpha (that we want)
    # Fill Beta:
    Beta = np.zeros((N_train+1, N_train+1))  # LS-SVM regression matrix

    # help-matrix containing all products of Y_train
    Y_train_mult = np.array([Y_train*i for i in Y_train])

    # Make Omega-matrix using matrix mul and help-matrix
    Omega = np.matmul(HDC_cont_train, HDC_cont_train.T)
    Omega = np.multiply(Omega, Y_train_mult)

    # Fill up Beta-matrix
    Beta[0, 1:] = Y_train
    Beta[1:, 0] = Y_train
    Beta[1:, 1:] = Omega + (gamma**-1)*np.eye(N_train)

    # Target vector L:
    L = np.ones(N_train+1)
    L[0] = 0

    # Solve the system of equations to get the vector alpha:
    v = np.linalg.solve(Beta, L)
    alpha = v[1:]

    # Get HDC prototype for class cla, still in floating point (µ)
    final_HDC_centroid = np.dot(Y_train * alpha, HDC_cont_train)

    r_min = -2**(D_b-1)
    r_max = 2**(D_b-1)-1

    # Amplification factor for the LS-SVM bias
    fact = min(abs(r_min/final_HDC_centroid.min()), abs(r_max/final_HDC_centroid.max()))

    # Quantize HDC prototype to D_b-bit
    final_HDC_centroid_q = np.round(final_HDC_centroid*fact).astype(np.int8)

    if np.max(np.abs(final_HDC_centroid)) == 0:
        print("Kernel matrix badly conditionned! Ignoring...")
        centroids_q.append(np.ones(final_HDC_centroid_q.shape))  # trying to manage badly conditioned matrices, do not touch
        biases_q.append(10000)
    else:
        centroids_q.append(final_HDC_centroid_q*1)
        biases_q.append(alpha[0]*fact)

    centroids.append(final_HDC_centroid*1)
    biases.append(alpha[0])

    return centroids, biases, centroids_q, biases_q


# Evaluate the Nelder-Mead cost F(x) over "Nbr_of_trials" trials
# (HDC_cont_all, LABELS) is the complete dataset with labels
# beta_ is the output accumulator increment of the HDC encoder (a float)
# bias_ are the random starting value of the output accumulators
# gamma is the LS-SVM regularization hyper-parameter
# alpha_sp is the encoding threshold
# n_class is the number of classes, N_train is the number training points, D_b the HDC prototype quantization bit width
# lambda_1, lambda_2 define the balance between Accuracy and Sparsity: it returns lambda_1*Acc + lambda_2*Sparsity
def evaluate_F_of_x(Nbr_of_trials, HDC_cont_all, beta_, bias_, gamma, alpha_sp, n_class, N_train, D_b, lambda_1, lambda_2, B_cnt, Y_train, Y_test):
    # local_avg = np.zeros(Nbr_of_trials)
    # local_avgre = np.zeros(Nbr_of_trials)
    # local_sparse = np.zeros(Nbr_of_trials)
    local_avg = None
    local_avgre = None
    loal_sparse = None
    t_mod = 2**B_cnt-1
    # Estimate F(x) over "Nbr_of_trials" trials
    for trial_ in range(Nbr_of_trials):
        # Removed for less fuzzy results
        # HDC_cont_all, LABELS = shuffle(HDC_cont_all, LABELS)  # Shuffle dataset for random train-test split

        HDC_cont_train_ = HDC_cont_all[:N_train]  # Take training set
        HDC_cont_train_cpy = HDC_cont_train_ * 1

        # Apply cyclic accumulation with biases and accumulation speed beta_
        HDC_cont_train_cpy = (HDC_cont_train_cpy*beta_).astype(np.uint8)    # we can do this because even if beta is very large, our accumulators
                                                                            # are defined on 8 bits so basically they will overflow each iteration
                                                                            # after the multiplication, if there were negative values it won't matter
                                                                            # since the accumulators are not working with ca2 and if (for example)
                                                                            # the result of all the sums * beta is -2, it would mean that the accumulator
                                                                            # will show 253, that's why we can cast it with uint
        
        HDC_cont_train_cpy = (HDC_cont_train_cpy + bias_).astype(np.uint8)                                       # not really with 8 bits we have a modulo operation, it's good
        # HDC_cont_train_cpy = np.mod(HDC_cont_train_cpy, 2**B_cnt-1)
        # HDC_cont_train_cpy &= t  # equivalent to np.mod(HDC_cont_train_cpy, 2**B_cnt-1)
        # HDC_cont_train_cpy = np.mod(HDC_cont_train_cpy, t_mod)

        # Ternary thresholding with threshold alpha_sp:
        # HDC_cont_train_cpy = vthreshold(HDC_cont_train_cpy, alpha_sp, B_cnt)
        x = HDC_cont_train_cpy - 2**(B_cnt-1)
        HDC_cont_train_cpy = np.where(x > alpha_sp, 1, np.where(abs(x) <= alpha_sp, 0, -1)).astype(np.int8)
        # Y_train = LABELS[:N_train]  # Labels have to be {-1, 1}
        # Y_train = Y_train.astype(int)

        # Train the HDC system to find the prototype hypervectors, _q means quantized
        centroids, biases, centroids_q, biases_q = train_HDC_RFF(n_class, N_train, Y_train, HDC_cont_train_cpy, gamma, D_b)
        
        # Do the same encoding steps with the test set
        HDC_cont_test_ = HDC_cont_all[N_train:]
        HDC_cont_test_cpy = HDC_cont_test_ * 1

        # Apply cyclic accumulation with biases and accumulation speed beta_
        # HDC_cont_test_cpy *= beta_
        # HDC_cont_test_cpy = (HDC_cont_test_cpy+bias_).astype(np.uint8)
        HDC_cont_test_cpy = (HDC_cont_test_cpy*beta_).astype(np.uint8) # 16 is needed because there could a bias = to 256 or near there that summed would give a wrong value
        HDC_cont_test_cpy = (HDC_cont_test_cpy + bias_).astype(np.uint8)
        # HDC_cont_test_cpy = np.mod(HDC_cont_test_cpy, t_mod)
        # HDC_cont_test_cpy &= t_mod  # equivalent to np.mod(HDC_cont_train_cpy, 2**B_cnt-1)

        # Ternary thresholding with threshold alpha_sp:
        # HDC_cont_test_cpy = vthreshold(HDC_cont_test_cpy, alpha_sp, B_cnt)
        x = HDC_cont_test_cpy - 2**(B_cnt-1)
        HDC_cont_test_cpy = np.where(x > alpha_sp, 1, np.where(abs(x) <= alpha_sp, 0, -1)).astype(np.int8)
        # Y_test = LABELS[N_train:]
        # Y_test = Y_test.astype(int)
        
        # Compute accuracy and sparsity of the test set w.r.t the HDC prototypes
        Acc = compute_accuracy(HDC_cont_test_cpy, Y_test, centroids_q, biases_q)
        sparsity_HDC_centroid = np.array(centroids_q).flatten()
        nbr_zero = np.sum((sparsity_HDC_centroid == 0).astype(int))
        SPH = nbr_zero/(sparsity_HDC_centroid.shape[0])
        # local_avg[trial_] = lambda_1 * Acc + lambda_2 * SPH  # Cost F(x) is defined as 1 - (this quantity)
        # local_avgre[trial_] = Acc
        # local_sparse[trial_] = SPH
        local_avg = lambda_1 * Acc + lambda_2 * SPH  # Cost F(x) is defined as 1 - (this quantity)
        local_avgre = Acc
        local_sparse = SPH
    return local_avg, local_avgre, local_sparse
