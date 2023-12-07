import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from os import mkdir
plt.close('all')


def compute_accuracy(HDC_cont_test, Y_test, centroids, biases):
    Acc = 0
    for i in range(Y_test.shape[0]):
        # final_HDC_centroid = centroids[cl]
        final_HDC_centroid = centroids

        # compute LS-SVM response
        response = (np.inner(final_HDC_centroid, HDC_cont_test[i]) + biases) >= 0

        # Give labels +1 and -1
        all_resp = 1 if response else -1

        if all_resp == Y_test[i]:  # I changed quite some stuff in this function compared to the original
            Acc += 1
    return Acc/Y_test.shape[0]


def threshold(x, t, B_cnt):
    x = x-2**(B_cnt-1)
    if x > t:
        return 1
    elif abs(x) <= t:
        return 0
    else:
        return -1


vthreshold = np.vectorize(threshold)


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
# def encode_HDC_RFF(img, position_table, grayscale_table, dim):
#     img_hv = np.zeros(dim, dtype=np.int16)
#     container = np.zeros((len(position_table), dim))
#     for pixel in range(len(position_table)):
#         # Get the input-encoding and XOR-ing result:
#
#         hv = grayscale_table[img[pixel], :] #LUT # -> INSERT YOUR CODE
#
#         # container[pixel, :] = hv*1
#         container[pixel, :] = hv*position_table[pixel, :]
#
#     img_hv = np.sum(container, axis=0) #bundling without the cyclic step yet
#     return img_hv
def encode_HDC_RFF(img, position_table, grayscale_table, dim):
    # Get the input-encoding and XOR-ing result: (own, faster parallel implementation)
    hv = grayscale_table[img, :]
    container = hv*position_table  # XOR
    img_hv = np.sum(container, axis=0)  # bundling without the cyclic step yet
    return img_hv


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
    Beta = np.zeros((N_train+1, N_train+1))  # LS-SVM regression matrix
    # Fill Beta:
    Omega = np.zeros((N_train, N_train))
    for i in range(Omega.shape[0]):
        for j in range(Omega.shape[1]):
            Omega[i, j] = Y_train[i]*Y_train[j]*np.inner(HDC_cont_train[i], HDC_cont_train[j])

    Beta[0, 1:N_train+1] = Y_train
    Beta[1:N_train+1, 0] = Y_train
    Beta[1:N_train+1, 1:N_train+1] = Omega + (gamma**-1)*np.eye(N_train)

    # Target vector L:
    L = np.ones(N_train+1)
    L[0] = 0
    # Solve the system of equations to get the vector alpha:
    v = np.linalg.solve(Beta, L)
    alpha = v[1:N_train+1]

    # Get HDC prototype for class cla, still in floating point (µ)
    # final_HDC_centroid = sum([Y_train[i]*alpha[i]*HDC_cont_train[i] for i in range(N_train)])
    final_HDC_centroid = np.zeros(HDC_cont_train.shape[1])
    for i in range(N_train):
        final_HDC_centroid += Y_train[i]*alpha[i]*HDC_cont_train[i]

    r_min = -2**(D_b-1)
    r_max = 2**(D_b-1)-1
    fact = min(abs(r_min/final_HDC_centroid.min()), abs(r_max/final_HDC_centroid.max()))

    final_HDC_centroid_q = final_HDC_centroid*fact
    final_HDC_centroid_q = np.round(final_HDC_centroid_q)

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
def evaluate_F_of_x(Nbr_of_trials, HDC_cont_all, LABELS, beta_, bias_, gamma, alpha_sp, n_class, N_train, D_b, lambda_1, lambda_2, B_cnt):
    local_avg = np.zeros(Nbr_of_trials)
    local_avgre = np.zeros(Nbr_of_trials)
    local_sparse = np.zeros(Nbr_of_trials)
    # Estimate F(x) over "Nbr_of_trials" trials
    for trial_ in range(Nbr_of_trials): 
        HDC_cont_all, LABELS = shuffle(HDC_cont_all, LABELS)  # Shuffle dataset for random train-test split
            
        HDC_cont_train_ = HDC_cont_all[:N_train, :]  # Take training set
        HDC_cont_train_cpy = HDC_cont_train_ * 1
        
        # Apply cyclic accumulation with biases and accumulation speed beta_
        HDC_cont_train_cpy *= beta_
        HDC_cont_train_cpy += bias_
        HDC_cont_train_cpy = np.mod(HDC_cont_train_cpy, 2**B_cnt-1)

        # Ternary thresholding with threshold alpha_sp:
        HDC_cont_train_cpy = vthreshold(HDC_cont_train_cpy, alpha_sp, B_cnt)

        # Y_train = 2*LABELS[:N_train] - 3  # Labels have to be {-1, 1}
        Y_train = LABELS[:N_train]  # Labels have to be {-1, 1}
        Y_train = Y_train.astype(int)
        
        # Train the HDC system to find the prototype hypervectors, _q means quantized
        centroids, biases, centroids_q, biases_q = train_HDC_RFF(n_class, N_train, Y_train, HDC_cont_train_cpy, gamma, D_b)

        # Compute accuracy and sparsity of the test set w.r.t the HDC prototypes
        Acc = compute_accuracy(HDC_cont_train_cpy, Y_train, centroids, biases)
        sparsity_HDC_centroid = np.array(centroids).flatten()
        nbr_zero = np.sum((sparsity_HDC_centroid == 0).astype(int))
        SPH = nbr_zero/(sparsity_HDC_centroid.shape[0])
        local_avg[trial_] = lambda_1 * Acc + lambda_2 * SPH  # Cost F(x) is defined as 1 - (this quantity)
        local_avgre[trial_] = Acc
        local_sparse[trial_] = SPH
        
    return local_avg, local_avgre, local_sparse



"""
1) HDC_RFF parameters: DO NOT TOUCH
"""
##################################   
# Replace the path "WISCONSIN/data.csv" with wathever path you have. Note, on Windows, you must put the "r" in r'C:etc..'
dataset_path = 'WISCONSIN/data.csv' 
##################################   
imgsize_vector = 30  # Each input vector has 30 features
n_class = 2
D_b = 8  # We target 4-bit HDC prototypes
B_cnt = 8
maxval = 256  # The input features will be mapped from 0 to 255 (8-bit)
D_HDC = 100  # HDC hypervector dimension
portion = 1  # 100% training data for this test file
Nbr_of_trials = 1  # Test accuracy averaged over Nbr_of_trials runs
N_tradeof_points = 1  # Number of tradeoff points - use 100 - original: 40
N_fine = int(N_tradeof_points*0.4)  # Number of tradeoff points in the "fine-grain" region - use 30
# Initialize the sparsity-accuracy hyperparameter search
lambda_fine = np.linspace(-0.2, 0.2, N_tradeof_points-N_fine)
lambda_sp = np.concatenate((lambda_fine, np.linspace(-1, -0.2, N_fine//2), np.linspace(0.2, 1, N_fine//2)))
N_tradeof_points = lambda_sp.shape[0]

"""
2) Load dataset: if it fails, replace the path "WISCONSIN/data.csv" with wathever 
path you have. Note, on Windows, you must put the "r" in r'C:etc..'
"""
DATASET = np.loadtxt(dataset_path, dtype=object, delimiter=',', skiprows=1)
X = DATASET[:, 2:].astype(float)
LABELS = DATASET[:, 1]
LABELS[LABELS == 'M'] = -1
LABELS[LABELS == 'B'] = 1
LABELS = LABELS.astype(float)
X = X.T / np.max(X, axis=1)
X, LABELS = shuffle(X.T, LABELS)
imgsize_vector = X.shape[1]
# N_train = int(X.shape[0]*portion)
N_train_list = []
ACCS_LIST = []
for N_train in range(1, 300, 20):
    """
    3) Generate HDC LUTs and bundle dataset
    """
    grayscale_table = lookup_generate(D_HDC, maxval, mode=1)  # Input encoding LUT
    position_table = lookup_generate(D_HDC, imgsize_vector, mode=0)  # weight for XOR-ing
    HDC_cont_all = np.zeros((X.shape[0], D_HDC))  # Will contain all "bundled" HDC vectors

    bias_ = np.random.randint(256, size=D_HDC)  # generate the random biases once

    for i in range(X.shape[0]):
        if i % 100 == 0:
            print(str(i) + "/" + str(X.shape[0]))
        HDC_cont_all[i, :] = encode_HDC_RFF(np.round((maxval - 1) * X[i, :]).astype(int), position_table, grayscale_table, D_HDC)

    print("HDC bundling finished...")

    """
    4) Nelder-Mead circuit optimization and HDC training
    """
    ##################################
    # Nelder-Mead parameters
    NM_iter = 20  # Maximum number of iterations
    STD_EPS = 0.002  # Threshold for early-stopping on standard deviation of the Simplex
    # Contraction, expansion,... coefficients:
    alpha_simp = 1 * 0.5
    gamma_simp = 2 * 0.6
    rho_simp = 0.5
    sigma_simp = 0.5
    ##################################

    ACCS = np.zeros(N_tradeof_points)
    SPARSES = np.zeros(N_tradeof_points)
    load_simplex = False  # Keep it to true in order to have somewhat predictive results
    for optimalpoint in range(N_tradeof_points):
        print("Progress: " + str(optimalpoint+1) + "/" + str(N_tradeof_points))
        # F(x) = 1 - (lambda_1 * Accuracy + lambda_2 * Sparsity) : TO BE MINIMIZED by Nelder-Mead
        lambda_1 = 1  # Weight of Accuracy contribution in F(x)
        # lambda_2 = lambda_sp[optimalpoint]  # Weight of Sparsity contribution in F(x): varies!
        lambda_2 = 0
        # Initialize the Simplex or either load a pre-defined one (we will use a pre-defined one in the lab for reproducibility)
        print("Loading simplex")

        if load_simplex is False:
            Simplex = []
            N_p = 11
            for ii in range(N_p):
                alpha_sp = np.random.uniform(0, 1) * ((2**B_cnt) / 2)
                gam_exp = np.random.uniform(-5, -1)
                beta_ = np.random.uniform(0, 2) * (2**B_cnt-1)/imgsize_vector
                gamma = 10**gam_exp
                simp_arr = np.array([gamma, alpha_sp, beta_])
                Simplex.append(simp_arr*1)
                # np.savez("Simplex2.npz", data = Simplex)

        else:
            Simplex = np.load("Simplex2.npz", allow_pickle=True)['data']

        # Compute the cost F(x) associated to each point in the Initial Simplex
        F_of_x = []
        Accs = []
        Sparsities = []
        for init_simp in range(len(Simplex)):
            simp_arr = Simplex[init_simp]  # Take Simplex from list
            gamma = simp_arr[0]  # Regularization hyperparameter
            alpha_sp = simp_arr[1]  # Threshold of accumulators
            beta_ = simp_arr[2]  # incrementation step of accumulators
            ############### F(x) for Nelder-Mead with x = [gamma, alpha_sp, beta] ###################
            # The function "evaluate_F_of_x_2" performs:
            #    a) thresholding and encoding of bundled dataset into final HDC "ternary" vectors (-1, 0, +1)
            #    b) Training and testing the HDC system on "Nbr_of_trials" trials (with different random dataset splits)
            #    c) Returns lambda_1*Acc + lambda_2*Sparsity, Accuracy and Sparsity for each trials
            local_avg, local_avgre, local_sparse = evaluate_F_of_x(Nbr_of_trials, HDC_cont_all, LABELS, beta_, bias_, gamma, alpha_sp, n_class, N_train, D_b, lambda_1, lambda_2, B_cnt)
            F_of_x.append(1 - np.mean(local_avg))  # Append cost F(x)
            Accs.append(np.mean(local_avgre))
            Sparsities.append(np.mean(local_sparse))
            ##################################

        # Transform lists to numpy array:
        F_of_x = np.array(F_of_x)
        Accs = np.array(Accs)
        Sparsities = np.array(Sparsities)
        Simplex = np.array(Simplex)

        objective_ = []  # Will contain the objective value F(x) for each simplex as the Nelder-Mead search goes on
        STD_ = []  # Will contain the standard deviation of all F(x) as the Nelder-Mead search goes on

        # For the details about the Nelder-Mead step, please refer to the course notes / reference, we are simply implementing that
        for iter_ in range(NM_iter):

            STD_.append(np.std(F_of_x))
            if np.std(F_of_x) < STD_EPS and 100 < iter_:
                break  # Early-stopping criteria

            # 1) sort Accs, Sparsities, F_of_x, Simplex, add best objective to array "objective_"
            sort_indices = np.argsort(F_of_x)
            F_of_x = F_of_x[sort_indices]  # We want to minimise this
            Accs = Accs[sort_indices]
            Sparsities = Sparsities[sort_indices]
            Simplex = Simplex[sort_indices]

            objective_.append(F_of_x[0])

            # 2) average simplex x_0
            x_0 = np.average(Simplex[:-1], axis=0)
            x_n_plus_1 = Simplex[-1]

            # 3) Reflexion x_r
            x_r = x_0 + alpha_simp*(x_0-x_n_plus_1)

            # Evaluate cost of reflected point x_r
            gamma_x_r = x_r[0]  # Regularization hyperparameter
            alpha_sp_x_r = x_r[1]  # Threshold of accumulators
            beta_x_r = x_r[2]  # incrementation step of accumulators
            local_avg_x, acc_curr, sparse_curr = evaluate_F_of_x(Nbr_of_trials, HDC_cont_all, LABELS, beta_x_r, bias_, gamma_x_r, alpha_sp_x_r, n_class, N_train, D_b, lambda_1, lambda_2, B_cnt)
            F_curr = 1-local_avg_x

            if F_of_x[0] <= F_curr < F_of_x[-2]:
                F_of_x[-1] = F_curr
                Simplex[-1] = x_r
                Accs[-1] = acc_curr
                Sparsities[-1] = sparse_curr
                rest = False
            else:
                rest = True

            if rest:
                # 4) Expansion x_e
                if F_curr < F_of_x[0]:  # Reflected point is best point so far
                    x_e = x_0 + gamma_simp*(x_r-x_0)  # Expanded point

                    # Evaluate cost of reflected point x_e
                    gamma_exp = x_e[0]  # Regularization hyperparameter
                    alpha_sp_exp = x_e[1]  # Threshold of accumulators
                    beta_exp = x_e[2]  # incrementation step of accumulators
                    local_avg_exp, acc_exp, sparse_exp = evaluate_F_of_x(Nbr_of_trials, HDC_cont_all, LABELS, beta_exp, bias_, gamma_exp, alpha_sp_exp, n_class, N_train, D_b, lambda_1, lambda_2, B_cnt)
                    F_exp = 1-local_avg_exp

                    if F_exp < F_curr:
                        F_of_x[-1] = F_exp
                        Simplex[-1] = x_e
                        Accs[-1] = acc_exp
                        Sparsities[-1] = sparse_exp
                    else:
                        F_of_x[-1] = F_curr
                        Simplex[-1] = x_r
                        Accs[-1] = acc_curr
                        Sparsities[-1] = sparse_curr

                else:  # F_curr >= F_of_x[-2]
                    # 4) Contraction x_c
                    if F_curr < F_of_x[-1]:
                        x_c = x_0 + rho_simp*(x_r - x_0)

                    elif F_curr >= F_of_x[-1]:
                        x_c = x_0 + rho_simp*(Simplex[-1] - x_0)

                    # Evaluate cost of contracted point x_e
                    gamma_c = x_c[0]  # Regularization hyperparameter
                    alpha_sp_c = x_c[1]  # Threshold of accumulators
                    beta_c = x_c[2]  # incrementation step of accumulators
                    local_avg_c, acc_c, sparse_c = evaluate_F_of_x(Nbr_of_trials, HDC_cont_all, LABELS, beta_c, bias_, gamma_c, alpha_sp_c, n_class, N_train, D_b, lambda_1, lambda_2, B_cnt)
                    F_c = 1-local_avg_c

                    if F_c < F_curr or F_c < F_of_x[-1]:
                        F_of_x[-1] = F_c
                        Simplex[-1] = x_c
                        Accs[-1] = acc_c
                        Sparsities[-1] = sparse_c
                    else:
                        # 4) Shrinking
                        for rep in range(1, Simplex.shape[0]):
                            Simplex[rep] = Simplex[0] + sigma_simp*(Simplex[rep] - Simplex[0])


        ##################################
        # At the end of the Nelder-Mead search and training, save Accuracy and Sparsity of the best cost F(x) into the ACCS and SPARSES arrays
        idx = np.argsort(F_of_x)
        F_of_x = F_of_x[idx]
        Accs = Accs[idx]
        Sparsities = Sparsities[idx]
        Simplex = Simplex[idx, :]
        ACCS[optimalpoint] = Accs[0]
        SPARSES[optimalpoint] = Sparsities[0]
        ##################################


    """
    Plot results (DO NOT TOUCH CODE)
    Your code above should return:
        SPARSES: array with sparsity of each chosen lambda_
        ACCS: array of accuracy of each chosen lambda_
        objective_: array of the evolution of the Nelder-Mead objective of the last lambda_ under test
        STD_: array of the standard deviation of the simplex of the last lambda_ under test
    
    """

    # Plot tradeoff curve between Accuracy and Sparsity
    # SPARSES_ = SPARSES[SPARSES > 0]
    # ACCS_ = ACCS[SPARSES > 0]
    # plt.figure(1)
    # plt.plot(SPARSES_, ACCS_, 'x', markersize = 10)
    # plt.grid('on')
    # plt.xlabel("Sparsity")
    # plt.ylabel("Accuracy")
    #
    # from sklearn.svm import SVR
    # y = np.array(ACCS_)
    # X = np.array(SPARSES_).reshape(-1, 1)
    # regr = SVR(C=1.0, epsilon=0.005)
    # regr.fit(X, y)
    # X_pred = np.linspace(np.min(SPARSES_), np.max(SPARSES_), 100).reshape(-1, 1)
    # Y_pred = regr.predict(X_pred)
    # plt.plot(X_pred, Y_pred, '--')
    # plt.savefig("./img/"+str(N_tradeof_points)+"_"+str(NM_iter)+"/acc_spar.png")
    #
    # # Plot the evolution of the Nelder-Mead objective and the standard deviation of the simplex for the last run
    # plt.figure(2)
    # plt.subplot(2, 1, 1)
    # plt.plot(objective_, '.-')
    # plt.title("Objective")
    # plt.grid("on")
    # plt.subplot(2, 1, 2)
    # plt.plot(STD_, '.-')
    # plt.title("Standard deviation")
    # plt.grid("on")
    # plt.savefig("./img/"+str(N_tradeof_points)+"_"+str(NM_iter)+"/obj_std.png")
    #
    #
    # plt.figure(3)
    # plt.plot(lambda_sp, ACCS)
    # plt.xlabel("accuracy")
    # plt.savefig("./img/"+str(N_tradeof_points)+"_"+str(NM_iter)+"/acc.png")
    #
    # plt.figure(4)
    # plt.plot(lambda_sp, SPARSES)
    # plt.xlabel("sparsity)")
    # plt.savefig("./img/"+str(N_tradeof_points)+"_"+str(NM_iter)+"/sparsity.png")

    print("accuracy:", ACCS)
    N_train_list.append(N_train)
    ACCS_LIST.append(ACCS[0])

plt.figure(1)
plt.plot(N_train_list, ACCS_LIST)
plt.xlabel("N_train")
plt.ylabel("Accuracy")
plt.show()

