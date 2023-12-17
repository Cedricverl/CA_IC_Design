"""
Unit test 3:
Test file to test Nelder Mead algorithm on easer, easy to follow (analytical) functions.
All iterations can be plotted and exported to jpg files using the function create_animation()

greetz Cedric
"""

from create_simplex_animation import create_animation
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator

from os import mkdir
plt.close('all')

# np.random.seed(seed=69)
def evaluate_F_of_x(x, y):
    return 0.005*x**4+0.01*x**3-0.4*x**2+11+0.1*y**2
    # return 0.01*(x**2) + 0.01*(y**2)


NM_iter = 50
load_simplex = False

# Contraction, expansion,... coefficients:
alpha_simp = 1*0.5  # Reflection > 0
gamma_simp = 2*0.6  # Expansion > 1 and gamma > alpha
rho_simp = 0.5  # Contraction 0 < rho <= 0.5 or 0<rho<1
sigma_simp = 0.5  # Shrink 0<sigma<1

# Initialize the Simplex or either load a pre-defined one (we will use a pre-defined one in the lab for reproducibility)
print("Loading simplex")

if not load_simplex:
    # Simplex = []
    # N_p = 5
    # for ii in range(N_p):
    #     x_sp = np.random.uniform(0, 1)*18-10
    #     y_sp = np.random.uniform(0, 1)*30-15
    #     # x_sp = -2.5+(np.random.uniform(0, 1))*5
    #     # y_sp =  5+np.random.uniform(0, 1)*3
    #     simp_arr = np.array([x_sp, y_sp])
    #     Simplex.append(simp_arr*1)
    Simplex = np.array([[7,-14], [-9, -14], [-9, 14], [7,14]]).astype(float)
        # np.savez("Simplex_test_np.npz", data = Simplex)

else:
    Simplex = np.load("Simplex_test_np.npz", allow_pickle=True)['data']

# Compute the cost F(x) associated to each point in the Initial Simplex
F_of_x = []
for init_simp in range(len(Simplex)):
    simp_arr = Simplex[init_simp]  # Take Simplex from list
    x_sp = simp_arr[0]
    y_sp = simp_arr[1]
    ############### F(x) for Nelder-Mead with x = [gamma, alpha_sp, beta] ###################
    #T he function "evaluate_F_of_x_2" performs:
    #    a) thresholding and encoding of bundled dataset into final HDC "ternary" vectors (-1, 0, +1)
    #    b) Training and testing the HDC system on "Nbr_of_trials" trials (with different random dataset splits)
    #    c) Returns lambda_1*Acc + lambda_2*Sparsity, Accuracy and Sparsity for each trials
    # local_avg, local_avgre, local_sparse = evaluate_F_of_x(Nbr_of_trials, HDC_cont_all, LABELS, beta_, bias_, gamma, alpha_sp, n_class, N_train, D_b, lambda_1, lambda_2, B_cnt)
    local_avg = evaluate_F_of_x(x_sp, y_sp)
    F_of_x.append(local_avg)  # Append cost F(x)
    ##################################

#Transform lists to numpy array:
F_of_x = np.array(F_of_x).astype(float)
Simplex = np.array(Simplex).astype(float)

objective_ = []  # Will contain the objective value F(x) for each simplex as the Nelder-Mead search goes on

# keep track of Simplex in every iteration for plotting purposes
Simplex_list = [Simplex]
print("Start NM")
# For the details about the Nelder-Mead step, please refer to the course notes / reference, we are simply implementing that
for iter_ in range(NM_iter):
    print("NM", iter_,"of", NM_iter)
    #  1) Sort Accs, Sparsities, F_of_x, Simplex, add best objective to array "objective_"
    sort_indices = np.argsort(F_of_x)
    F_of_x = F_of_x[sort_indices]  # We want to minimise this
    Simplex = Simplex[sort_indices]
    objective_.append(F_of_x[0])
    Simplex_list.append(Simplex)

    # 2)Average simplex x_0
    x_0 = np.average(Simplex[:-1], axis=0)  # centroid of all vertices except the last one

    # Reflection x_r

    x_r = x_0 + alpha_simp*(x_0-Simplex[-1])

    # Evaluate cost of reflected point x_r
    F_r = evaluate_F_of_x(x_r[0], x_r[1])

    if F_of_x[0] <= F_r < F_of_x[-2]:
        print("Reflection")
        F_of_x[-1] = F_r
        Simplex[-1] = x_r
        continue

    # Expansion x_e
    if F_r < F_of_x[0]:  # Reflected point is best point so far
        print("Expansion")
        x_e = x_0 + gamma_simp*(x_r-x_0)

        # Evaluate cost of reflected point x_e
        F_e = evaluate_F_of_x(x_e[0], x_e[1])

        if F_e < F_r:
            F_of_x[-1] = F_e
            Simplex[-1] = x_e
            continue
        else:
            F_of_x[-1] = F_r
            Simplex[-1] = x_r
            continue

    flag = None
    x_c = None
    if F_r < F_of_x[-1]:  # else shrink
        x_c = x_0 + rho_simp*(x_r - x_0)
        flag = 0
    if F_r >= F_of_x[-1]:
        x_c = x_0 + rho_simp*(Simplex[-1] - x_0)
        flag = 1
    F_c = evaluate_F_of_x(x_c[0], x_c[1])

    if (F_c < F_r and flag == 0) or (F_c < F_of_x[-1] and flag == 1):
        print("Contraction")
        F_of_x[-1] = F_c
        Simplex[-1] = x_c
        continue

    # Shrinking
    print("Shrinking")
    x1 = Simplex[0]
    for rep in range(1, Simplex.shape[0]):
        Simplex[rep] = x1 + sigma_simp*(Simplex[rep] - x1)
        F_of_x[rep] = evaluate_F_of_x(Simplex[rep][0], Simplex[rep][1])

# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
fig, ax = plt.subplots()
lim = 15
X = np.arange(-10, 8, 0.1)
Y = np.arange(-lim, lim, 0.1)
X, Y = np.meshgrid(X, Y)
Z = evaluate_F_of_x(X, Y)
create_animation(Simplex_list, X, Y, Z)
print("Simplex animation is created.")
