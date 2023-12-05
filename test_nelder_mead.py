"""
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


def evaluate_F_of_x(x, y):
    return 0.005*x**4+0.01*x**3-0.4*x**2+11+0.1*y**2
    # return 0.01*(x**2) + 0.01*(y**2)


NM_iter = 30
load_simplex = False

# Contraction, expansion,... coefficients:
alpha_simp = 1 # Reflection
gamma_simp = 2 # Expansion
rho_simp = 0.5  # Contraction
sigma_simp = 0.5  # Shrink

# Initialize the Simplex or either load a pre-defined one (we will use a pre-defined one in the lab for reproducibility)
print("Loading simplex")

if not load_simplex:
    Simplex = []
    N_p = 5
    for ii in range(N_p):
        x_sp = -2.5+(np.random.uniform(0, 1))*5
        y_sp =  5+np.random.uniform(0, 1)*3
        simp_arr = np.array([x_sp, y_sp])
        Simplex.append(simp_arr*1)
        # np.savez("Simplex_test_np.npz", data = Simplex)

else:
    Simplex = np.load("Simplex_test_np.npz", allow_pickle=True)['data']

# Compute the cost F(x) associated to each point in the Initial Simplex
F_of_x = []
for init_simp in range(len(Simplex)):
    simp_arr = Simplex[init_simp] # Take Simplex from list
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
F_of_x = np.array(F_of_x)
Simplex = np.array(Simplex)

objective_ = []  # Will contain the objective value F(x) for each simplex as the Nelder-Mead search goes on

# keep track of Simplex in every iteration for plotting purposes
Simplex_list = []
# For the details about the Nelder-Mead step, please refer to the course notes / reference, we are simply implementing that
for iter_ in range(NM_iter):


    #  1) Sort Accs, Sparsities, F_of_x, Simplex, add best objective to array "objective_"
    sort_indices = np.argsort(F_of_x)

    F_of_x = F_of_x[sort_indices] # We want to minimise this
    Simplex = Simplex[sort_indices]
    objective_.append(F_of_x[0])

    # 2)Average simplex x_0
    x_0 = np.average(Simplex[:-1,:], axis=0)  # centroid of all vertices except the last one
    x_n_plus_1 = Simplex[-1,:]

    #  3)Reflexion x_r
    x_r = x_0 + alpha_simp*(x_0-x_n_plus_1)

    # Evaluate cost of reflected point x_r
    x_r_ = x_r[0]
    y_r_ = x_r[1]
    local_avg_x = evaluate_F_of_x(x_r_, y_r_)
    F_curr = local_avg_x

    if F_of_x[0] <= F_curr < F_of_x[-2]:
        F_of_x[-1] = F_curr
        Simplex[-1] = x_r
        rest = False
    else:
        rest = True

    if rest:
        # 4) Expansion x_e
        if F_curr < F_of_x[0]:  # Reflected point is best point so far

            x_e = x_0 + gamma_simp*(x_r-x_0)  # Expanded point

            # Evaluate cost of reflected point x_e
            x_exp_ = x_e[0]
            y_exp_ = x_e[1]
            local_avg_exp = evaluate_F_of_x(x_exp_, y_exp_)
            F_exp = local_avg_exp

            if F_exp < F_curr:
                F_of_x[-1] = F_exp
                Simplex[-1] = x_e
            else:
                F_of_x[-1] = F_curr
                Simplex[-1] = x_r

        else:  # F_curr >= F_of_x[-2]
            # 4) Contraction x_c
            if F_curr < F_of_x[-1]:
                x_c = x_0 + rho_simp*(x_r - x_0)  # Contracted point on the outside

            elif F_curr >= F_of_x[-1]:
                x_c = x_0 + rho_simp*(Simplex[-1] - x_0)  # Contracted point on the inside

            # Evaluate cost of contracted point x_e
            x_c_ = x_c[0]
            y_c_ = x_c[1]
            local_avg_c = evaluate_F_of_x(x_c_, y_c_)
            F_c = local_avg_c

            if F_c < F_curr or F_c < F_of_x[-1]:
                F_of_x[-1] = F_c
                Simplex[-1] = x_c
            else:
                # 4) Shrinking
                for rep in range(1, Simplex.shape[0]):
                    Simplex[rep] = Simplex[0] + sigma_simp*(Simplex[rep] - Simplex[0])

    idx = np.argsort(F_of_x)
    Simplex_list.append(Simplex[idx, :])
#
# try:
#     mkdir("./img/test/"+str(N_tradeof_points)+"_"+str(NM_iter))
#     print("folder for plots created")
# except FileExistsError:
#     print("reusing existing folder to save plots")

# Plot the evolution of the Nelder-Mead objective and the standard deviation of the simplex for the last run
# plt.figure(1)
# plt.subplot(1,1,1)
# plt.plot(objective_, '.-')
# plt.title("Objective")
# plt.grid("on")
# plt.savefig("./img/test/"+str(N_tradeof_points)+"_"+str(NM_iter)+"/obj_std.png")


### plot 3d figure
# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
fig, ax = plt.subplots()
lim = 15
X = np.arange(-10, 8, 0.1)
Y = np.arange(-lim, lim, 0.1)
X, Y = np.meshgrid(X, Y)
Z = evaluate_F_of_x(X, Y)
create_animation(Simplex_list, X, Y, Z)
print("Simplex animation is created.")
