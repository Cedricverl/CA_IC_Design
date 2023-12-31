import numpy as np
import matplotlib.pyplot as plt
from os import mkdir
from datetime import datetime

def create_animation(simplex_list, X, Y, Z):
    folder_name = "./animation/"+datetime.now().strftime("%m-%d_%H-%M")
    try:
        mkdir(folder_name)
        print("folder for animation created")
    except FileExistsError:
        print("reusing existing folder to save animation")

    for i, simplex in enumerate(simplex_list):
        fig, ax = plt.subplots()

        # ax.computed_zorder = False
        levels = np.linspace(Z.min(), Z.max(), 30)
        # plot the surface of F(x)
        # surf = ax.plot_surface(X, Y, Z, cmap='viridis',linewidth=0, antialiased=False)
        ax.contourf(X, Y, Z, levels=levels)

        # plot Simplex
        triangle_indices = list(range(simplex.shape[0]))
        triangle_indices.append(0)
        ax.plot(simplex[triangle_indices, 0], simplex[triangle_indices, 1], '-', marker='o', color="black")

        # save frame
        plt.savefig(folder_name+"/"+str(i))
        plt.close()

# ax.set_zlim(-1, 2)
# ax.zaxis.set_major_locator(LinearLocator(10))
# ax.zaxis.set_major_formatter('{x:.02f}')
# fig.colorbar(surf, shrink=0.5, aspect=5)
# ax.set_zlim(-3)
# plt.show()
