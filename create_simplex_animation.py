import numpy as np
import matplotlib.pyplot as plt


def create_animation(simplex_list, X, Y, Z):
    for i, simplex in enumerate(simplex_list):
        fig, ax = plt.subplots()

        # ax.computed_zorder = False
        levels = np.linspace(Z.min(), Z.max(), 15)
        # plot the surface of F(x)
        # surf = ax.plot_surface(X, Y, Z, cmap='viridis',linewidth=0, antialiased=False)
        ax.contourf(X, Y, Z, levels=levels)

        # plot Simplex
        triangle_indices = list(range(simplex.shape[0]))
        triangle_indices.append(0)
        ax.plot(simplex[triangle_indices, 0], simplex[triangle_indices, 1], '-', marker='o', color="black")

        # save frame
        plt.savefig("./animation/"+str(i))
        plt.close()

# ax.set_zlim(-1, 2)
# ax.zaxis.set_major_locator(LinearLocator(10))
# ax.zaxis.set_major_formatter('{x:.02f}')
# fig.colorbar(surf, shrink=0.5, aspect=5)
# ax.set_zlim(-3)
# plt.show()
