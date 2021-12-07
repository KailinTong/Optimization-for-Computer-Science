#!/usr/bin/env python

""" Python code submission file.

IMPORTANT:
- Do not include any additional python packages.
- Do not change the existing interface and return values of the task functions.
- Prior to your submission, check that the pdf showing your plots is generated.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import inv
from matplotlib.backends.backend_pdf import PdfPages
from typing import Callable


def task1():

    """ Lagrange Multiplier Problem

        Requirements for the plots:
            - ax[0] Contour plot for a)
            - ax[1] Contour plot for b)
            - ax[2] Contour plot for c)
    """

    fig, ax = plt.subplots(1, 3, figsize=(18,6))
    fig.suptitle('Task 1 - Contour plots + Constraints', fontsize=16)

    ax[0].set_title('a)')
    ax[0].set_xlabel('$x_1$')
    ax[0].set_ylabel('$x_2$')
    ax[0].set_aspect('equal')

    ax[1].set_title('b)')
    ax[1].set_xlabel('$x_1$')
    ax[1].set_ylabel('$x_2$')
    ax[1].set_aspect('equal')

    ax[2].set_title('c)')
    ax[2].set_xlabel('$x_1$')
    ax[2].set_ylabel('$x_2$')
    ax[2].set_aspect('equal')


    """ Start of your code
    """
    x1, x2 = np.meshgrid(np.linspace(-5, 5), np.linspace(-5, 5))
    ax[0].contourf(x1, x2, x2 - x1, 100, cmap='gist_rainbow')
    x1c = np.linspace(-10, 10)
    x2c = 4 * x1c
    ax[0].plot(x1c, x2c, color="b")
    x1c = np.linspace(-10, 10)
    x2c = 1 / 10  * x1c ** 2 - 3
    ax[0].plot(x1c, x2c, color="b")
    ax[0].scatter(5, -1/2, c="r")
    ax[0].set_xlim([-5, 5])
    ax[0].set_ylim([-5, 5])

    x1, x2 = np.meshgrid(np.linspace(-5, 5), np.linspace(-5, 5))
    ax[1].contourf(x1, x2, x1 ** 2 + x2 ** 2, 100, cmap='gist_rainbow')
    x1c = np.linspace(-10, 10)
    x2c = 3 - x1c
    ax[1].plot(x1c, x2c, color="b")
    x1c = np.linspace(-10, 10)
    x2c = np.ones_like(x1c) * 2
    ax[1].plot(x1c, x2c, color="b")
    ax[1].scatter(1, 2, c="r")
    ax[1].set_xlim([-5, 5])
    ax[1].set_ylim([-5, 5])

    x1, x2 = np.meshgrid(np.linspace(-5, 5), np.linspace(-5, 5))
    ax[2].contourf(x1, x2, (x2 - 1) ** 2 + x1 * x2**2 - 2, 100, cmap='gist_rainbow')
    circle1 = plt.Circle((0, 0), 2, fill=False, color='b')
    ax[2].add_patch(circle1)
    ax[2].scatter(1, 0, c="r")
    ax[2].scatter([0, 0, (1 - 7**0.5)/3, (1 - 7**0.5)/3], [2**0.5, -2**0.5, 1.9233, -1.9233], c="g")

    ax[2].set_xlim([-5, 5])
    ax[2].set_ylim([-5, 5])


    """ End of your code
    """
    return fig


def task2():

    """ Lagrange Augmentation
        ax Filled contour plot including the constraints and the iterates of x_k

    """
    fig, ax = plt.subplots(1, 1, figsize=(8,8))
    fig.suptitle('Task 2 - Contour plots + Constraints + Iterations over k', fontsize=16)
    """ Start of your code
    """
    x1, x2 = np.meshgrid(np.linspace(-5, 5), np.linspace(-5, 5))
    ax.contourf(x1, x2, (x1 - 1)**2 - x1 * x2, 100, cmap='gist_rainbow')
    x1c = np.linspace(-10, 10)
    x2c = 4 - x1c
    ax.plot(x1c, x2c, color="b")
    ax.scatter(3/2, 5/2, c="r")
    ax.set_xlim([-2, 5])
    ax.set_ylim([-2, 5])
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    # algorithm 1
    lambd = 5
    alphas = [0.65]
    colors = ["g"]
    K = 20

    for a in range(len(alphas)):
        alpha = alphas[a]
        x1 = []
        x2 = []
        for i in range(K):
            x = np.array([(6 * alpha - lambd) / (4 * alpha - 1), (10 * alpha - 3 * lambd + 2) / (4 * alpha - 1)])
            lambd = alpha * (x[0] + x[1] - 4) + lambd
            x1.append(x[0])
            x2.append(x[1])
        ax.plot(x1, x2, "x", color=colors[a], label='alpha = ' + str(alpha))
    plt.legend()

    """ End of your code
    """
    return fig


def task3():

    """ Least Squares Fitting
        ax 3D scatter plot and wireframe of the computed solution
    """
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='3d')
    fig.suptitle('Task 3 - Data points vs. LS solution', fontsize=16)

    with np.load('data.npz') as fc:
        x = fc['data'][:,0]
        y = fc['data'][:,1]
        z = fc['data'][:,2]

    N = len(x)
    A = None
    x_solution = None
    """ Start of your code
    """
    ax.scatter(x, y, z, marker='o')
    A = np.zeros((N, 9))
    for i in range(3):
        for j in range(3):
            A[:, 3*i+j] = x ** i * y ** j
    x_solution = inv(A.T @ A) @ A.T @ np.expand_dims(z, axis=1)
    x_hat, y_hat = np.meshgrid(np.linspace(-4, 4), np.linspace(-4, 4))
    z_hat = np.zeros_like(x_hat)
    for i in range(3):
        for j in range(3):
            z_hat += x_solution[3*i+j] * x_hat ** i * y_hat ** j

    ax.plot_wireframe(x_hat, y_hat, z_hat, color='g', rstride=2, cstride=2)
    np.set_printoptions(precision=3, suppress=True)
    print("coefficients are:\n", x_solution)

    """ End of your code
    """
    return fig, A, x_solution


if __name__ == '__main__':
    tasks = [task1, task2, task3]

    pdf = PdfPages('figures.pdf')
    for task in tasks:
        retval = task()
        fig = retval[0] if type(retval) is tuple else retval
        pdf.savefig(fig)
    pdf.close()
