#!/usr/bin/env python

""" Python code submission file.

IMPORTANT:
- Do not include any additional python packages.
- Do not change the existing interface and return values of the task functions.
- Prior to your submission, check that the pdf showing your plots is generated.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.optimize import approx_fprime, linprog
from typing import Callable

# Modify the following global variables to be used in your functions
""" Start of your code
"""

alpha = 1
beta = 1

d = 2.5
b = np.arange(1., 6)
D = np.diag([1., 2., 3., 4., 5.])
A = np.array([[1., 1., 1., 0., 2.],
              [1., 1., 0., 1., 3.],
              [0., 0., -3., 0., 4.],
              [0., 0., 0., -4., 5.],
              [1., -2., 0., 2., 3.]])

""" End of your code
"""


def task1():
    """ Characterization of Functions

        Requirements for the plots:
            - ax[0, 0] Contour plot for a)
            - ax[0, 1] Contour plot for b)
            - ax[1, 0] Contour plot for c)
            - ax[1, 1] Contour plot for d)
    """

    fig, ax = plt.subplots(2, 2, figsize=(12, 12))
    fig.suptitle('Task 1 - Contour plots of functions', fontsize=16)

    ax[0, 0].set_title('a)')
    ax[0, 0].set_xlabel('$x_1$')
    ax[0, 0].set_ylabel('$x_2$')

    ax[0, 1].set_title('b)')
    ax[0, 1].set_xlabel('$x_1$')
    ax[0, 1].set_ylabel('$x_2$')

    ax[1, 0].set_title('c)')
    ax[1, 0].set_xlabel('$x_1$')
    ax[1, 0].set_ylabel('$x_2$')

    ax[1, 1].set_title('d)')
    ax[1, 1].set_xlabel('$x_1$')
    ax[1, 1].set_ylabel('$x_2$')

    """ Start of your code
    """

    x1, x2 = np.meshgrid(np.linspace(-5, 5), np.linspace(-5, 5))
    ax[0, 0].contour(x1, x2, (-x1 + 3 * x2 - 2.5) ** 2, 50)
    c = np.linspace(-5, 5, 1000)
    ax[0, 0].scatter(3 * c - 2.5, c)
    ax[0, 0].set_xlim(-5, 5)
    ax[0, 0].set_ylim(-5, 5)

    ax[0, 1].contour(x1, x2, (x1 - 2) ** 2 + x1 * x2 ** 2 - 2, 50)
    ax[0, 1].scatter([0, 0, 2], [2, -2, 0])
    ax[0, 1].set_xlim(-5, 5)
    ax[0, 1].set_ylim(-5, 5)

    ax[1, 0].contour(x1, x2, x1 ** 2 + x1 * (x1 ** 2 + x2 ** 2) + (x1 ** 2 + x2 ** 2), 50)
    ax[1, 0].scatter([0, -4 / 3, -1, -1], [0, 0, 1, -1])
    ax[1, 0].set_xlim(-5, 5)
    ax[1, 0].set_ylim(-5, 5)

    ax[1, 1].contour(x1, x2, alpha * x1 ** 2 - 2 * x1 + beta * x2 ** 2, 50)
    ax[1, 1].scatter([1 / alpha], [0])

    """ End of your code
    """
    return fig


# Modify the function bodies below to be used for function value and gradient computation
def approx_grad_task1(func: Callable[[np.ndarray], float], x: np.ndarray) -> np.ndarray:
    """ Numerical Gradient Computation
        @param x Vector of size (2,)
        This function shall compute the gradient approximation for a given point 'x' and a function 'func'
        using the given central differences formulation for 2D functions. (Task1 functions)
        @return The gradient approximation
    """
    assert (len(x) == 2)
    epsilon = 0.0001
    x_1_plus = x + np.array([epsilon, 0])
    x_1_minus = x + np.array([-epsilon, 0])
    x_2_plus = x + np.array([0, epsilon])
    x_2_minus = x + np.array([0, -epsilon])

    return 1 / (2 * epsilon) * np.array([func(x_1_plus) - func(x_1_minus),
                                         func(x_2_plus) - func(x_2_minus)])


def approx_grad_task2(func: Callable[[np.ndarray], float], x: np.ndarray) -> np.ndarray:
    """ Numerical Gradient Computation
        @param x Vector of size (n,)
        This function shall compute the gradient approximation for a given point 'x' and a function 'func'
        using scipy.optimize.approx_fprime(). (Task2 functions)
        @return The gradient approximation
    """
    return approx_fprime(x, func, 0.0001)


def func_1a(x: np.ndarray) -> float:
    """ Computes and returns the function value for function 1a) at a given point x
        @param x Vector of size (2,)
    """
    return (-x[0] + 3 * x[1] - 2.5) ** 2


def grad_1a(x: np.ndarray) -> np.ndarray:
    """ Computes and returns the analytical gradient result for function 1a) at a given point x
        @param x Vector of size (2,)
    """
    return np.array([2 * (x[0] - 3 * x[1] + 2.5), -6 * (x[0] - 3 * x[1] + 2.5)])


def func_1b(x: np.ndarray) -> float:
    """ Computes and returns the function value for function 1b) at a given point x
        @param x Vector of size (2,)
    """
    return (x[0] - 2) ** 2 + x[0] * x[1] ** 2 - 2


def grad_1b(x: np.ndarray) -> np.ndarray:
    """ Computes and returns the analytical gradient result for function 1b) at a given point x
        @param x Vector of size (2,)
    """
    return np.array([2 * x[0] + x[1] ** 2 - 4, 2 * x[0] * x[1]])


def func_1c(x: np.ndarray) -> float:
    """ Computes and returns the function value for function 1c) at a given point x
        @param x Vector of size (2,)
    """
    return x[0] ** 2 + x[0] * (x[0] ** 2 + x[1] ** 2) + (x[0] ** 2 + x[1] ** 2)


def grad_1c(x: np.ndarray) -> np.ndarray:
    """ Computes and returns the analytical gradient result for function 1c) at a given point x
        @param x Vector of size (2,)
    """
    return np.array([3 * x[0] ** 2 + 4 * x[0] + x[1] ** 2, 2 * x[1] + 2 * x[0] * x[1]])


def func_1d(x: np.ndarray) -> float:
    """ Computes and returns the function value for function 1d) at a given point x
        @param x Vector of size (2,)
    """
    return alpha * x[0] ** 2 - 2 * x[0] + beta * x[1] ** 2


def grad_1d(x: np.ndarray) -> np.ndarray:
    """ Computes and returns the analytical gradient result for function 1d) at a given point x
        @param x Vector of size (2,)
    """
    return np.array([2 * alpha * x[0] - 2, 2 * beta * x[1]])


def func_2a(x: np.ndarray) -> float:
    """ Computes and returns the function value for function 2a) at a given point x
        @param x Vector of size (n,)
    """
    return 1 / 4 * np.linalg.norm(x - b) ** 4


def grad_2a(x: np.ndarray) -> np.ndarray:
    """ Computes and returns the analytical gradient result for function 2a) at a given point x
        @param x Vector of size (n,)
    """
    return np.linalg.norm(x - b) ** 2 * (x - b)


def func_2b(x: np.ndarray) -> float:
    """ Computes and returns the function value for function 2b) at a given point x
        @param x Vector of size (n,)
    """
    return np.sum(1/2 * (A @ x) ** 2 + A @ x).item()


def grad_2b(x: np.ndarray) -> np.ndarray:
    """ Computes and returns the analytical gradient result for function 2b) at a given point x
        @param x Vector of size (n,)
    """
    u = np.ones(5)
    return A.T @ A @ x + A.T @ u


def func_2c(x: np.ndarray) -> float:
    """ Computes and returns the function value for function 2c) at a given point x
        @param x Vector of size (n,)
    """
    return (x / b).T @ D @ (x / b)


def grad_2c(x: np.ndarray) -> np.ndarray:
    """ Computes and returns the analytical gradient result for function 2c) at a given point x
        @param x Vector of size (n,)
    """
    return (D @ (x / b) + D.T @ (x / b)) / b


def task3():
    """ Numerical Gradient Verification
        ax[0] to ax[3] Bar plot comparison, analytical vs numerical gradient for Task 1
        ax[4] to ax[6] Bar plot comparison, analytical vs numerical gradient for Task 2

    """
    n = 5
    fig = plt.figure(figsize=(15, 10), constrained_layout=True)
    fig.suptitle('Task 3 - Barplots numerical vs analytical', fontsize=16)
    ax = [None, None, None, None, None, None, None]
    keys = ['a)', 'b)', 'c)', 'd)']
    gs = fig.add_gridspec(7, 12)

    n = 2
    for i in range(4):
        ax[i] = fig.add_subplot(gs[1:4, 3 * i:(3 * i + 3)])
        ax[i].set_title('1 ' + keys[i])
        ax[i].set_xticks(np.arange(n))
        ax[i].set_xticklabels((r'$\frac{\partial}{\partial x_1}$', r'$\frac{\partial}{\partial x_2}$'), fontsize=16)

    n = 5
    for k, i in enumerate(range(4, 7)):
        ax[i] = fig.add_subplot(gs[4:, 4 * k:(4 * k + 4)])
        ax[i].set_title('2 ' + keys[k])
        ax[i].set_xticks(np.arange(n))
        ax[i].set_xticklabels((r'$\frac{\partial}{\partial x_1}$', r'$\frac{\partial}{\partial x_2}$',
                               r'$\frac{\partial}{\partial x_3}$', r'$\frac{\partial}{\partial x_4}$',
                               r'$\frac{\partial}{\partial x_5}$'), fontsize=16)

    """ Start of your code
    """

    # Example for plot usage
    bw = 0.3
    x = 10 * np.random.random_sample((2,)) - 5

    # func 1a
    x_ana = grad_1a(x)
    x_num = approx_grad_task1(func_1a, x)

    ax[0].bar([0 - bw / 2, 1 - bw / 2], [x_ana[0], x_ana[1]], bw)
    ax[0].bar([0 + bw / 2, 1 + bw / 2], [x_num[0], x_num[1]], bw)

    # func 1b
    x_ana = grad_1b(x)
    x_num = approx_grad_task1(func_1b, x)

    ax[1].bar([0 - bw / 2, 1 - bw / 2], [x_ana[0], x_ana[1]], bw)
    ax[1].bar([0 + bw / 2, 1 + bw / 2], [x_num[0], x_num[1]], bw)

    # func 1c
    x_ana = grad_1c(x)
    x_num = approx_grad_task1(func_1c, x)

    ax[2].bar([0 - bw / 2, 1 - bw / 2], [x_ana[0], x_ana[1]], bw)
    ax[2].bar([0 + bw / 2, 1 + bw / 2], [x_num[0], x_num[1]], bw)

    # func 1d
    x_ana = grad_1d(x)
    x_num = approx_grad_task1(func_1d, x)

    ax[3].bar([0 - bw / 2, 1 - bw / 2], [x_ana[0], x_ana[1]], bw)
    ax[3].bar([0 + bw / 2, 1 + bw / 2], [x_num[0], x_num[1]], bw)


    x = 10 * np.random.random_sample((5,)) - 5

    # func 2a
    x_ana = grad_2a(x)
    x_num = approx_grad_task2(func_2a, x)

    ax[4].bar([0 - bw / 2, 1 - bw / 2], [x_ana[0], x_ana[1]], bw)
    ax[4].bar([0 + bw / 2, 1 + bw / 2], [x_num[0], x_num[1]], bw)

    # func 2b
    x_ana = grad_2b(x)
    x_num = approx_grad_task2(func_2b, x)

    ax[5].bar([0 - bw / 2, 1 - bw / 2], [x_ana[0], x_ana[1]], bw)
    ax[5].bar([0 + bw / 2, 1 + bw / 2], [x_num[0], x_num[1]], bw)

    # func 2c
    x_ana = grad_2c(x)
    x_num = approx_grad_task2(func_2c, x)

    ax[6].bar([0 - bw / 2, 1 - bw / 2], [x_ana[0], x_ana[1]], bw)
    ax[6].bar([0 + bw / 2, 1 + bw / 2], [x_num[0], x_num[1]], bw)

    """ End of your code
    """
    return fig


def task4():
    """ Scheduling Optimization Problem
        @return The scheduling plan M
    """

    """ Start of your code
    """
    c = [0.11, 0.13, 0.09, 0.12, 0.15, 0.14, 0.11, 0.12, 0.10, 0.13, 0.08, 0.13, 0.14, 0.14, 0.09, 0.13]
    A_eq = np.block([np.eye(8), np.eye(8)])
    b_eq = np.array([1200, 1500, 1400, 400, 1000, 800, 760, 1300])
    A_ub = np.block([[np.ones(8), np.zeros(8)], [np.zeros(8), np.ones(8)]])
    b_ub = np.array([4500, 4500])
    bounds = [(480, None), (600, None), (560, None), (0, None), (0, None), (0, None), (0, None), (0, None), (0, None), (0, None),
              (0, None), (0, None), (0, None), (0, None), (0, None), (0, None)]

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='revised simplex')
    print(res)
    M = np.vstack((res.x[0:8],res.x[8:])).T
    print("M is {}".format(M))
    energy = np.sum(c * res.x)
    print("Total energy consumption is {}".format(energy))

    # question (g)
    bounds = [(1200, None), (1500, None), (1400, None), (0, None), (0, None), (0, None), (0, None), (1300, None), (0, None), (0, None),
              (0, None), (0, None), (0, None), (0, None), (0, None), (0, None)]
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='revised simplex')
    print(res)
    """ End of your code 
    """
    return M


if __name__ == '__main__':
    tasks = [task1, task3]

    pdf = PdfPages('figures.pdf')
    for task in tasks:
        retval = task()
        pdf.savefig(retval)
    pdf.close()

    task4()
