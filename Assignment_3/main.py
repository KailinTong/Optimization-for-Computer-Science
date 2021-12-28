import numpy as np
from scipy.optimize import approx_fprime
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import json
from typing import Callable, Type
from copy import deepcopy
import pprint

MU = 0
SiGMA = 0.5
EPOCH = 350
pp = pprint.PrettyPrinter(indent=4)

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)  # only difference


def feed_forward(W1, b1, W0, b0, x, y):
    x = x.T
    S = y.size
    y_label_prob = np.zeros((3, S))
    for i in range(y.size):
        y_label_prob[y[i]][i] = 1

    # expand the dimension again
    W0 = np.expand_dims(W0, axis=0)
    b0 = np.expand_dims(b0, axis=0)
    W1 = np.expand_dims(W1, axis=1)
    b1 = np.expand_dims(b1, axis=1)

    z1 = W0 @ x + b0
    a1 = np.log(1 + np.exp(z1))
    z2 = W1 @ a1 + b1
    y_prob = softmax(z2)
    L = -np.sum(np.log(y_prob) * y_label_prob)

    return L.item()


def func_w1(W1, b1, W0, b0, x, y):
    return feed_forward(W1=W1, b1=b1, W0=W0, b0=b0, x=x, y=y)


def func_b1(b1, W1, W0, b0, x, y):
    return feed_forward(W1=W1, b1=b1, W0=W0, b0=b0, x=x, y=y)


def func_w0(W0, b1, W1, b0, x, y):
    return feed_forward(W1=W1, b1=b1, W0=W0, b0=b0, x=x, y=y)


def func_b0(b0, b1, W0, W1, x, y):
    return feed_forward(W1=W1, b1=b1, W0=W0, b0=b0, x=x, y=y)


def approx_grad_w1(func: Callable[[np.ndarray], float], W1, b1, W0, b0, x, y) -> np.ndarray:
    return approx_fprime(W1, func, 0.0001, b1, W0, b0, x, y)


def approx_grad_b1(func: Callable[[np.ndarray], float], W1, b1, W0, b0, x, y) -> np.ndarray:
    return approx_fprime(b1, func, 0.0001, W1, W0, b0, x, y)


def approx_grad_w0(func: Callable[[np.ndarray], float], W1, b1, W0, b0, x, y) -> np.ndarray:
    return approx_fprime(W0, func, 0.0001, b1, W1, b0, x, y)


def approx_grad_b0(func: Callable[[np.ndarray], float], W1, b1, W0, b0, x, y) -> np.ndarray:
    return approx_fprime(b0, func, 0.0001, b1, W0, W1, x, y)


def validate_gradient():
    """
        validate the gradient
    """
    num_input = 4
    num_hidden = 1
    num_output = 3
    W0 = np.random.normal(MU, SiGMA, size=(num_hidden, num_input))
    W1 = np.random.normal(MU, SiGMA, size=(num_output, num_hidden))
    b0 = np.random.normal(MU, SiGMA, size=(num_hidden, 1))
    b1 = np.random.normal(MU, SiGMA, size=(num_output, 1))
    net = NN(num_input, num_hidden, num_output, gradient_method='GD')
    net.set_theta(W1=W1, b1=b1, W0=W0, b0=b0)

    xs = x_train_g[[0], :]
    ys = y_train_g[[0]]

    net.feed_forward(xs)
    net.back_propogate(ys)
    ana_grad_w1 = net.gradient['W1'].squeeze()
    ana_grad_b1 = net.gradient['b1'].squeeze()
    ana_grad_w0 = net.gradient['W0'].squeeze()
    ana_grad_b0 = net.gradient['b0'].squeeze()

    # squeeze inputs for gradient approximation
    W0 = W0.squeeze(axis=0)
    W1 = W1.squeeze(axis=1)
    b0 = b0.squeeze(axis=0)
    b1 = b1.squeeze(axis=1)

    app_grad_w1 = approx_grad_w1(func_w1, W1, b1, W0, b0, xs, ys)
    app_grad_b1 = approx_grad_b1(func_b1, W1, b1, W0, b0, xs, ys)
    app_grad_w0 = approx_grad_w0(func_w0, W1, b1, W0, b0, xs, ys)
    app_grad_b0 = approx_grad_b0(func_b0, W1, b1, W0, b0, xs, ys)

    print('Difference of two graidents of w1: {}'.format(app_grad_w1 - ana_grad_w1))
    print('Difference of two graidents of b1: {}'.format(app_grad_b1 - ana_grad_b1))
    print('Difference of two graidents of w0: {}'.format(app_grad_w0 - ana_grad_w0))
    print('Difference of two graidents of b0: {}'.format(app_grad_b0 - ana_grad_b0))


class NN(object):
    def __init__(self, num_input: int, num_hidden: int, num_output: int, gradient_method: str, dtype=np.float32,
                 learning_rate=0.01):
        self.num_input = num_input
        self.num_hidden = num_hidden
        self.num_output = num_output
        self.dtype = dtype
        self.gradient_method = gradient_method
        self.var = {}
        self.diff = {}
        self.theta = {}
        self.gradient = {}
        self.init_params()
        self.lr = learning_rate
        self.this_theta_nag = 0
        self.prev_p = deepcopy(self.theta)
        self.this_p = deepcopy(self.theta)

    def init_params(self):
        self.theta['W0'] = np.random.normal(MU, SiGMA, size=(self.num_hidden, self.num_input))
        self.theta['W1'] = np.random.normal(MU, SiGMA, size=(self.num_output, self.num_hidden))
        self.theta['b0'] = np.random.normal(MU, SiGMA, size=(self.num_hidden, 1))
        self.theta['b1'] = np.random.normal(MU, SiGMA, size=(self.num_output, 1))

    def set_theta(self, W1, b1, W0, b0):
        self.theta['W1'] = W1
        self.theta['b1'] = b1
        self.theta['W0'] = W0
        self.theta['b0'] = b0

    def feed_forward(self, x: np.ndarray, training: bool = True):

        assert x.shape[1] == self.num_input, "x shape is {}; it should be {}".format(x.shape, (self.num_input, 1))
        x = x.T
        z1 = self.theta['W0'] @ x + self.theta['b0']
        a1 = np.log(1 + np.exp(z1))
        z2 = self.theta['W1'] @ a1 + self.theta['b1']
        y_prob = softmax(z2)
        y_pred = np.argmax(y_prob, axis=0)
        if training:
            self.var['x'] = x
            self.var['z1'] = z1
            self.var['a1'] = a1
            self.var['z2'] = z2
            self.var['y_prob'] = y_prob
            self.var['y_pred'] = y_pred
        return y_pred

    def back_propogate(self, y: np.ndarray):
        S = y.size
        y_label_prob = np.zeros((3, S))

        for i in range(y.size):
            y_label_prob[y[i]][i] = 1

        '''calculate gradient for one sample'''
        gradient_w1_sum = np.zeros((self.num_output, self.num_hidden))
        gradient_b1_sum = np.zeros((self.num_output, 1))
        gradient_w0_sum = np.zeros((self.num_hidden, self.num_input))
        gradient_b0_sum = np.zeros((self.num_hidden, 1))

        for s in range(S):
            '''Prepare variables for one sample '''
            ys = y_label_prob[:, [s]]
            xs = self.var['x'][:, [s]]
            yprob = self.var['y_prob'][:, [s]]
            z2 = self.var['z2'][:, [s]]
            a1 = self.var['a1'][:, [s]]
            z1 = self.var['z1'][:, [s]]


            '''gradient w1 b1'''
            dldy = - ys * 1 / yprob
            sigma1, sigma2, sigma3 = softmax(z2)[0].item(), softmax(z2)[1].item(), softmax(z2)[2].item()
            dydz = np.array([[sigma1 * (1 - sigma1), -sigma1 * sigma2, -sigma1 * sigma3],
                             [-sigma1 * sigma2, sigma2 * (1 - sigma2), -sigma2 * sigma3],
                             [-sigma1 * sigma3, -sigma2 * sigma3, sigma3 * (1 - sigma3)]])
            e2 = dydz @ dldy
            gradient_w1 = e2 @ a1.T
            gradient_b1 = e2

            '''gradient w0 b0'''
            dzda = self.theta['W1'].T
            dadz = np.diag(np.exp(z1[:, 0]) / (1 + np.exp(z1[:, 0])))
            e1 = dadz @ dzda
            gradient_w0 = e1 @ e2 @ xs.T
            gradient_b0 = e1 @ e2

            '''save the sum'''
            gradient_w1_sum += gradient_w1
            gradient_b1_sum += gradient_b1
            gradient_w0_sum += gradient_w0
            gradient_b0_sum += gradient_b0

        self.gradient['W1'] = gradient_w1_sum / S
        self.gradient['b1'] = gradient_b1_sum / S
        self.gradient['W0'] = gradient_w0_sum / S
        self.gradient['b0'] = gradient_b0_sum / S

        '''calculate loss'''
        loss = 1.0 / S * -np.sum(y_label_prob * np.log(self.var['y_prob']))
        return loss

    def export_model(self):
        with open(f'model_{self.gradient_method}.json', 'w') as fp:
            json.dump({key: value.tolist() for key, value in self.theta.items()}, fp)

    def update(self):
        if self.gradient_method == "GD":
            self.theta['W0'] = self.theta['W0'] - self.lr * self.gradient['W0']
            self.theta['W1'] = self.theta['W1'] - self.lr * self.gradient['W1']
            self.theta['b0'] = self.theta['b0'] - self.lr * self.gradient['b0']
            self.theta['b1'] = self.theta['b1'] - self.lr * self.gradient['b1']
        if self.gradient_method == "NAG":
            next_theta = (1 + (1+4*self.this_theta_nag**2)**0.5) / 2.0
            for key in self.this_p:
                this_q = self.this_p[key] + (self.this_theta_nag - 1)/next_theta * (self.this_p[key] - self.prev_p[key])
                self.prev_p[key] = deepcopy(self.this_p[key])
                self.this_p[key] = this_q - self.lr * self.gradient[key]
                self.theta[key] = deepcopy(self.this_p[key])
            self.this_theta_nag = next_theta

def task1():
    """ Neural Network

        Requirements for the plots:
            - ax[0] Plot showing the training loss for both variants
            - ax[1] Plot showing the training and test accuracy for both variants
    """
    # validate the gradient
    validate_gradient()

    # Create the models
    # Model using steepest descent
    net_GD = NN(4, 16, 3, gradient_method='GD')

    # training
    pp.pprint("*"*50)
    pp.pprint("Training with Steepest Descent starts!")
    loss_GD = []
    accuracy_GD = [ ]
    accuracy_GD_test = []
    accuracy_train = accuracy_test = 0
    for i in range(EPOCH):
        y_pred = net_GD.feed_forward(x_train_g)
        loss = net_GD.back_propogate(y_train_g)
        accuracy_train = np.mean(y_pred == y_train_g)
        net_GD.update()
        pp.pprint("Loss: {}".format(loss))
        loss_GD.append(loss)
        accuracy_GD.append(accuracy_train)
        # test
        y_pred = net_GD.feed_forward(x_test_g, training=False)
        accuracy_test = np.mean(y_pred == y_test_g)
        accuracy_GD_test.append(accuracy_test)
    pp.pprint("GD final training accuracy: {}; final test accuracy: {}".format(accuracy_train, accuracy_test))

    # Model using Nesterovs method
    net_NAG = NN(4, 16, 3, gradient_method='NAG')

    # training
    pp.pprint("*"*50)
    pp.pprint("Training with Nesterov Accelerated Gradient descent starts!")
    loss_NGA = []
    accuracy_NGA = []
    accuracy_NGA_test = []
    for i in range(EPOCH):
        # training
        y_pred = net_NAG.feed_forward(x_train_g)
        loss = net_NAG.back_propogate(y_train_g)
        accuracy_train = np.mean(y_pred == y_train_g)
        net_NAG.update()
        pp.pprint("Loss: {}".format(loss))
        loss_NGA.append(loss)
        accuracy_NGA.append(accuracy_train)
        # test
        y_pred = net_NAG.feed_forward(x_test_g, training=False)
        accuracy_test = np.mean(y_pred == y_test_g)
        accuracy_NGA_test.append(accuracy_test)
    pp.pprint("NGA final training accuracy: {}; final test accuracy: {}".format(accuracy_train, accuracy_test))


    net_GD.export_model()
    net_NAG.export_model()

    # Configure plot
    fig = plt.figure(figsize=[12, 6])
    axs = []
    axs.append(fig.add_subplot(121))
    axs.append(fig.add_subplot(122))

    epoch = np.arange(EPOCH)
    axs[0].set_title('Loss')
    axs[0].grid()
    axs[0].semilogy(epoch, loss_GD, label="GD")
    axs[0].semilogy(epoch, loss_NGA, label="NGA")
    axs[0].legend()
    axs[1].set_title('Accuracy')
    axs[1].grid()
    axs[1].plot(epoch, accuracy_GD,  label="GD Training")
    axs[1].plot(epoch, accuracy_NGA, label="NGA Training")
    axs[1].plot(epoch, accuracy_GD_test, label="GD Test")
    axs[1].plot(epoch, accuracy_NGA_test, label="NGA Test")

    axs[1].legend()


    return fig




if __name__ == '__main__':

    # load the data set
    with np.load('data_train.npz') as data_set:
        # get the training data
        x_train_g = data_set['x']
        y_train_g = data_set['y']

    with np.load('data_test.npz') as data_set:
        # get the test data
        x_test_g = data_set['x']
        y_test_g = data_set['y']

    print(f'Training set with {x_train_g.shape[0]} data samples.')
    print(f'Test set with {x_test_g.shape[0]} data samples.')

    tasks = [task1]

    pdf = PdfPages('figures.pdf')
    for task in tasks:
        retval = task()
        fig = retval[0] if type(retval) is tuple else retval
        pdf.savefig(fig)
    pdf.close()
