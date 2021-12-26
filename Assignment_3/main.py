import numpy as np
from scipy.optimize import approx_fprime
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import json


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference


class NN(object):
    def __init__(self, num_input: int, num_hidden: int, num_output: int, gradient_method: str, dtype=np.float32):
        self.num_input = num_input
        self.num_hidden = num_hidden
        self.num_output = num_output
        self.dtype = dtype
        self.gradient_method = gradient_method
        self.var = {}
        self.diff = {}
        self.theta = {}
        self.gradient_w1: np.ndarray
        self.gradient_b1: np.ndarray
        self.gradient_w0: np.ndarray
        self.gradient_b0: np.ndarray

        self.init_params()

    def init_params(self):
        MU = 0
        SiGMA = 0.5
        self.theta['W0'] = np.random.normal(MU, SiGMA, size=(self.num_hidden, self.num_input))
        self.theta['W1'] = np.random.normal(MU, SiGMA, size=(self.num_output, self.num_hidden))
        self.theta['b0'] = np.random.normal(MU, SiGMA, size=(self.num_hidden, 1))
        self.theta['b1'] = np.random.normal(MU, SiGMA, size=(self.num_output, 1))

    def feed_forward(self, x: np.ndarray):

        assert x.shape[1] == self.num_input, "x shape is {}; it should be {}".format(x.shape, (self.num_input, 1))
        x = x.T
        self.var['x'] = x
        self.var['z1'] = self.theta['W0'] @ x + self.theta['b0']
        self.var['a1'] = np.log(1 + np.exp( self.var['z1']))
        self.var['z2'] = self.theta['W1'] @ self.var['a1'] + self.theta['b1']
        self.var['y_prob'] = softmax(self.var['z2'])
        self.var['y_pred'] = np.argmax(self.var['y_prob'], axis=0)

    def back_propogate(self, y: np.ndarray):
        S = y.size
        y_label = np.zeros((3, S))

        for i in range(y.size):
            y_label[y[i]][i] = 1

        # Calculate gradient for one sample
        gradient_w1_sum = np.zeros((3, self.num_hidden))
        gradient_b1_sum = np.zeros((3, 1))
        gradient_w0_sum = np.zeros((self.num_hidden, 4))
        gradient_b0_sum = np.zeros((self.num_hidden, 1))

        for s in range(S):
            '''Prepare variables for one sample '''
            ys = y_label[:, [s]]
            xs = self.var['x'][:, [s]]
            yprob = self.var['y_prob'][:, [s]]
            z2 = self.var['z2'][:, [s]]
            a1 = self.var['a1'][:, [s]]
            z1 = self.var['z1'][:, [s]]

            '''gradient w1 b1'''
            dldy = - ys * 1 / yprob
            sigma1, sigma2, sigma3 = softmax(z2)[0].item(), softmax(z2)[1].item(), softmax(z2)[2].item()
            dydz = np.array([[sigma1*(1-sigma1), -sigma1*sigma2, -sigma1*sigma3],
                             [-sigma1*sigma2, sigma2*(1-sigma2), -sigma2*sigma3],
                             [-sigma1*sigma3, -sigma2*sigma3, sigma3*(1-sigma3)]])
            e2 = dydz @ dldy
            gradient_w1 = e2 @ a1.T
            gradient_b1 = e2

            '''gradient w0 b0'''
            dzda = gradient_w1.T
            dadz = np.diag( np.exp(z1[:, 0]) / (1 + np.exp(z1[:, 0])))
            e1 = dadz @ dzda
            gradient_w0 = e1 @ e2 @ xs.T
            gradient_b0 = e1 @ e2

            '''save the sum'''
            gradient_w1_sum += gradient_w1
            gradient_b1_sum += gradient_b1
            gradient_w0_sum += gradient_w0
            gradient_b0_sum += gradient_b0

        self.gradient_w1 = gradient_w1_sum / S
        self.gradient_b1 = gradient_b1_sum / S
        self.gradient_w0 = gradient_w0_sum / S
        self.gradient_b0 = gradient_b0_sum / S

    def export_model(self):
        with open(f'model_{self.gradient_method}.json', 'w') as fp:
            json.dump({key: value.tolist() for key, value in self.theta.items()}, fp)


def task1():
    """ Neural Network

        Requirements for the plots:
            - ax[0] Plot showing the training loss for both variants
            - ax[1] Plot showing the training and test accuracy for both variants
    """

    # Create the models
    # Model using steepest descent
    net_GD = NN(4, 16, 3, gradient_method='GD')

    # training
    net_GD.feed_forward(x_train_g)
    net_GD.back_propogate(y_train_g)
    # Model using Nesterovs method
    net_NAG = NN(1, 1, 1, gradient_method='NAG')

    net_GD.export_model()
    net_NAG.export_model()

    # Configure plot
    fig = plt.figure(figsize=[12,6])
    axs = []
    axs.append(fig.add_subplot(121))
    axs.append(fig.add_subplot(122))

    axs[0].set_title('Loss')
    axs[0].grid()

    axs[1].set_title('Accuracy')
    axs[1].grid()
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

    
