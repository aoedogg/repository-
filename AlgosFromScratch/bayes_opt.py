# Bayesian Optimization
# Based on tutorial: http://krasserm.github.io/2018/03/21/bayesian-optimization/

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern

noise = 0.2

class bopt:
    def __init__(self, bounds):
        self.bounds = bounds
        pass

    def acq(self, X, X_sample, Y_sample, gpr, xi=0.01):
        # Expected improvement acq function

        mu, sigma = gpr.predict(X, return_std=True)
        sigma = sigma.reshape(-1, 1)
        mu_sample_opt = np.max(Y_sample)
        #mu_sample_opt = np.max(gpr.predict(X_sample))

        with np.errstate(divide='warn'):
            imp = mu - mu_sample_opt - xi
            Z  = imp / sigma
            ei = imp * norm.cdf(Z) + sigma* norm.pdf(Z)
            ei[sigma == 0.0] = 0.0
        return ei

    def propose_location(self, X_sample, Y_sample, gpr, n_restarts=25):

        dim = X_sample.shape[1]
        min_val = 1
        min_x = None
        # wrapper
        def min_obj(X):
            # Minimization objective is the negative acquisition function
            # GPR function expect 2d array while scipy minimize take 1d array
            return -self.acq(X.reshape(-1, 1), X_sample, Y_sample, gpr)

        for x0 in np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_restarts, dim)):
            res = minimize(min_obj, x0=x0, bounds=self.bounds, method='L-BFGS-B')
            if res.fun < min_val:
                min_val = res.fun[0]
                min_x = res.x

        return min_x


def f(X, noise=noise):
    return -np.sin(3*X) - X**2 + 0.7*X + noise*np.random.randn(*X.shape)

bounds = np.array([[-1.0, 2.0]])
X = np.arange(bounds[:,0], bounds[:, 1], 0.01).reshape(-1, 1)
Y = f(X, 0)

X_init = np.array([[-0.9], [1.1]])
Y_init = f(X_init)
print(Y_init.shape)

# Plot optimization objective with noise level
plt.plot(X, Y, 'y--', lw=2, label='Noise-free objective')
plt.plot(X, f(X), 'bx', lw=1, alpha=0.1, label='Noisy samples')
plt.plot(X_init, Y_init, 'kx', mew=3, label='Initial samples')
plt.legend()
plt.show()

n = 20

# Initial samples for GP model
X_sample = X_init
Y_sample = Y_init

# Init GP
gpr = GaussianProcessRegressor()

# Init Bayes Opt
bayes = bopt(bounds)

for ii in range(n):

    # train GP model using currently available sampled data
    gpr.fit(X_sample, Y_sample)
    # Obtain next sampling point
    X_next = bayes.propose_location(X_sample, Y_sample, gpr)
    Y_next = f(X_next, noise)

    # Add sample to previous samples
    X_sample = np.vstack((X_sample, X_next))
    Y_sample = np.vstack((Y_sample, Y_next))


plt.plot(X, Y, 'y--', lw=2, label='Noise-free objective')
plt.plot(X_sample, Y_sample, 'kx', mew=3)
plt.show()