# Python FIFA-GP

This package contains a python module implementing the **F**ast **I**ncreased **F**idelity **A**pproximate **G**aussian **P**rocess (FIFA-GP) algorithm from [Moran and Wheeler (2020)](https://arxiv.org/abs/2006.06537) in C++. 
A GP with the following structure is estimated: 

![equation](https://latex.codecogs.com/png.latex?y_i%20%3D%20f%28%5Cmathbf%7Bx%7D_i%29%20&plus;%20e_i%2C%20%5Ctext%7Bwhere%20%7D%20e_i%20%5Csim%20N%280%2C%5Ctau%5E%7B-1%7D%29)

![equation](https://latex.codecogs.com/png.latex?%5Cmathbf%7Bf%7D%20%5Csim%20N%280%2C%5Cmathbf%7BK%7D%29%2C%20%5Ctext%7Bwhere%20%7D%20K_%7Bij%7D%20%3D%20%5Csigma%5E2%20%5Cexp%5B-1/2%28%5Cmathbf%7Bx%7D_i%20-%20%5Cmathbf%7Bx%7D_j%29%27%20%5Ctext%7Bdiag%7D%28%5Crho%5E%7B-2%7D%29%20%28%5Cmathbf%7Bx%7D_i%20-%20%5Cmathbf%7Bx%7D_j%29%5D)

Currently hosted on the test PyPi, this package is installable via pip with the below command: 

```
pip install --index_url https://test.pypi.org/simple/ fifa-gp
```

A basic example is shown below, where X is an $n \times p$ numpy array of covariates and $Y$ is an $n \times 1$ array of outputs. 
See mwe.py for a more details. 
The *first* time you call the function `FIFA_GP`, cppimport will check whether you have a compiled C++ module to sample the GP. 
If not, the C++ files will be compiled for you (this will take a few seconds). 
All subsequent calls to `FIFA_GP` will use that complied file and initialization will be near instantaneous. 

```python
import numpy as np
import scipy.spatial
from fifa_gp.regression import FIFA_GP

np.random.seed(196)

# simulate data from a square exponential kernel with sigma=rho=1, tau = 1/5, and one covariate

n = 500 
X = np.random.randn(n,1)
Xtest = np.random.randn(n//4,1)

D = scipy.spatial.distance_matrix(X,X)**2
cov = np.exp(-D/2)
Y = np.random.multivariate_normal(np.zeros(n), cov + 5 * np.eye(n))

gp = FIFA_GP()

# Fit FIFA-GP to the data, using default prior and MCMC parameters
gp.fit(X, Y)

# Predict f for observations in Xtest using the posterior predictive mean
gp.predict_f_mean(Xtest)

# Predict outcomes for units in Xtest using the posterior predictive mean
gp.predict_y_mean(Xtest)

# Posterior means of the GP parameters
gp.get_params_mean()

#compute MSE on training data
MSE = ((Y.reshape(n, 1) - gp.f) ** 2).mean()
print('MSE: ' + str(MSE))
```