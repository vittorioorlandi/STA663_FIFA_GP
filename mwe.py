import numpy as np
import matplotlib.pyplot as plt
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

#discrete uniform prior on 10 values between 0.5 and 2.5. 
rho_min = 0.5
rho_max = 2.5
rho_len = 10
rho_choices = np.linspace(rho_min,rho_max,rho_len)

gp = FIFA_GP()
gp.fit(X,Y,
       1,rho_choices,1,
       True,False,
       True,-1,
       1,1,1,1,
       1e-12,20,True,169,"none",
       0,100,1,False)

fpred = gp.predict_f_mean(Xtest)

#show parameter summaries
print(gp.get_params_mean())
print(gp.get_params())

#show mean for test samle
print(np.mean(fpred, axis = 1))

#compute MSE on training data
mse_mean = ((Y.reshape(n,1) - gp.f)**2).mean()
print('MSE: ' + str(mse_mean))

