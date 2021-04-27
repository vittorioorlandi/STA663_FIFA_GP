import numpy as np
import scipy.spatial
from fifa_gp.regression import FIFA_GP

np.random.seed(196)

# Simulate data from a square exponential kernel with sigma=rho=1, tau = 1/5, and one covariate

n = 500 
X = np.random.randn(n, 1)
Xtest = np.random.randn(n // 4, 1)

D = scipy.spatial.distance_matrix(X, X) ** 2
cov = np.exp(-D / 2)
Y = np.random.multivariate_normal(np.zeros(n), cov + 5 * np.eye(n))

gp = FIFA_GP()
gp.fit(X, Y)

# Compute mean on test sample
fpred = gp.predict_f_mean(Xtest)

# Compute mean predicted outcome on test sample
ypred = gp.predict_y_mean(Xtest)

# Show parameter posterior summaries
print(gp.get_params_mean())

# Compute MSE on training data
mse_mean = ((Y.reshape(n, 1) - gp.f) ** 2).mean()
print('MSE: ' + str(mse_mean))

