import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial
from regression import FIFA_GP

np.random.seed(196)

# create some fake data from (I think) square exponential kernel with tau=sigma=rho=1

n = 500 #for small n, the code gives a segfault when assembling the HODLR tree, things are more stable with big n
X = np.random.randn(n,1)
Xtest = np.random.randn(n//4,1)

D = scipy.spatial.distance_matrix(X,X)**2
cov = np.exp(-D/2)
Y = np.random.multivariate_normal(np.zeros(n), cov + 5 * np.eye(n))

rho_min = 0.5
rho_max = 2.5
rho_len = 100
rho_choices = [1, 1.2] #np.arange(rho_len)/rho_len*rho_max+rho_min

gp = FIFA_GP()
gp.fit(X,Y,
       1,rho_choices,1,
       True,False,
       True,-1,
       1,1,1,1,
       1e-12,20,True,169,"none",
       0,100,1,False)

fpred = gp.predict_f_mean(Xtest)

print('Successfully executed sample')

print(gp.get_params_mean())

print(gp.get_params())

# print(np.mean(gp.fstar, axis = 1)[:10]) # when axis = 0, all the same? 
# print(np.mean(fpred, axis = 0)[:10])

# print(np.mean(gp.fstar, axis = 0).shape)
print(np.mean(fpred, axis = 1))


# fig, ax = plt.subplots(1)
# plt.scatter(np.ravel(Xtest), np.mean(gp.fstar, axis = 1), c = 'red') # internal method
# plt.scatter(np.ravel(Xtest), np.mean(fpred, axis = 0), c = 'blue') # predict method
# plt.show()

# print(gp.fstar[:5, :5])
# print(fpred[:5, :5])

mse_mean = ((Y.reshape(n,1) - gp.f)**2).mean()
print('MSE: ' + str(mse_mean))

# import pickle
# with open('mwe_result.pk','wb') as f:
# 	pickle.dump(out,f)

## comment to note what the argument positions are (default pybind complilationd doesn't allow named arguments)
# out = fifa_gp.samplegp(X=X,Y=Y,Xtest=Xtest,
#                            sig=1,rho_choices = [0.5],tau = 1,
#                            regression = True,Gibbs_ls = False,
#                            default_MHkernel=True,numeric_MHkernel=-1,
#                            a_f=1,b_f=1,a_tau=1,b_tau=1,
#                            tol=1e-12,M=20,save_fsamps=True,seed=196,fpath="none",
#                            burnin=1000,nsamps=100,thin=10,verbose=False)

