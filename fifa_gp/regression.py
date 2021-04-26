import cppimport
import pandas as pd
import numpy as np
import sys, os

for p in sys.path:
	if os.path.isdir(p + '/fifa_gp'):
		sys.path.append(p + '/fifa_gp')

if not any('fifa_gp' in p for p in sys.path):
	sys.path.append('fifa_gp')

class FIFA_GP:
	"""Implementation of Fast Increased Fidelity Approximate Gaussian Process Regressor."""
	def __init__(self):
		self.fifa_gp = cppimport.imp("sample")

	def fit(self, X, y, sig, rho_choices, tau, 
			regression, Gibbs_ls, default_MHkernel, numeric_MHkernel, 
			a_f, b_f, a_tau, b_tau, tol, M, save_fsamps, seed, fpath,
			burnin, nsamps, thin, verbose = False):
		"""Runs a Gibbs sampler to fit the GP to (X, y)"""

		self.regression = regression
		self.X = X
		self.y = y
		self.tol = tol
		self.M = M
		self.nsamps = nsamps

  		# tau, sig_f, rho, f, fstar, total time, sampling time
		out = self.fifa_gp.samplegp(X, y, sig, rho_choices, tau, 
			regression, Gibbs_ls, default_MHkernel, numeric_MHkernel, 
			a_f, b_f, a_tau, b_tau, tol, M, save_fsamps, seed, fpath,
			burnin, nsamps, thin, verbose)

		self.tau = out[0]
		self.sig_f = out[1]
		self.rho = out[2]
		self.f = out[3]
		# self.fstar = out[4]
		return self

	def predict_f_mean(self, Xnew):
		# for x in [self.X, self.y, Xnew, self.sig_f, self.rho, self.tau, 1.0, self.M, self.tol]:
			# print(type(x))
		"""Posterior mean of f."""
		# return np.random.normal(size = (10, 10))

		if self.regression:
			return self.fifa_gp.predict_f(self.X, self.y, Xnew, 
				self.sig_f, self.rho, self.tau, 1.0, self.M, self.tol, self.nsamps)

		else:
			sys.exit('Prediction in non-regression setting unavailable')

	def predict_f_post(self, X):
		"""Posterior samples of f."""
		return

	def get_params_post(self):
		"""Posterior samples of parameters."""
		params_post = pd.DataFrame(np.c_[self.tau, self.rho, self.sig_f], 
			columns = ['Noise variance', 'Kernel Length-Scale', 'Kernel Variance'])
		return params_post

	def get_params(self):
		"""Posterior samples of parameters."""
		params_post = np.c_[self.tau, self.rho, self.sig_f]
		params_post = pd.DataFrame(params_post,
			columns = ['Noise variance', 'Kernel Length-Scale', 'Kernel Variance'])

		return params_post

	def get_params_mean(self):
		"""Posterior means of parameters."""
		params_post = np.c_[self.tau, self.rho, self.sig_f]
		params_post_mean = pd.DataFrame(np.mean(params_post, axis = 0).reshape(1, -1), 
			columns = ['Noise variance', 'Kernel Length-Scale', 'Kernel Variance'])

		return params_post_mean

	def predict_y_mean(self, X):
		"""Predictive means of new points."""
		return
	def predict_y_post(self, X):
		"""Predictive samples of new points."""
		return 