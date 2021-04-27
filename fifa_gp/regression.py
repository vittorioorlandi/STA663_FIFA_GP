import cppimport
import pandas as pd
import numpy as np
import sys, os

# add fifa_gp to sys.path to find compiled cpp functions
for p in sys.path:
	if os.path.isdir(p + '/fifa_gp'):
		sys.path.append(p + '/fifa_gp')

# for running from inside the tar.gz file without package installation
if not any('fifa_gp' in p for p in sys.path):
	sys.path.append('fifa_gp')

class FIFA_GP:
	"""Implementation of Fast Increased Fidelity Approximate Gaussian Process Regressor."""
	def __init__(self):
		self.fifa_gp = cppimport.imp("sample")

	def fit(self, X, y, rho_choices = np.logspace(-2, 1), 
			regression = True, Gibbs_ls = False, default_MHkernel = True, numeric_MHkernel = -1.0, 
			a_f = 1, b_f = 1, a_tau = 1, b_tau = 1, tol = 1e-12, M = 20, save_fsamps = True, seed = 169,
			burnin = 1000, nsamps = 100, thin = 10, sig = 1, tau = 1, verbose = False):
		"""Runs a Gibbs sampler to fit the GP to (X, y)"""

		self.regression = regression
		self.X = X
		self.y = y
		self.tol = tol
		self.M = M
		self.nsamps = nsamps

		out = self.fifa_gp.samplegp(X, y, sig, rho_choices, tau, 
			regression, Gibbs_ls, default_MHkernel, numeric_MHkernel, 
			a_f, b_f, a_tau, b_tau, tol, M, save_fsamps, seed,
			burnin, nsamps, thin, verbose)

		self.tau = out[0]
		self.sig_f = out[1]
		self.rho = out[2]
		self.f = out[3]
		return self

	def predict_f_post(self, Xnew):
		"""Posterior samples of f."""
		if self.regression:
			return self.fifa_gp.predict_f(self.X, self.y, Xnew, 
				self.sig_f, self.rho, self.tau, 1.0, self.M, self.tol, self.nsamps)
		
		raise RuntimeError('Prediction in non-regression setting currently unavailable.')

	def predict_f_mean(self, Xnew):
		"""Posterior mean of f."""
		if self.regression:
			f_star_samps = self.fifa_gp.predict_f(self.X, self.y, Xnew, 
				self.sig_f, self.rho, self.tau, 1.0, self.M, self.tol, self.nsamps)
			
			f_star_mean = np.mean(f_star_samps, axis = 1)
			return f_star_mean

		raise RuntimeError('Prediction in non-regression setting currently unavailable.')

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

	def predict_y_post(self, Xnew):
		"""Predictive samples of new points."""
		f_star_samps = self.predict_f_post(Xnew)
		y_star_samps = f_star_samps + (1 / self.tau)[:, None]
		return y_star_samps

	def predict_y_mean(self, Xnew):
		"""Predictive means of new points."""
		return np.mean(self.predict_y_post(Xnew), axis = 1)

