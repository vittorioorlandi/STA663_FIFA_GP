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
	"""Implementation of Fast Increased Fidelity Approximate Gaussian Process Regressor.

	Parameters
	----------
	None

	Attributes
	----------
	After self.fit has been called (see self.fit for more details):
	regression: True if the responses are interpreted as noisy observations; False, otherwise.
	X: Covariates
	y: Scalar response
	tol: Tolerance for HODLR matrix approximation
	M: Maximum submatrix size for HODLR matrix approximation
	save_fsamps: True if samples of f were saved while sampling; False, otherwise.
	nsamps: Number of samples returned from the MCMC
	tau: Posterior samples of noise precision tau
	sig_f: Posterior samples of function standard deviation sigma_f
	f: Posterior samples of the function f
	"""
	def __init__(self):
		self.fifa_gp = cppimport.imp("sample")

	def fit(self, X, y, rho_choices = np.logspace(-2, 1), 
			regression = True, Gibbs_ls = False, default_MHkernel = True, numeric_MHkernel = -1.0, 
			a_f = 1, b_f = 1, a_tau = 1, b_tau = 1, tol = 1e-12, M = 20, save_fsamps = True, seed = 169,
			burnin = 1000, nsamps = 100, thin = 10, sig = 1, tau = 1, verbose = False):
		"""Runs a Gibbs sampler to fit a FIFA-GP to (X, y). 

		Parameters
		---------- 
		X: A numpy array of n x p covariate values. Currently only supports p = 1. 
		y: An n-dimensional numpy array containing the scalar responses for the units described by X.
		rho_choices: Possible values for the squared exponential kernel lengthscale to sample from.
		regression: True if the responses y are taken to be noisy observations; False, otherwise.
		Gibbs_ls: True if Gibbs sampling should be used to sample the lengthscale; otherwise, uses Metropolis Hastings.
		default_MHkernel: True if default values should be used to construct the Metropolis Hastings update for the lengthscale.
		numeric_MHkernel: Positive scaling factor to use in the construction of the Metropolis Hastings update for the lengthscale.
		a_f: Shape parameter for the gamma prior on the function precision. 
		b_f: Rate parameter for the gamma prior on the function precision.
		a_tau: Shape parameter for the gamma prior on the noise precision. 
		b_tau: Rate parameter for the gamma prior on the noise precision. 
		tol: Tolerance to be used in the HODLR approximation. 
		M: Maximum submatrix size for the HODLR approximation.
		save_fsamps: True if samples of f should be saved while sampling; False, otherwise.
		seed: Seed to fix random number generation.
		burnin: Number of initial burn in samples to throw away after sampling. 
		nsamps: Number of posterior samples to return. 
		thin: Thinning factor for the MCMC; every `thin` samples will be returned.
		sig: Starting value for function standard deviation sigma_f in the sampler. 
		tau: Starting value for noise precision tau in the sampler. 
		verbose: True to output sampler progress while running; False, otherwise.

		Returns
		-------
		self: returns an instance of self
		"""

		self.regression = regression
		self.X = X
		self.y = y
		self.tol = tol
		self.M = M
		self.nsamps = nsamps
		self.save_fsamps = save_fsamps

		out = self.fifa_gp.samplegp(X, y, sig, rho_choices, tau, 
			regression, Gibbs_ls, default_MHkernel, numeric_MHkernel, 
			a_f, b_f, a_tau, b_tau, tol, M, save_fsamps, seed,
			burnin, nsamps, thin, verbose)

		self.tau = out[0]
		self.sig_f = out[1]
		self.rho = out[2]
		if self.save_fsamps:
			self.f = out[3]
		return self

	def predict_f_post(self, Xnew):
		"""Returns posterior samples of f at new data points.
		
		Parameters
		----------
		Xnew: A numpy array of n_test x p new covariate values to predict at. Currently only supports p = 1.

		Returns
		-------
		An n_test x self.nsamps numpy array of predicted function values. 
		"""
		if self.regression:
			return self.fifa_gp.predict_f(self.X, self.y, Xnew, 
				self.sig_f, self.rho, self.tau, 1.0, self.M, self.tol, self.nsamps)
		
		raise RuntimeError('Prediction in non-regression setting currently unavailable.')

	def predict_f_mean(self, Xnew):
		"""Returns posterior mean of f at new data points.
		
		Parameters
		----------
		Xnew: A numpy array of n_test x p new covariate values to predict at. Currently only supports p = 1.

		Returns
		-------
		An n_test-dimensional numpy array of predicted function values. 
		"""
		if self.regression:
			f_star_samps = self.fifa_gp.predict_f(self.X, self.y, Xnew, 
				self.sig_f, self.rho, self.tau, 1.0, self.M, self.tol, self.nsamps)
			
			f_star_mean = np.mean(f_star_samps, axis = 1)
			return f_star_mean

		raise RuntimeError('Prediction in non-regression setting currently unavailable.')

	def get_params_post(self):
		"""Return posterior samples of parameters.

		Parameters
		----------
		None

		Returns
		-------
		A pandas data frame with self.nsamps many rows, each corresponding to a posterior sample for tau, rho, and sig_f. 
		"""
		params_post = pd.DataFrame(np.c_[self.tau, self.rho, self.sig_f], 
			columns = ['Noise variance', 'Kernel Length-Scale', 'Kernel Variance'])
		return params_post

	def get_params_mean(self):
		"""Return posterior means of parameters.

		Parameters
		----------
		None

		Returns
		-------
		A pandas data frame with a single row containing posterior means for tau, rho, and sig_f. 
		"""
		params_post = np.c_[self.tau, self.rho, self.sig_f]
		params_post_mean = pd.DataFrame(np.mean(params_post, axis = 0).reshape(1, -1), 
			columns = ['Noise variance', 'Kernel Length-Scale', 'Kernel Variance'])

		return params_post_mean

	def predict_y_post(self, Xnew):
		"""Returns posterior samples of the predicted response at new data points.
		
		Parameters
		----------
		Xnew: A numpy array of n_test x p new covariate values to predict at. Currently only supports p = 1.

		Returns
		-------
		An n_test x self.nsamps numpy array of predicted responses.
		"""
		f_star_samps = self.predict_f_post(Xnew)
		y_star_samps = f_star_samps + (1 / self.tau)[:, None]
		return y_star_samps

	def predict_y_mean(self, Xnew):
		"""Returns posterior mean of predicted responses at new data points.
		
		Parameters
		----------
		Xnew: A numpy array of n_test x p new covariate values to predict at. Currently only supports p = 1.

		Returns
		-------
		An n_test-dimensional numpy array of predicted responses.
		"""
		return np.mean(self.predict_y_post(Xnew), axis = 1)

