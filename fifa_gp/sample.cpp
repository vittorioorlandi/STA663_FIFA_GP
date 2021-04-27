<%
cfg['compiler_args'] = ['-std=c++11','-DUSE_DOUBLE']
cfg['include_dirs'] = ['eigen', 'include']
cfg['sources'] = ['./HODLR_Matrix.cpp',
    './HODLR_Node.cpp',
    './HODLR_Tree.cpp',
    './HODLR_Tree_NonSPD.cpp',
    './HODLR_Tree_SPD.cpp',
    './KDTree.cpp',
    './sample_funs.cpp',
    './predict.cpp']
setup_pybind11(cfg)
%>

#include <squaredeMat.hpp> //squared exponential kernel
#include <squaredeP1Mat.hpp>
#include <random>
#include <chrono>
#include <iostream>
#include <fstream>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>
#include "HODLR_Tree.hpp"
#include "sample_funs.h"

using std::gamma_distribution;
using std::uniform_real_distribution;
using std::normal_distribution;

namespace py = pybind11;

void predict_module(py::module &);

Eigen::VectorXd samplefstar_HODLR(Eigen::MatrixXd X, Eigen::MatrixXd Y, Eigen::MatrixXd Xtest, double sig, double rho, double tau, HODLR_Tree* T, double multiplier, std::function<float()> &r_std_normal, int s) {
  
    /* 
       Sample a draw of the GP function f*|f, sig, rho, tau, x*.
       Assume a squared exponential Gaussian Process based on 
       observed function f at new test points x*.
    */

    int Ntest =  Xtest.rows();  
    int N =  X.rows();
    int D =  X.cols();
   
    double sigsq = pow(sig, 2.0);
    double tmpSSR;
    
    Eigen::VectorXd fstarsamp(Ntest);
    Eigen::VectorXd KobsNew(N);
    
    for (int i = 0; i < Ntest; i++) {
      Eigen::RowVectorXd Xtest_i = Xtest.row(i);
      
      // Covariance between X and Xtest
      for (int j = 0; j < N; j++) {
        Eigen::RowVectorXd tmp = X.row(j) - Xtest_i;
        tmpSSR = 0.0;
        for (int d = 0; d < D; d++) {
          tmpSSR = tmpSSR + pow(tmp(d), 2.0);
        }
        KobsNew(j) = sigsq * exp(- tmpSSR * rho);
      }

      // Variance at Xtest
      double kNewNew = sigsq + 1e-8;
      
      // Posterior mean and variance of f* at xtest(i)
      double sdstar = pow(kNewNew - (multiplier * KobsNew.transpose() * T->solve(tau * KobsNew))(0, 0), 0.5);
      double mustar = (multiplier * KobsNew.transpose() * T->solve(tau * Y))(0, 0);

      auto normal_samp = r_std_normal();

      fstarsamp(i) =  sdstar * normal_samp + mustar;     
    }
    
    return fstarsamp;
}

Eigen::VectorXd get_transition_probs_byind(Eigen::VectorXi rhovec_inds, double krnl, int ind_current) {
  /*
    Creates transition probability matrix for Metropolis Hastings update of lengthscale
  */
  int nrhos = rhovec_inds.rows();
  Eigen::VectorXd prop_prob(nrhos);
  for (int i = 0; i < nrhos; i++) {
    prop_prob(i) = exp(-krnl * pow((double)rhovec_inds(i) - (double)ind_current, 2.0));
  }
  prop_prob(ind_current) = 0.0;
  double sumInv = 1.0 / prop_prob.sum();
  prop_prob = sumInv * prop_prob;
  return prop_prob;
}

py::list sampleGP_HODLR(Eigen::MatrixXd X, Eigen::VectorXd Y, 
                          double sig, Eigen::VectorXd rho_choices, double tau,
                          bool regression=true,
                          bool Gibbs_ls=false, bool default_MHkernel=true, 
                          double numeric_MHkernel=-1.0,
                          double a_f=1, double b_f=1,
                          double a_tau=1, double b_tau=1, 
                          double tol=1e-12, int M=20, 
                          bool save_fsamps=true, unsigned int seed=169, 
                          int burnin=1000, int nsamps=100, int thin=10, bool verbose=false) {
  /*
    C++ sample hyperparameter and function draws 
    from a squared exponential Gaussian Process
    based on observed data y having precision tau.
    X   -matrix of locations
    Y   -matrix of observed data
    tau -data precision 
    sig -function std deviation
    rho -length-scale
    a_f and b_f - prior sig^(-2) ~ Ga(a_f/2, b_f/2)
    a_tau and b_tau - prior tau ~ Ga(a_tau/2, b_tau/2)
    tol -specified tolerance for accuracy of calculations
    Gibbs_ls: bool, whether to use Gibbs or MH to sample length scale
    M   -Max submatrix size
  */
  
  // For sampling random numbers
  std::mt19937 generator(seed);

  // For sampling standard uniforms (for MH acceptance evaluation)
  uniform_real_distribution<float> unif(0, 1); 
  std::function<float()> runif = bind(unif, generator);

  // For sampling standard normals (e.g. for sampling f)
  normal_distribution<float> normal(0, 1);
  std::function<float()> r_std_normal = bind(normal, generator);
  
  // Data sizes and initializing fsamp and params / transformed params
  int N = X.rows();

  Eigen::VectorXd KobsNew(N);
  Eigen::VectorXd fsamp(N);

  double sqrt_tau;
  if (!regression) { // If you observe non-noisy data
    fsamp = Y;
  } 
  else {
    sqrt_tau = pow(tau, 0.5);
  }

  double sigsq = pow(sig, 2.0);
  double prec_f = 1 / sigsq;
  
  // Set up HODLR details
  int n_levels = log(N / M) / log(2);
  bool is_sym = true; 
  bool is_pd  = true;

  // Create rho and matrices for each value option
  int nrhos = rho_choices.rows();
  Eigen::MatrixXd transition_probs_mat(nrhos,nrhos);
  bool fix_ls;
  int rho_ind;

  Eigen::VectorXd logdetK_all(nrhos);
  SQRExponential_Kernel* K_tmp;
  HODLR_Tree* S_tmp;
  std::vector<HODLR_Tree*> Svec;
  Svec.reserve(nrhos);

  if (nrhos != 1) {
    fix_ls = false; 
    rho_ind = round(nrhos / 2);
    if (!Gibbs_ls) {
      // Make transition probability matrix from each "view" of the data
      // (i.e., from each index at which you could make a proposal)
      bool cond1 = (numeric_MHkernel > 0); 
      double krnl;
      if (cond1 | default_MHkernel) {
        if (cond1) {
          krnl = numeric_MHkernel; 
        }

        if (default_MHkernel) {
          krnl = 1.0 / pow((double)nrhos, 1.0); 
        }

        Eigen::VectorXi rhovec_inds(nrhos);
        for (int i = 0; i < nrhos; i++) {
          rhovec_inds(i) = i;
        }
        for (int i = 0; i < nrhos; i++) { // Each col is "view" from that index.
          transition_probs_mat.col(i) = get_transition_probs_byind(rhovec_inds, krnl, i);
        }
      } else{
        printf("default_MHkernel must be TRUE or numeric_MHkernel needs to be set > 0(see function description for details)."); 
      }
    }
  } 
  else {
    fix_ls = true;
    rho_ind = 0;
    if (verbose) {
      printf("Length scale fixed to provided input: %.2f.\n", rho_choices(rho_ind));
      printf("To sample length scale from discrete options provide vector rho_choices.\n");
    }
  }

  // Svec here is K in the paper
  for (int i = 0; i < nrhos; i++) {
    // Set up squared exponential kernel K with sigf fixed to 1.
    K_tmp  = new SQRExponential_Kernel(X, N, 1, rho_choices(i));
    S_tmp  = new HODLR_Tree(n_levels, tol, K_tmp); // Without noise (i.e., Sigma)
    Svec.push_back(S_tmp);
    Svec[i]->assembleTree(is_sym, is_pd);
    Svec[i]->factorize();
    logdetK_all(i) = Svec[i]->logDeterminant();
  }

  // Initialize first sample of rho
  double rho = rho_choices(rho_ind); 

  // Set up storage
  int samp_count = 0;
  int total_draws = burnin + nsamps * thin;
  Eigen::VectorXd tau_save(nsamps);
  Eigen::VectorXd sigf_save(nsamps);
  Eigen::VectorXd rho_save(nsamps);
  int nsamps_f = nsamps;
  if (!save_fsamps) {
    nsamps_f = 0;
  }
  Eigen::MatrixXd f_save(N, nsamps_f);
  
  for (int s = 0; s < total_draws; s++) {
    bool save_samps = (s >= burnin) & (((s + 1) % thin) == 0);
  
    // Non-noisy data
    if (regression) {
      
      // Assemble the kernel with noise term tau
      SQRExponentialP1_Kernel* L  = new SQRExponentialP1_Kernel(X, N, sig, rho, tau);
      HODLR_Tree* T = new HODLR_Tree(n_levels, tol, L); // With noise (i.e. Sigma + I/tau)
      T->assembleTree(is_sym, is_pd);
      T->factorize();

      // Sample f
      fsamp = samplef_HODLR(X, Y, sig, tau, T, Svec[rho_ind], r_std_normal);

      delete T;
      delete L;
    }
    
    // Sample function variance
    prec_f = sample_prec_f(Svec[rho_ind], fsamp, a_f, b_f, generator);
    sig = pow(prec_f, -0.5); 
    sigsq = pow(sig, 2.0);

    // Sample length scale
    if (!fix_ls) {
      if (Gibbs_ls) { 
        rho_ind = sample_rho_gibbs(prec_f, fsamp, Svec, logdetK_all, generator);
      } 
      else { 
        rho_ind = sample_rho_mh(rho_ind, transition_probs_mat, prec_f, fsamp, Svec, logdetK_all, generator, runif);
      }
      rho = rho_choices(rho_ind);
    }
    
    // Sample noise precision
    if (regression) {
      tau = sample_tau(Y, fsamp, a_tau, b_tau, generator);
      sqrt_tau = pow(tau, 0.5);
    }
    
    if ((s == burnin) and verbose) {
      printf("Finished with burnin, beginning sampling. \n");
    }
    if ((10 * (s + 1) % total_draws == 0) and verbose) {
      printf("%d%% done \n", 100 * (s + 1) / total_draws);
    }
    
    // Save samples if you're past burnin
    if (save_samps) {
      if (regression) {
        tau_save(samp_count) = tau;
      }
      sigf_save(samp_count) = sig;
      rho_save(samp_count) = rho;
      
      if (save_fsamps) {
        f_save.col(samp_count) = fsamp;
      }

      samp_count = samp_count + 1;
    }
    
  } // END for (int s=0; s < total_draws; s++)

  delete S_tmp;
  delete K_tmp;
  std::vector<HODLR_Tree*>().swap(Svec);
  
  py::list out;
  out.append(tau_save);
  out.append(sigf_save);
  out.append(rho_save);
  out.append(f_save);
  return out;
}

PYBIND11_MODULE(sample, m) {
    m.doc() = "C++ Implementation of FIFA-GP.";
    m.def("samplegp", &sampleGP_HODLR, "1-d GP.");
    predict_module(m);
}