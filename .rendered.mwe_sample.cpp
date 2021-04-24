

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

// VO 
using std::gamma_distribution;
using std::uniform_real_distribution;
using std::normal_distribution;

namespace py = pybind11;

void predict_module(py::module_ &);

// // AZ: draw random samples from f
// // S is an argument that in the code should be Svec[rho_ind]
// // Eigen::VectorXd samplef_HODLR(Eigen::MatrixXd X, Eigen::MatrixXd Y, double sig, double tau, HODLR_Tree* T, HODLR_Tree* S, std::default_random_engine generator) {
// Eigen::VectorXd samplef_HODLR(Eigen::MatrixXd X, Eigen::MatrixXd Y, double sig, double tau, HODLR_Tree* T, HODLR_Tree* S, std::function<float()> &r_std_normal) {
  
//   // Create the standard normal generator
//   // normal_distribution<double> norm(0, 1);
//   // std::mt19937 rng;
//   // auto r_std_normal = bind(norm, rng);
  
//   //// Extract the number of rows from X
//   int N =  X.rows();
  
//   //// Initialize the empty vector of samples
//   Eigen::VectorXd fsamp(N);
  
//   //// Ccompute sigma square
//   double sigsq = pow(sig, 2.0);
//   double sqrt_tau = pow(tau, 0.5);
  
//   //// STEP 1
//   ////// Sample a draw of the GP function f|y,sig,rho,tau.
//   ////// Assume a squared exponential Gaussian Process with params sigf and rho, 
//   ////// based on observed data y having precision tau.
  
//   // Algorithm 1: Sample a, b ~ N(0, I)
//   Eigen::MatrixXd A(N, 1);
//   Eigen::MatrixXd B(N, 1);
//   for (int ii = 0; ii < N; ii++) {
//     A(ii,0) = r_std_normal();
//     B(ii,0) = r_std_normal();
//   }
  
//   //// STEP 2 - Obtain the samples     
  
//   // Get mean
//   Mat mu =  T->solve(tau * Y);  
//   mu     =  sigsq * S->matmatProduct(mu);
  
//   // We want MVN ~ N(0,Sigma*(Sigma*tau+I)^-1) 
//   //  1) Set fsamp = \sqrt(tau) * Sigma * A + W * B, where A, B ~ N (0, I_N).
//   //     Equivalent to \sqrt(tau) * \sigma^2 * S * A + \sigma * W * B, 
//   //     where S = Sigma/(\sigma^2) (i.e. S is the same kernel but with function var 1).
//   fsamp = sqrt_tau * sigsq * S->matmatProduct(A) + sig * S->symmetricFactorProduct(B);
//   // Now fsamp ~ (0, Sigma*(tau*Sigma+I)).
//   //  2) Multipliy (tau*Sigma+I)^-1 by fsamp. 
//   fsamp = T->solve(fsamp); // fsamp ~ (0, (tau*Sigma+I)^-1*Sigma)
//   // We now have fsamp ~ N(0,Sigma*(Sigma*tau+I)^-1) 
//   // bc both matrices are symmetric and their prod is symmetric so they commute.
//   //  3) Add the mean Sigma*(Sigma*tau+I)*(tau*Y); 
//   fsamp = fsamp + mu;
//   //  4) Now fsamp is our new Gibbs Sampled GP 
  
//   return fsamp;
// }

// AZ: draw random samples from fstar
Eigen::VectorXd samplefstar_HODLR(Eigen::MatrixXd X, Eigen::MatrixXd Y, Eigen::MatrixXd Xtest, double sig, double rho, double tau, HODLR_Tree* T, double multiplier, std::function<float()> &r_std_normal, int s) {
  
    // printf("SAMPLE: sig, tau, rho: s = %d: %.2f \t%.2f \t%.2f\n", s, sig, tau, rho);

    // Create the standard normal generator
    // normal_distribution<double> norm(0, 1);
    // std::mt19937 rng;
    // auto r_std_normal = bind(norm, rng);
    
    ////Number of observations in train and test
    int Ntest =  Xtest.rows();  
    int N =  X.rows();
    int D =  X.cols();
   
    
    // Sigma square and SSR
    double sigsq = pow(sig, 2.0);
    double tmpSSR;
    
    // Allocate fstarsamp
    Eigen::VectorXd fstarsamp(Ntest);
    Eigen::VectorXd KobsNew(N);
    
    // VO: Should be able to do this through matrix operations
    // AZ: cycle through the observations in the test set
    for (int i = 0; i < Ntest; i++) {
      ////// Sample a draw of the GP function f*|f,sig,rho,tau,x*.
      ////// Assume a squared exponential Gaussian Process with params sigf and rho, 
      ////// based on observed function f at new test points x*.
      Eigen::RowVectorXd Xtest_i = Xtest.row(i);
      // Get covariance between X and Xtest
      for (int j = 0; j < N; j++) {
        Eigen::RowVectorXd tmp = X.row(j) - Xtest_i;
        tmpSSR = 0.0;
        for (int d = 0; d < D; d++) {
          tmpSSR = tmpSSR + pow(tmp(d), 2.0);
        }
        KobsNew(j) = sigsq * exp(- tmpSSR * rho);
      }
      // Get variance at Xtest
      double kNewNew = sigsq + 1e-8;
      // Get posterior mean and variance of f* at point xtest(i)
      double sdstar = pow(kNewNew - (multiplier * KobsNew.transpose() * T->solve(tau * KobsNew))(0,0), 0.5);
      double mustar = (multiplier * KobsNew.transpose() * T->solve(tau * Y))(0,0);


      auto normal_samp = r_std_normal();

      if (i == 3 && s % 10 == 0) {
        // printf("SAMPLE: mu = %.3f, sd = %.3f\n", mustar, sdstar);
        // std::cout << normal_samp << std::endl;
      }
      

      fstarsamp(i) =  sdstar * normal_samp + mustar;
      //std::normal_distribution<double> distribution(mustar, sdstar);
      //fstarsamp(i) = distribution(generator);
      
    }
    
    return fstarsamp;
}

Eigen::VectorXd get_transition_probs_byind(Eigen::VectorXi rhovec_inds, double krnl, int ind_current) {
  int nrhos = rhovec_inds.rows();
  Eigen::VectorXd prop_prob(nrhos);
  for(int i = 0; i < nrhos; i++) {
    prop_prob(i) = exp(-krnl * pow((double)rhovec_inds(i) - (double)ind_current, 2.0));
  }
  prop_prob(ind_current) = 0.0;
  double sumInv = 1.0 / prop_prob.sum();
  prop_prob = sumInv * prop_prob;
  return prop_prob;
}

// // VO: Can we always just pass in the std::mt19937 and redefine the standard normals in the above to get what we want? 
// double sample_prec_f(HODLR_Tree* S, Eigen::VectorXd fsamp, double a_f, double b_f, std::mt19937 rng) {
//     int N = fsamp.rows();
//     // Equation (6)
//     // Svec is K_{σ_f, ρ} in the paper 
//     Mat tmp1 =  S->solve(fsamp); // = S^(-1) f, result is a n x 1 matrix
//     Mat tmp2 = fsamp.transpose() * tmp1;


//     std::gamma_distribution<float> distribution( (N + a_f) / 2.0, 2.0 / (tmp2(0, 0) + b_f) );
//     double prec_f = distribution(rng);
//     // std::cout << prec_f << std::endl; 
//     return prec_f; 
// }

//  double sample_tau(Eigen::VectorXd Y, Eigen::VectorXd fsamp, double a_tau, double b_tau, std::mt19937 rng) { 
//     int N = fsamp.rows();
//     Mat RSS = (Y - fsamp).transpose() * (Y - fsamp); // 1 x 1
//     std::gamma_distribution<float> distribution( (N + a_tau) / 2.0, 2.0 / (RSS(0,0) + b_tau ) );
//     double tau = distribution(rng);
//     return tau;
// }

// int sample_rho_gibbs(double prec_f, Eigen::VectorXd fsamp, std::vector<HODLR_Tree*>Svec, Eigen::VectorXd logdetK_all, std::mt19937 rng) {
//   int nrhos = logdetK_all.rows();
//   Eigen::VectorXd log_prob(nrhos);
//   for (int i = 0; i < nrhos; i++) {
//     Mat quad_tmp = prec_f * fsamp.transpose() * Svec[i]->solve(fsamp);
//     log_prob(i) = -0.5 * (logdetK_all(i) + quad_tmp(0, 0));
//   }
//   double maxlg = log_prob.maxCoeff();
//   Eigen::VectorXd weights(nrhos);
//   // https://stats.stackexchange.com/questions/66616/converting-normalizing-very-small-likelihood-values-to-probability
//   for (int i = 0; i < nrhos; i++) {
//     weights(i) = exp(log_prob(i) - maxlg); 
//   }
//   weights = weights / (weights.sum());
//   std::discrete_distribution<int> dist(weights.data(), weights.data() + weights.size());
//   int rho_ind = dist(rng);
//   return rho_ind;
// }

// // also pass rng by reference? 
// int sample_rho_mh(int rho_ind, Eigen::MatrixXd transition_probs_mat, double prec_f, Eigen::VectorXd fsamp, std::vector<HODLR_Tree*> Svec, Eigen::VectorXd logdetK_all, std::mt19937 rng, std::function<float()> &runif) {
//   Eigen::VectorXd prob_old_to_new = transition_probs_mat.col(rho_ind);
//   std::discrete_distribution<int> dist(prob_old_to_new.data(), prob_old_to_new.data() + prob_old_to_new.size());
//   int ind_new = dist(rng);
//   double lQoldToNew = log(prob_old_to_new(ind_new));
//   double lQnewToOld = log(transition_probs_mat(rho_ind, ind_new));
//   Mat quad_tmp_old = prec_f * fsamp.transpose() * Svec[rho_ind]->solve(fsamp);
//   double llold = -0.5 * (logdetK_all(rho_ind) + quad_tmp_old(0, 0));
//   Mat quad_tmp_new = prec_f * fsamp.transpose() * Svec[ind_new]->solve(fsamp);
//   double llnew = -0.5 * (logdetK_all(ind_new) + quad_tmp_new(0, 0));
//   // Acceptance probability A is min(1, pi(y) Q(x_t|y) / (pi(x_t) Q(y|x_t)) )
//   // log(A) = min(0, log(p(y)) + log(Q(x_t|y)) - log(pi(x_t)) -log(Q(y|x_t)) )
//   // where y is the "proposed" value for x_{t+1}
//   double lA = llnew + lQnewToOld - llold - lQoldToNew;

//   // VO Substitution of R uniform function
//   if (runif() < exp(lA)) { // if lA>0, this will always eval
//     rho_ind = ind_new;
//   }
//   return rho_ind;
// }
/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////


py::list sampleGP_HODLR(Eigen::MatrixXd X, Eigen::VectorXd Y, 
                          double sig, Eigen::VectorXd rho_choices, double tau,
                          bool regression=true,
                          bool Gibbs_ls=false, bool default_MHkernel=true, 
                          double numeric_MHkernel=-1.0,
                          double a_f=1, double b_f=1,
                          double a_tau=1, double b_tau=1, 
                          double tol=1e-12, int M=20, 
                          bool save_fsamps=true, unsigned int seed=99999, std::string fpath="none",
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
  M   -Max submatrix size
  */

  // VO: Gibbs_ls is a boolean that is true if you supplied several fixed ρ values to sample from
  //  bc this allows you to gibbs sample instead of MH to update to "arbitrary" values 
  
  // Start timer (whole function time)
  auto start = std::chrono::high_resolution_clock::now();
  
  // For sampling random numbers
  // std::default_random_engine generator (seed);
  std::mt19937 generator(seed);

  // if (seed != 99999) {
  //   rng.seed(seed);
  //   printf("Random number generator seed set to provided input.\n");
  // } 
  // else {
  //   std::random_device rd;
  //   rng.seed(rd());
  // }

  // For sampling standard uniforms (for MH acceptance evaluation)
  uniform_real_distribution<float> unif(0, 1); 
  // VO compiler complains if I pass auto in function prototype. 
  // std::function<float()> runif = bind(unif, rng);
  std::function<float()> runif = bind(unif, generator);
  // auto runif = bind(unif, rng);

  normal_distribution<float> normal(0, 1);
  std::function<float()> r_std_normal = bind(normal, generator);
  
  // For debugging, will save file to fpath
  bool debug = (fpath.std::string::compare("none") != 0);
  std::string fname;
  
  // Data sizes and initializing fsamp and params / transformed params
  int N = X.rows();
  
  // VO: Now in auxiliary function
  // int D = X.cols();
  
  // int Ntest =  Xtest.rows();
  Eigen::VectorXd KobsNew(N);
  Eigen::VectorXd fsamp(N);
  // Eigen::VectorXd fstarsamp(Ntest);

  // VO: Now in auxiliary function 
  // double tmpSSR;

  double sqrt_tau;
  if(!regression){ // If you observe non-noisy data
    fsamp = Y;
  } else{
    sqrt_tau = pow(tau, 0.5);
  }
  double sigsq = pow(sig, 2.0);
  double prec_f = 1/sigsq;
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
  Svec.reserve( nrhos );
  if (nrhos != 1) {
    fix_ls = false; 
    rho_ind = round(nrhos/2);
    if (!Gibbs_ls) {
      // Make transition probability matrix from each "view" of the data
      // (i.e., from each index at which you could make a proposal)
      bool cond1 = (numeric_MHkernel > 0); 
      double krnl;
      if (cond1 | default_MHkernel){
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
  } else {
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

  double rho = rho_choices(rho_ind); // Initialize rho.
  
  // Start timer for sampling
  auto start_samps = std::chrono::high_resolution_clock::now();

  // Set up empty matrices to house results and initialize count
  int samp_count = 0;
  int total_draws = burnin + nsamps * thin;
  Eigen::VectorXd tau_save(nsamps);
  Eigen::VectorXd sigf_save(nsamps);
  Eigen::VectorXd rho_save(nsamps);
  int nsamps_f = nsamps;
  if (!save_fsamps){
    nsamps_f = 0;
  }

  Eigen::MatrixXd f_save(N, nsamps_f);
  // Eigen::MatrixXd fstar_save(Ntest, nsamps);
  
  for (int s = 0; s < total_draws; s++) {
    bool save_samps = (s >= burnin) & (((s + 1) % thin) == 0);
  
    if (regression) {
      
      // Assemble the kernel with noise term tau
      SQRExponentialP1_Kernel* L  = new SQRExponentialP1_Kernel(X, N, sig, rho, tau);
      HODLR_Tree* T = new HODLR_Tree(n_levels, tol, L); // With noise (i.e. Sigma + I/tau)
      T->assembleTree(is_sym, is_pd);
      T->factorize();
        
      ////// Sample a draw of the GP function f|y,sig,rho,tau.
      ////// Assume a squared exponential Gaussian Process with params sigf and rho, 
      ////// based on observed data y having precision tau.
      
      // AZ replace these code with 
      fsamp = samplef_HODLR(X, Y, sig, tau, T, Svec[rho_ind], r_std_normal);

      // if (save_samps & (Ntest > 0)) {
      //   // std::cout << "getting here" << std::endl;
      //   fstarsamp = samplefstar_HODLR(X, Y, Xtest, sig, rho, tau, T, 1, r_std_normal, s);
      //   // std::cout << "getting there" << std::endl;
      // }
      
      delete T;
      delete L;
      
    }

    // if ((!regression) & save_samps & (Ntest > 0)) {
    //   fstarsamp = samplefstar_HODLR(X, Y, Xtest, sig, rho, 1, Svec[rho_ind], prec_f, r_std_normal, s);
    // }
    
    ////// Sample the function variance for the GP sig^(-2)|f,rho,tau. 
    prec_f = sample_prec_f(Svec[rho_ind], fsamp, a_f, b_f, generator);
    sig = pow(prec_f, -0.5); // Reparameterize
    sigsq = pow(sig, 2.0);

    ////// Sample the length-scale for the GP rho|f,sig,tau.
    if (!fix_ls) {
      if (Gibbs_ls) { // Use Gibbs sampling, loop over all rho choices
        //  changed from rng
        rho_ind = sample_rho_gibbs(prec_f, fsamp, Svec, logdetK_all, generator);
      } 
      else { // Use MH step, making it so you don't have to do tons of inversions!
        // changed from rng
        rho_ind = sample_rho_mh(rho_ind, transition_probs_mat, prec_f, fsamp, Svec, logdetK_all, generator, runif);
      }
      rho = rho_choices(rho_ind);
    }
    
    ////// Sample the noise precision for the regression tau|y,sig,rho.
    // Eq (5)
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

      // if (Ntest > 0) {
      //   fstar_save.col(samp_count) = fstarsamp;
      // }
      samp_count = samp_count + 1;
    }
    
    // In debugging mode, output draws to file
    if (debug) {
      fname = fpath;
      fname.std::string::append("samp_");
      fname.std::string::append(std::to_string(s));
      fname.std::string::append(".txt");
      std::ofstream saveFile (fname);
      saveFile << tau; saveFile << '\n';
      saveFile << sig; saveFile << '\n';
      saveFile << rho; saveFile << '\n';
      if (save_fsamps) {for(int i = 0; i < N; i++) {saveFile << fsamp(i); saveFile << '\n';}}
      saveFile.close();
    }
    
  } // END for(int s=0; s < total_draws; s++)

  delete S_tmp;
  delete K_tmp;
  std::vector<HODLR_Tree*>().swap(Svec);
  
  // Stop timer and get duration for whole thing and sampling
  auto stop = std::chrono::high_resolution_clock::now(); 
  std::chrono::duration<float> duration_all = stop - start;
  std::chrono::duration<float> duration_samp = stop - start_samps;
  
  py::list out;
  out.append(tau_save);
  out.append(sigf_save);
  out.append(rho_save);
  out.append(f_save);
  // out.append(fstar_save);
  out.append(duration_all.count()); // Time all, in seconds
  out.append(duration_samp.count()); // Time sampling, in seconds

  return out;
}

PYBIND11_MODULE(mwe_sample, m) {
    m.doc() = "C++ Implementation of FIFA-GP.";
    m.def("samplegp", &sampleGP_HODLR, "1-d GP.");
    predict_module(m);
}