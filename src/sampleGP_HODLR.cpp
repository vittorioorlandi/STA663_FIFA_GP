#include <squaredeMat.hpp> //squared exponential kernel
#include <squaredeP1Mat.hpp>
#include <random>
#include <chrono>
#include <iostream>
#include <fstream>
#include "sampleMHfuncs.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

namespace py = pybind11;

py::list sampleGP_HODLR(Eigen::MatrixXd X, Eigen::VectorXd Y, Eigen::MatrixXd Xtest, 
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
  std::default_random_engine generator;
  std::mt19937 rng;
  if (seed != 99999) {
    rng.seed(seed);
    printf("Random number generator seed set to provided input.\n");
  } 
  else {
    std::random_device rd;
    rng.seed(rd());
  }

  // For sampling standard normals 
  std_normal<double> norm(0, 1);
  auto r_std_normal = bind(rnorm, rng);
  
  // For debugging, will save file to fpath
  bool debug = (fpath.std::string::compare("none")!=0);
  std::string fname;
  
  // Data sizes and initializing fsamp and params / transformed params
  int N =  X.rows();
  int D =  X.cols();
  int Ntest =  Xtest.rows();
  Eigen::VectorXd KobsNew(N);
  Eigen::VectorXd fsamp(N);
  Eigen::VectorXd fstarsamp(Ntest);
  double tmpSSR;
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
      bool cond1 = (numeric_MHkernel>0); 
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
  Eigen::MatrixXd fstar_save(Ntest, nsamps);
  
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
      
      // VO: Set up random normal vectors.
      // Algorithm 1: Sample a, b ~ N(0, I)
      // This chunk seems like it could def be simplified. Commenting it out and subbing the chunk after. 
      // NumericVector tWA = rnorm(N, 0, 1);
      // NumericVector tWB = rnorm(N, 0, 1);
      // Eigen::Map<Eigen::MatrixXd> ttWA(Rcpp::as<Eigen::Map<Eigen::MatrixXd>>(tWA)); 
      // Eigen::Map<Eigen::MatrixXd> ttWB(Rcpp::as<Eigen::Map<Eigen::MatrixXd>>(tWB)); 
      // Eigen::MatrixXd A = ttWA;
      // Eigen::MatrixXd B = ttWB;

      Eigen::VectorXd A(N);
      Eigen::VectorXd B(N);
      for (int ii = 0; ii < N; ii++) {
        A << r_std_normal();
        B << r_std_normal();
      }
      
      // Get mean
      Mat mu =  T->solve(tau * Y);  
      mu     =  sigsq * Svec[rho_ind]->matmatProduct(mu);
      
      // VO: Difference between Svec and T? Maybe just approximation? 
      // VO: Sigma here is K, or K_tilde, in the paper 
      // VO: Svec here is such that sigsq * Svec is K or K_tilde
      // We want MVN ~ N(0,Sigma*(Sigma*tau+I)^-1) 
      //  1) Set fsamp = \sqrt(tau) * Sigma * A + W * B, where A, B ~ N (0, I_N).
      //     Equivalent to \sqrt(tau) * \sigma^2 * S * A + \sigma * W * B, 
      //     where S = Sigma/(\sigma^2) (i.e. S is the same kernel but with function var 1).
      // VO: This is where the "Construct W" in algo 1 of the paper implicitly happens 
      // VO: At this stage, fsamp = Z in notatio of algo 1 in the paper
      fsamp = sqrt_tau * sigsq * Svec[rho_ind]->matmatProduct(A) + sig * Svec[rho_ind]->symmetricFactorProduct(B);
      // Now fsamp ~ (0, Sigma*(tau*Sigma+I)).
      //  2) Multipliy (tau*Sigma+I)^-1 by fsamp. 
      // VO: This works because of the relation noted above Algo 1 in the paper 
      fsamp = T->solve(fsamp); // fsamp ~ (0, (tau*Sigma+I)^-1*Sigma)
      // We now have fsamp ~ N(0,Sigma*(Sigma*tau+I)^-1) 
      // bc both matrices are symmetric and their prod is symmetric so they commute.
      //  3) Add the mean Sigma*(Sigma*tau+I)*(tau*Y); 
      fsamp = fsamp + mu;
      //  4) Now fsamp is our new Gibbs Sampled GP
      
      if (save_samps & (Ntest > 0)) {
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
          double sdstar = pow(kNewNew - (KobsNew.transpose() * T->solve(tau * KobsNew))(0,0), 0.5);
          double mustar = (KobsNew.transpose() * T->solve(tau * Y))(0,0);
          std::normal_distribution<double> distribution(mustar, sdstar);
          fstarsamp(i) = distribution(generator);
        }
      }
      
      delete T;
      delete L;
      
    }

    if ((!regression) & save_samps & (Ntest > 0)) {
      for (int i = 0; i < Ntest; i++) {
        ////// Sample a draw of the GP function f*|f,sig,rho,x*.
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
        double sdstar = pow(kNewNew - (prec_f * KobsNew.transpose() * Svec[rho_ind]->solve(KobsNew))(0,0), 0.5);
        double mustar = (prec_f * KobsNew.transpose() * Svec[rho_ind]->solve(Y))(0,0);
        std::normal_distribution<double> distribution(mustar, sdstar);
        fstarsamp(i) = distribution(generator);
      }
    }
    
    ////// Sample the function variance for the GP sig^(-2)|f,rho,tau. 
    // Equation (6)
    // Svec is K_{σ_f, ρ} in the paper 
    Mat tmp1 =  Svec[rho_ind]->solve(fsamp); // = S^(-1) f, result is a n x 1 matrix
    Mat tmp2 = fsamp.transpose() * tmp1;
    prec_f = rgamma(1, (N + a_f) / 2.0, 2.0 / (tmp2(0, 0) + b_f))[0]; // rgamma(n, shape, scale)
    sig = pow(prec_f, -0.5); // Reparameterize
    sigsq = pow(sig, 2.0);
    
    ////// Sample the length-scale for the GP rho|f,sig,tau.
    if (!fix_ls) {
      if (Gibbs_ls) { // Use Gibbs sampling, loop over all rho choices
        Eigen::VectorXd log_prob(nrhos);
        for (int i = 0; i < nrhos; i++) {
          Mat quad_tmp = prec_f * fsamp.transpose() * Svec[i]->solve(fsamp);
          log_prob(i) = -0.5 * (logdetK_all(i) + quad_tmp(0, 0));
        }
        double maxlg = log_prob.maxCoeff();
        Eigen::VectorXd weights(nrhos);
        // https://stats.stackexchange.com/questions/66616/converting-normalizing-very-small-likelihood-values-to-probability
        for (int i = 0; i < nrhos; i++) {
          weights(i) = exp(log_prob(i) - maxlg); 
        }
        weights = weights / (weights.sum());
        std::discrete_distribution<int> dist(weights.data(), weights.data() + weights.size());
        rho_ind = dist(rng);
      } else{ // Use MH step, making it so you don't have to do tons of inversions!
        Eigen::VectorXd prob_old_to_new = transition_probs_mat.col(rho_ind);
        std::discrete_distribution<int> dist(prob_old_to_new.data(), prob_old_to_new.data() + prob_old_to_new.size());
        int ind_new = dist(rng);
        double lQoldToNew = log(prob_old_to_new(ind_new));
        double lQnewToOld = log(transition_probs_mat(rho_ind,ind_new));
        Mat quad_tmp_old = prec_f * fsamp.transpose() * Svec[rho_ind]->solve(fsamp);
        double llold = -0.5 * (logdetK_all(rho_ind) + quad_tmp_old(0, 0));
        Mat quad_tmp_new = prec_f * fsamp.transpose() * Svec[ind_new]->solve(fsamp);
        double llnew = -0.5 * (logdetK_all(ind_new) + quad_tmp_new(0, 0));
        // Acceptance probability A is min(1, pi(y) Q(x_t|y) / (pi(x_t) Q(y|x_t)) )
        // log(A) = min(0, log(p(y)) + log(Q(x_t|y)) - log(pi(x_t)) -log(Q(y|x_t)) )
        // where y is the "proposed" value for x_{t+1}
        double lA = llnew + lQnewToOld - llold - lQoldToNew;
        if (R::runif(0, 1) < exp(lA)) { // if lA>0, this will always eval
          rho_ind = ind_new;
        }
      }
      rho = rho_choices(rho_ind);
    }
    
    ////// Sample the noise precision for the regression tau|y,sig,rho.
    // Eq (5)
    if (regression) {
      Mat RSS = (Y - fsamp).transpose() * (Y - fsamp); // 1x1 
      tau = rgamma(1, (N + a_tau) / 2.0, 2.0 / (RSS(0, 0) + b_tau))[0];
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

      if (Ntest > 0) {
        fstar_save.col(samp_count) = fstarsamp;
      }
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
  
  py::list out = [];
  out.append(tau_save);
  out.append(sigf_save);
  out.append(rho_save);
  out.append(f_save);
  out.append(fstar_save);
  out.append(duration_all.count()); // Time all, in seconds
  out.append(duration_samp.count()); // Time sampling, in seconds

  return out
}
