#include "sample_funs.h"

// S is an argument that in the code should be Svec[rho_ind]
Eigen::VectorXd samplef_HODLR(Eigen::MatrixXd X, Eigen::MatrixXd Y, double sig, double tau, HODLR_Tree* T, HODLR_Tree* S, std::function<float()> &r_std_normal) {
  /*
    Sample a draw of the GP function f|y,sig,rho,tau.
    Assume a squared exponential Gaussian Process with params sig and rho,
    based on observed data y having precision tau.
  */ 
  int N =  X.rows();
  
  Eigen::VectorXd fsamp(N);
  
  double sigsq = pow(sig, 2.0);
  double sqrt_tau = pow(tau, 0.5);
  
  // Algorithm 1: Sample a, b ~ N(0, I)
  Eigen::MatrixXd A(N, 1);
  Eigen::MatrixXd B(N, 1);
  for (int ii = 0; ii < N; ii++) {
    A(ii, 0) = r_std_normal();
    B(ii, 0) = r_std_normal();
  }   
  
  // Get mean
  Mat mu = T->solve(tau * Y);  
  mu = sigsq * S->matmatProduct(mu);
  
  // We want MVN ~ N(0, Sigma * (Sigma * tau + I)^-1) 
  //  1) Set fsamp = \sqrt(tau) * Sigma * A + W * B, where A, B ~ N (0, I_N).
  //     Equivalent to \sqrt(tau) * \sigma^2 * S * A + \sigma * W * B, 
  //     where S = Sigma/(\sigma^2) (i.e. S is the same kernel but with function var 1).
  fsamp = sqrt_tau * sigsq * S->matmatProduct(A) + sig * S->symmetricFactorProduct(B);
  // Now fsamp ~ (0, Sigma*(tau*Sigma+I)).
  //  2) Multipliy (tau * Sigma + I)^-1 by fsamp. 
  fsamp = T->solve(fsamp); // fsamp ~ (0, (tau * Sigma + I)^-1 * Sigma)
  // We now have fsamp ~ N(0, Sigma * (Sigma * tau + I) ^ -1) 
  // bc both matrices are symmetric and their prod is symmetric so they commute.
  //  3) Add the mean Sigma * (Sigma * tau + I) * (tau * Y); 
  fsamp = fsamp + mu;
  //  4) Now fsamp is our new Gibbs Sampled GP 
  
  return fsamp;
}


double sample_prec_f(HODLR_Tree* S, Eigen::VectorXd fsamp, double a_f, double b_f, std::mt19937 rng) {
  /*

    Sample the function variance for the GP sig^(-2)|f,rho,tau. 

  */

    int N = fsamp.rows();
    
    // Equation (6)
    // Svec is K_{σ_f, ρ} in the paper 
    Mat tmp1 =  S->solve(fsamp); // = S^(-1) f, result is a n x 1 matrix
    Mat tmp2 = fsamp.transpose() * tmp1;

    std::gamma_distribution<float> distribution((N + a_f) / 2.0, 2.0 / (tmp2(0, 0) + b_f));
    double prec_f = distribution(rng);
    return prec_f; 
}

double sample_tau(Eigen::VectorXd Y, Eigen::VectorXd fsamp, double a_tau, double b_tau, std::mt19937 rng) { 
  /*

    Sample the noise precision for the regression tau|y,sig,rho (eq (5) in paper). 

  */

    int N = fsamp.rows();
    Mat RSS = (Y - fsamp).transpose() * (Y - fsamp); // 1 x 1
    std::gamma_distribution<float> distribution((N + a_tau) / 2.0, 2.0 / (RSS(0,0) + b_tau ));
    double tau = distribution(rng);
    return tau;
}

int sample_rho_mh(int rho_ind, Eigen::MatrixXd transition_probs_mat, double prec_f, Eigen::VectorXd fsamp, std::vector<HODLR_Tree*> Svec, Eigen::VectorXd logdetK_all, std::mt19937 rng, std::function<float()> &runif) {

  /*
    Sample the GP length-scale rho|f,sig,tau, using Metropolis Hastings, 
    without having to do tons of inversions.
  */

  Eigen::VectorXd prob_old_to_new = transition_probs_mat.col(rho_ind);
  std::discrete_distribution<int> dist(prob_old_to_new.data(), prob_old_to_new.data() + prob_old_to_new.size());
  int ind_new = dist(rng);
  double lQoldToNew = log(prob_old_to_new(ind_new));
  double lQnewToOld = log(transition_probs_mat(rho_ind, ind_new));
  Mat quad_tmp_old = prec_f * fsamp.transpose() * Svec[rho_ind]->solve(fsamp);
  double llold = -0.5 * (logdetK_all(rho_ind) + quad_tmp_old(0, 0));
  Mat quad_tmp_new = prec_f * fsamp.transpose() * Svec[ind_new]->solve(fsamp);
  double llnew = -0.5 * (logdetK_all(ind_new) + quad_tmp_new(0, 0));
  // Acceptance probability A is min(1, pi(y) Q(x_t|y) / (pi(x_t) Q(y|x_t)) )
  // log(A) = min(0, log(p(y)) + log(Q(x_t|y)) - log(pi(x_t)) -log(Q(y|x_t)) )
  // where y is the "proposed" value for x_{t+1}
  double lA = llnew + lQnewToOld - llold - lQoldToNew;

  if (runif() < exp(lA)) { 
    rho_ind = ind_new;
  }
  return rho_ind;
}

int sample_rho_gibbs(double prec_f, Eigen::VectorXd fsamp, std::vector<HODLR_Tree*>Svec, Eigen::VectorXd logdetK_all, std::mt19937 rng) {
  
  /*
    Sample the length-scale for the GP rho|f,sig,tau, using Gibbs, looping over all rho choices.
  */

  int nrhos = logdetK_all.rows();
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
  int rho_ind = dist(rng);
  return rho_ind;
}