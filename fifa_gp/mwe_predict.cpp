#include <random>
#include <Eigen/Dense>
#include "HODLR_Tree.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <squaredeMat.hpp> //squared exponential kernel
#include <squaredeP1Mat.hpp>

using std::normal_distribution;
namespace py = pybind11;

// AZ: draw random samples from fstar
Eigen::MatrixXd predict(Eigen::MatrixXd X, Eigen::MatrixXd Y, Eigen::MatrixXd Xtest, Eigen::VectorXd sig_samps, Eigen::VectorXd rho_samps, Eigen::VectorXd tau_samps, double multiplier, int M, double tol, int nsamps) {
    /*
    Temporarily assumes regression = True
    */ 

  //////////// PASS THE GENERATOR //////////

    // Create the standard normal generator
    normal_distribution<double> norm(0, 1);
    std::mt19937 rng;
    auto r_std_normal = bind(norm, rng);
    
    ////Number of observations in train and test
    int Ntest =  Xtest.rows();  
    int N =  X.rows();
    int D =  X.cols();
   

    int n_levels = log(N / M) / log(2);
    bool is_sym = true;
    bool is_pd  = true;

    double tau;
    double rho; 
    double sig; 
    
    // Sigma square and SSR
    double sigsq; // = pow(sig, 2.0);
    double tmpSSR;
    
    // Allocate fstarsamp
    Eigen::MatrixXd fstarsamp(nsamps, Ntest);
    Eigen::VectorXd KobsNew(N);
    
    for (int s = 0; s < nsamps; s++) {

      sig = sig_samps(s);
      tau = tau_samps(s);
      rho = rho_samps(s);

      if (s % 100 == 0) {
        // printf("PREDICT: Sig, Tau, Rho: s = %d, %.2f \t%.2f \t%.2f\n", s, sig, tau, rho);
      }

      sigsq = pow(sig, 2.0);

      SQRExponentialP1_Kernel* L  = new SQRExponentialP1_Kernel(X, N, sig, rho, tau);
      HODLR_Tree* T = new HODLR_Tree(n_levels, tol, L); // With noise (i.e. Sigma + I/tau)
      T->assembleTree(is_sym, is_pd);
      T->factorize();

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

        fstarsamp(s, i) =  sdstar * normal_samp + mustar;
        //std::normal_distribution<double> distribution(mustar, sdstar);
        //fstarsamp(i) = distribution(generator);
      }
    }    
    return fstarsamp;
}

void predict_module(py::module &m) {
    m.def("predict_f", &predict, "predicted samples of f at new X");
}