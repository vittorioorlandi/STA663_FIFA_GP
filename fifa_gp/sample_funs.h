#include <random>
#include <Eigen/Dense>
#include "HODLR_Tree.hpp"

Eigen::VectorXd samplef_HODLR(Eigen::MatrixXd X, Eigen::MatrixXd Y, double sig, double tau, HODLR_Tree* T, HODLR_Tree* S, std::function<float()> &r_std_normal);

double sample_prec_f(HODLR_Tree* S, Eigen::VectorXd fsamp, double a_f, double b_f, std::mt19937 rng);

double sample_tau(Eigen::VectorXd Y, Eigen::VectorXd fsamp, double a_tau, double b_tau, std::mt19937 rng);

int sample_rho_mh(int rho_ind, Eigen::MatrixXd transition_probs_mat, double prec_f, Eigen::VectorXd fsamp, std::vector<HODLR_Tree*> Svec, Eigen::VectorXd logdetK_all, std::mt19937 rng, std::function<float()> &runif);

int sample_rho_gibbs(double prec_f, Eigen::VectorXd fsamp, std::vector<HODLR_Tree*>Svec, Eigen::VectorXd logdetK_all, std::mt19937 rng);