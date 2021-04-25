#include <Eigen/Dense>

Eigen::VectorXd  get_transition_probs_byind(Eigen::VectorXi rhovec_inds, double krnl, int ind_current){
  int nrhos = rhovec_inds.rows();
  Eigen::VectorXd  prop_prob(nrhos);
  for(int i=0; i<nrhos; i++){ prop_prob(i) = exp(-krnl*pow((double)rhovec_inds(i)-(double)ind_current,2.0)); }
  prop_prob(ind_current) = 0.0;
  double sumInv = 1.0 / prop_prob.sum();
  prop_prob = sumInv * prop_prob;
  return prop_prob;
}
