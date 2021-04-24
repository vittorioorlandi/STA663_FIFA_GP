#ifndef __RHODLR_SQREXP_P1_MAT__
#define __RHODLR_SQREXP_P1_MAT__

#include "HODLR_Matrix.hpp"
#include "HODLR_Tree.hpp"

// Taking tau times Squared Exponential Kernel ( K(r) = σ^2 * exp(-rho*||r||^2) ) 
// realization plus the identity, i.e. tau K + I; noise precision tau is homoskedastic.
class SQRExponentialP1_Kernel : public HODLR_Matrix 
{

private:
    Mat x;
    double sigma_squared, rho;
    double tau;
    int D;
public:

    // Constructor:
    SQRExponentialP1_Kernel(Mat tX, int N, double sigma, 
                                  double rho, double ttau) : HODLR_Matrix(N), x(tX) 
    {        
        this->sigma_squared = sigma * sigma;
        this->rho           = rho;
        this->tau           = ttau;    
        this->D             = tX.cols();
     };
    
    dtype getMatrixEntry(int i, int j) 
    {
        double temp = 0;
        for(int d=0; d<D; d++){ 
          double temp2 = x(i, d) - x(j, d);
          temp = temp + temp2*temp2;
        }
        double R_by_rho = temp * rho; 
        double nugget = (i ==j)?1e-8:0.0;    /*we have Sigma*tau + I*/
        double I = (i == j)?1:0.0;    // this is the identity portion    
        return tau*sigma_squared*exp(-R_by_rho)+nugget+I;
    }

    // Destructor:
    ~SQRExponentialP1_Kernel() {};
};

#endif
