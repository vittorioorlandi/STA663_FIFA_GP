#ifndef __RHODLR_SQREXP_MAT__
#define __RHODLR_SQREXP_MAT__

#include "HODLR_Matrix.hpp"
#include "HODLR_Tree.hpp"

// Taking Squared Exponential Kernel 
// K(r) = σ^2 * exp(-rho*||r||^2)
class SQRExponential_Kernel : public HODLR_Matrix 
{

private:
    Mat x;
    double sigma_squared, rho;
    int D;
    
public:

    // Constructor:
    SQRExponential_Kernel(Mat tX, int N, double sigma, double rho) : HODLR_Matrix(N), x(tX) 
    {        
        this->sigma_squared = sigma * sigma;
        this->rho           = rho;
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
        double nugget = (i ==j)?1e-8:0.0;    /*we have Sigma*tau + 1*/    
        return (sigma_squared * exp(-R_by_rho)+nugget);
    }

    // Destructor:
    ~SQRExponential_Kernel() {};
};

#endif
