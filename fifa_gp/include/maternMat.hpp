#ifndef __RHODLR_MATERN_MAT__
#define __RHODLR_MATERN_MAT__

#include "HODLR_Matrix.hpp"
#include "HODLR_Tree.hpp"

// Taking Matern Kernel for p = 2:
// K(r) = σ^2 * (1 + sqrt(5) * r / ρ + 5/3 * (r / ρ)^2) * exp(-sqrt(5) * r / ρ)
class Matern_Kernel : public HODLR_Matrix 
{

private:
    Mat x;
    double sigma_squared, rho;

public:

    // Constructor:
    Matern_Kernel(Mat tX, int N, double sigma, double rho) : HODLR_Matrix(N), x(tX) 
    {
        
        this->sigma_squared = sigma * sigma;
        this->rho           = rho;
        // This is being sorted to ensure that we get
        // optimal low rank structure:
        std::sort(x.data(), x.data() + x.size());
    };
    
    dtype getMatrixEntry(int i, int j) 
    {
        double R_by_rho = fabs(x(i) - x(j)) / rho;  
        double nugget = (i ==j)?1e-8:0.0;       
        return sigma_squared * (1 + sqrt(5) * R_by_rho + 5/3 * (R_by_rho * R_by_rho)) * exp(-sqrt(5) * R_by_rho)+nugget;
    }

    // Destructor:
    ~Matern_Kernel() {};
};

#endif
