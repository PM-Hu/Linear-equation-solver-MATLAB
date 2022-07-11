/* mexWeightedJacobi using matlab C Matrix api for mexArray
 * Performs Weighted-JacobiC to solve Ax = b
 *
 * usage in matlab:
 * mex mexWeightedJacobiC.cpp  % does not use the openmp here
 * x = mexWeightedJacobiC(transpose(A),b,x); % one iteration
 *
 * message: not the parallel version
 */

#include "mex.h"
#include "matrix.h"
#include <vector>

void WeightedJacobi(
    mwIndex *cols, mwIndex *rows, double *entries, double *rhs, std::vector<double> &x_old, mwSize ncols)
{
    std::vector<double> x_new(ncols, 0.0);
    double weight = 1.0 / 3.0;                // define the weight

    for (ptrdiff_t i = 0; i != ncols; i += 1)
    {  // loop for column
        double D = 1.0;
        double X = rhs[i] * weight;

        for (ptrdiff_t j = cols[i]; j < cols[i + 1]; ++j)  // 
        {  // loop in column
            
            ptrdiff_t rownum = rows[j];
            double v = entries[j];
            X -= weight * v * x_old[rownum]; //

            if (rownum == i)
            {
                D = v;
                X += D * x_old[rownum];
            }
        }
        x_new[i] = X / D;
    }

    #pragma omp parallel for
    for(ptrdiff_t ix=0; ix < ncols; ix++){
        x_old[ix] = x_new[ix];
    }
    
}

/* MEX gateway */

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    double *result;
    double *rhs, *solu;
    mwSize m, n;
    mwIndex *cols;
    mwIndex *rows;
    double *entries;

    m = mxGetM(prhs[0]); // rows
    n = mxGetN(prhs[0]); // columns

    // First output: Solution column vector
    plhs[0] = mxCreateDoubleMatrix(m, 1, mxREAL);
    result  = mxGetPr(plhs[0]);   

    cols    = mxGetJc(prhs[0]);
    rows    = mxGetIr(prhs[0]);
    entries = mxGetPr(prhs[0]);

//  matlab stores sparse matrix in compressed sparse column format.

    solu   = mxGetPr(prhs[1]);
    rhs    = mxGetPr(prhs[2]);
    std::vector<double> x_old(n, 0.0);  //

    #pragma omp parallel for
    for(int ix=0; ix < n; ix++)
        x_old[ix] = solu[ix];
    

    WeightedJacobi(cols, rows, entries, rhs, x_old, n);
    
    #pragma omp parallel for
    for(ptrdiff_t ix=0; ix < m; ix++)
        result[ix] = x_old[ix];
    
    return;
}


