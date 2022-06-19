/* mexGaussSeidel using matlab C Matrix api for mexArray
 * Performs Gauss-Seidel to solve Ax = b
 *
 * usage in matlab:
 * mex mexGaussSeidelC.cpp
 * x = mexGaussSeidelC(transpose(A),b,x); % one iteration
 *
 * message: this version(0.029s) is faster than mexGaussSeidel(6.5s) and mldivid(0.05s)
 * 
 * 22-6-19 update: create a std::vector<double> container to sotre the input and output solution,
 * fixed an unexplainable error in matlab that leads pcg to fail to converge. (the solution x in
 * matlab programm cannot be assigned)
 */

#include "mex.h"
#include "matrix.h"
#include <vector>

void serial_sweep_forward(
    mwIndex *cols, mwIndex *rows, double *entries, double *rhs, std::vector<double> &x, mwSize ncols)
{

    for (ptrdiff_t i = 0; i != ncols; i += 1)
    {  // loop for column
        double D = 1.0;
        double X = rhs[i];

        for (ptrdiff_t j = cols[i]; j < cols[i + 1]; ++j)  // 
        {  // loop in column
            
            ptrdiff_t rownum = rows[j];
            double v = entries[j];
            if (rownum == i)
                D = v;
            else
                X -= v * x[rownum]; // x - solution
        }
        x[i] = X / D;
    }
    
}

/* MEX gateway */

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{

    double *result, *rhs, *solu;
    mwSize m, n;
    mwIndex *cols, *rows;
    double *entries;

    if (nrhs == 0 && nlhs == 0)
    {
        mexPrintf("Gauss Seidel is compiled and ready for use.\n");
        return;
    }

    m = mxGetM(prhs[0]); // rows
    n = mxGetN(prhs[0]); // columns

    // First output: Solution column vector
    plhs[0] = mxCreateDoubleMatrix(m, 1, mxREAL);
    result  = mxGetPr(plhs[0]);   

    cols    = mxGetJc(prhs[0]);
    rows    = mxGetIr(prhs[0]);
    entries = mxGetPr(prhs[0]);

    /*
     * CSR - compressed sparse row  or CRS - compressed row storage
     * A   = [1 0 0;
     *        1 0 0;
     *        0 1 0];
     *
     * val = [1 1 1];
     * col = [0 0 1];
     * ptr = [0 1 2 3];
     * ---
     * But matlab stores sparse matrix in compressed sparse column format.
     */

    rhs    = mxGetPr(prhs[1]);
    solu   = mxGetPr(prhs[2]);
    std::vector<double> solx(n, 0.0);  // a container

    #pragma omp parallel for
    for(int ix=0; ix < n; ix++){
        solx[ix] = solu[ix];
    }

    serial_sweep_forward(cols, rows, entries, rhs, solx, n);

    #pragma omp parallel for
    for(ptrdiff_t ix=0; ix < m; ix++){
        result[ix] = solx[ix];
    }
    return;
}


