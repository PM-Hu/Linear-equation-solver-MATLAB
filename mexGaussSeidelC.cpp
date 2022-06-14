/* mexGaussSeidel using matlab C Matrix api for mexArray
 * Performs Gauss-Seidel to solve Ax = b
 *
 * usage in matlab:
 * mex mexGaussSeidelC.cpp
 * x = mexGaussSeidelC(transpose(A),b,x); % one iteration
 *
 * message: this version(0.029s) is faster than mexGaussSeidel(6.5s) and mldivid(0.05s)
 */

#include "mex.h"
#include "matrix.h"

void serial_sweep_forward(
    mwIndex *cols, mwIndex *rows, double *entries, double *rhs, double *x, mwSize ncols)
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

    // We use the tuple of CRS arrays to represent the system matrix.
    // Note that std::tie creates a tuple of references, so no data is actually
    // copied here:
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

    serial_sweep_forward(cols, rows, entries, rhs, solu, n);

    for(ptrdiff_t ix=0; ix < m; ix++){
        result[ix] = solu[ix];
    }
    return;
}


