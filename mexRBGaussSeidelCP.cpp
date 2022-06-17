/* mexRBGaussSeidelCP using matlab C Matrix api for mexArray
 * Performs Gauss-Seidel to solve Ax = b
 *
 * usage in matlab:
 * mex mexRBGaussSeidelCP.cpp
 * x = mexRBGaussSeidelCP(transpose(A),b,x); % one iteration
 *
 * message: enjoy! (P refer to parallel)
 */

#include "mex.h"
#include "matrix.h"
#include <vector>

void serial_sweep(
    mwIndex *cols, mwIndex *rows, double *entries, double *rhs, double *x, mwSize ncols)
{
    std::vector<double> x_old(ncols, 0.0); //

    #pragma omp parallel for
    for (ptrdiff_t ix = 0; ix < ncols; ix++)
    {
        x_old[ix] = x[ix];
    }

    #pragma omp parallel for
    for (ptrdiff_t i = 0; i != ncols; i += 1)
    { // loop for column
        double D = 1.0;
        double X = rhs[i];

        if (i % 2 == 0) // even components
        {
            for (ptrdiff_t j = cols[i]; j < cols[i + 1]; ++j) //
            {                                                 // loop in column

                ptrdiff_t rownum = rows[j];
                double v = entries[j];
                if (rownum == i)
                    D = v;
                else
                    X -= v * x[rownum]; // x - solution
            }
            x_old[i] = X / D;
        }
    }

    #pragma omp parallel for
    for (ptrdiff_t i = 0; i != ncols; i += 1)
    { // loop for column
        double D = 1.0;
        double X = rhs[i];

        if (i % 2 == 1) // odd components
        {
            for (ptrdiff_t j = cols[i]; j < cols[i + 1]; ++j) //
            {                                                 // loop in column

                ptrdiff_t rownum = rows[j];
                double v = entries[j];
                if (rownum == i)
                    D = v;
                else
                    X -= v * x_old[rownum]; // x - solution
            }
            x[i] = X / D;
        }
        else
            x[i] = x_old[i]; // assign the updated solution on even components to x
    }

    x_old.clear();
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
    result = mxGetPr(plhs[0]);

    cols = mxGetJc(prhs[0]);
    rows = mxGetIr(prhs[0]);
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
     * val = [1 1 1];   //
     * col = [0 0 1];   //
     * ptr = [0 1 2 3]; //
     * ---
     * But matlab stores sparse matrix in compressed sparse column format.
     */

    solu = mxGetPr(prhs[1]);
    rhs = mxGetPr(prhs[2]);

    serial_sweep(cols, rows, entries, rhs, solu, n);

    #pragma omp parallel for
    for (ptrdiff_t ix = 0; ix < m; ix++)
    {
        result[ix] = solu[ix];
    }
    return;
}
