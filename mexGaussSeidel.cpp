/* mexGaussSeidel
 * Performs Gauss-Seidel to solve Ax = b 
 * 
 * usage in matlab:
 * mex mexGaussSeidel.cpp
 * x = mexGaussSeidel(transpose(A),b,x); % one iteration
 * 
 * message: this version is inefficient compares to mldivide, needs further improve
*/

#include "mex.hpp"
#include "mexAdapter.hpp"

class MexFunction : public matlab::mex::Function {
public:
        typedef typename matlab::data::TypedArray<double> Matrix;  // define name
        typedef typename matlab::mex::ArgumentList matlabArgs;      

    void operator()(matlabArgs outputs, matlabArgs inputs) {
        
        matlab::data::SparseArray<double> A = std::move(inputs[0]);   // iterator, transpose
        Matrix rhs = std::move(inputs[1]);
        Matrix x = std::move(inputs[2]);
        int indx = 1;
        // const bool forward = inputs[3][0]; // forward or backward

        serial_sweep_forward(A, rhs, x); // 
        outputs[0] = x;
    }

private:
    void serial_sweep_forward(
                matlab::data::SparseArray<double> &A, const Matrix &rhs, Matrix &x)
        {
            typedef double val_type;
            typedef double rhs_type;

            matlab::data::SparseArray<double>::iterator it = A.begin();
            std::pair<size_t,size_t> kk= A.getIndex(it);
            for(size_t i=0; it != A.end(); i+=1) // only nonzero
            {    

                val_type D = 1.0;
                rhs_type X = rhs[i];

                while(kk.second == i)  // row
                {
                    val_type v = *it;
                    if(kk.first == i)  // column
                        D = v;
                    else
                        X -= v * x[kk.first];

                    ++it;
                    if (it == A.end()) 
                        break;
                    kk = A.getIndex(it);
                }

                x[i] = X / D;

            }
        }
};

