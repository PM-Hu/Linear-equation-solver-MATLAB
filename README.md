# Linear-equation-solver-MATLAB
Some algorithms to solve Linear equations in MATLAB, such as Jacobi, Guass-Seidel, and so on.

the [**mex**] function files are added: GuassSeidel.cpp,  GuassSeidelC.cpp

- **mexGuassSeidel.cpp** uses the MATLAB® Data API to read and write MATLAB data from C++ programs, but inefficient.
- **mexGuassSeidelC.cpp** uses `mxArray` to read and write MATLAB® data from C programs, highly efficient.

- **mexRBGaussSeidelCP.cpp**, the red-black Gauss-Seidel, suitable for **parallel computation**.

updating...
