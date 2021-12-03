%% ****************************************************************
% % Matlab implementation of Gauss seidel to solve Ax = b iteratively 
% % >>> Gauss seidel idea is derived here:
% %            A*x = b -> (L + D + U)*x = b
% % where L is the lower triangular elements, U is the upper triangular
% % elements, D is the diagonal elements.
% % >>> then:
% %            (L + D)*x_new = b - U*x_old
% % >>> obviously (Forward):
% %            x_new = (L + D)^{-1} * [b - U*x_old]
% % >>> similarly (backward):
% %            x_new = (D + U)^{-1} * [b - L*x_old]
% % ********************************************************************

function x = Gauss_seidel_large_scale_sparsity(A,x,b,tol,maxit,row_start,row_stop,row_step)
% NEED:   N*N:   A         - (matrix, diag dominant)
%         N*1:   x         - (initial choice, could be the start of GS)
%         N*1:   b         - (right hand)
%         float: tol       - (tolerence)
%         int:   maxit     - (max iteration number)
%         int:   row_start - (beginning of the sweep)
%         int:   row_stop  - (end of the sweep)
%         int:   row_step  - (stride used during the sweep, may be negative)
% 
%  OUT:   N*1:   x_new (approximate solution, sweep x after maxit or satisfy tol)
%             
% ====  Example: Run the Code or Press F5  ================================
     dofs = 100000;
        e = ones(dofs,1);
        A = spdiags([-1*e 2*e -1*e],-1:1,dofs,dofs);
        b = zeros(dofs,1);  b([1,end])= 1;
        x = zeros(dofs,1);
      tol = 1d-3;
    maxit = 1;          % max iteration number
row_start = 1;
row_stop  = dofs;
row_step  = 1;          % forward 
% 
% show: about 0.34s per circle, faster than the algorithm in Gauss_seidel.m
% ========================================================================= 

I = diag(A);
A = A - diag(diag(A));   % remove diag A
spy(A)                   % show sparsity

[indices,~, ~] = find(A');         % Aj - nonzero column index 
nzlg = (A ~= 0);
rowsum = full([0; sum(nzlg,2)]);   % number of nonzeros each row
indptr = cumsum(rowsum,1)+1;       % index pointer array

err = 1; it = 1;         % start 
while (err >= tol) ~= 0 && it <= maxit
    x_old = x;           % store x_old
    tic
    for ii = row_start:row_step:row_stop
        pt_start = indptr(ii);            % nonzeros column pointer - start
        pt_stop  = indptr(ii+1)-1;        % nonzeros column pointer - stop
        cols = indices(pt_start:pt_stop); % nonzeros column index
        B = b(ii) - A(ii,cols) * x(cols);
        x(ii) = B / I(ii);
    end
    toc
    err = sum(abs(x - x_old));
    it = it+1;
end

end

