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

function x = Gauss_seidel(A,x,b,tol,maxit,row_start,row_stop,row_step)
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
% ====  Example: Run the Code or Press F5  ==============================
        A = [10 0 1 -5;1 8 -3 0;3 2 8 1;1 -2 2 7];
        x = [1;1;1;1];     % initial random choice
        b = [-7;11;23;17];
      tol = 1d-3;
    maxit = 1000;          % max iteration number
row_start = 1;
row_stop  = 4;
row_step  = 1;             % forward 
% 
% show: it  = 8
%       x   = [ 0.3156;  2.0625;  1.9386;  2.4189]
%       A*x = [-6.9993; 10.9998; 22.9999; 17.0000] \approx b
% ========================================================================= 

I = diag(A);
A = A - diag(diag(A));   % remove diag A
err = 1; it = 1;         % start 
while (err >= tol) ~= 0 && it <= maxit
    x_old = x;           % store x_old
    for ii = row_start:row_step:row_stop
        B = b(ii) - A(ii,:) * x;  
        x(ii) = B / I(ii);
    end
    err = sum(abs(x - x_old));
    it = it+1;
end

end

