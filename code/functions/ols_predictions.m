function [estimates,residuals,SE_robust]=ols_predictions(Y,X)
    
% Purpose: estimate linear regression via OLS and make predictions
% Inputs:
% Y = (N x 1) vector of dependent variable
% X = (N x K) matrix of explanatory variables 

% Test setting
if nargin==0
    N=100;
    K=1;
    Y=randn(N,1);
    X=randn(N,K);
end

% Dimensions

% Add constant
if isempty(X)==1
    N=size(Y,1);
    X=ones(N,1);
    K=0;
else
    [N,K]=size(X);
    X=[ones(N,1),X];
end

% Robustness check
if size(X,1)~=size(Y,1)
    disp('Error: Y and X have to have the same number of observations.')
end

% OLS estimation
estimates=(X'*X)\(X'*Y);
residuals=Y-X*estimates;
inv_XX=inv((X'*X));
X_trans=X.*residuals(:,ones(1,K+1));
cov_mat_robust=N/(N-K-1).*inv_XX*(X_trans'*X_trans)*inv_XX; %#ok<*MINV>
SE_robust=sqrt(diag(cov_mat_robust));

end

