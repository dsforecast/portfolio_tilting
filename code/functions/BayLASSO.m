function results=BayLASSO(Y,X,B)

% Purpose: Bayesian Lasso Regression of Park and Gasella (2008)
%          via Gibbs Sampler
% Input:  Y       = (T x 1) column vector of left-handside centered variable
%         X       = (T x N) matrix of right-handside centered variables
%         B       = Scalar, number of Gibbs sampler iterations
% betaOLS = (N x 1) vector of OLS estimates
% sigmaOLS = scalar, OLS variance
% Output: beta    = (B x N) posterior draws of regression coefficients
%         sigma   = (B x 1) posterior draws of error variance
%         invtau2 = (B x N) posterior draws of 1/t_j^2
%         lambda  = (B x 1) posterior draws of lambda
% Reference: Park, T. and Casella, G. (2008): "The Bayesian Lasso,"
%            Journal of American Statistical Association,
%            103(482), 681--686.

%% Set-Up

% No warning messages
warning off; %#ok<WNOFF>

% Test setting
if nargin==0
    t=60;
    M=1;
    data=randn(t,M);
    Y=data(:,1);
    X=ones(size(Y));%data(:,2:end);
    B=100;
    p=1;
    prior=1;
    repfor = 1;  % Number of times to obtain a draw from the predictive density, for each generated draw of the parameters                     
    h=1;
    vprior=2;
end

% Dimensions
[T,N]=size(X);

% MCMC Set-Up
burnin=ceil(0.3*B);
ndraw=B+burnin;
beta=zeros(B,N);
sigma=zeros(B,1);
invtau2=zeros(B,N);
lambda=zeros(B,1);

 % OLS quantities
N=size(X,2);
betaOLS=pinv(X'*X)*X'*Y;
SSE=(Y-X*betaOLS)'*(Y-X*betaOLS);
sigmaOLS=SSE/(T-N*(N<T));

% Parameter values for first Gibbs sammple step
sigmaD=sigmaOLS*(T>N)+0;
invtau2D=(1./(betaOLS.^2))'*(T>N)+1*(T<=N);
lambdaD=0.1;

% Further hyperparameters
r=1;
delta=1;


%% MCMC
for i=1:ndraw
    
    % (1) Sample beta | tau, sigma
    invD=diag(invtau2D);
    % Rubustness Check
    try
        invA=pinv(X'*X+invD);
        invA=tidy_cov_mat(invA);
        betaD=mvnrnd(invA*X'*Y,sigmaD*invA,1);
        kickout=0;
    catch
        betaD=mvnrnd(invA*X'*Y,eye(N));
        kickout=1;
    end    
    
    % (2) Sample sigma | beta, tau
    s=(Y-X*betaD')'*(Y-X*betaD')./2+betaD*invD*betaD'./2;
    sigmaD=1/(gamrnd((T+N-1)./2,1./s));
    
    % (3) Sample invtau2 | sigma, beta
    for j=1:N
        mu=sqrt(lambdaD.*sigmaD./betaD(:,j).^2);
        invtau2D(1,j)=Draw_InvGaus(mu,lambdaD);
    end
    
    % (4) Sample lambda form Gamma | tau
    s2=sum(1./invtau2D)./2+delta;
    lambdaD=gamrnd(N+r,1./s2);
    
    % Save everything
    if i>burnin && kickout==0
        beta(i-burnin,:)=betaD;
        sigma(i-burnin,1)=sigmaD;
        invtau2(i-burnin,:)=invtau2D;
        lambda(i-burnin,1)=lambdaD;
    end
end

%% Delete knockout draws
beta=beta(beta(:,1)~=0,:);
sigma=sigma(sigma(:,1)~=0,1);
invtau2=invtau2(invtau2(:,1)~=0,:);
lambda=lambda(lambda(:,1)~=0,:);

%% Predictions
predictions=X(end,:)*beta';

%% Save results
results.beta=beta;
results.sigma=sigma;
results.predictions=predictions';

%% Thinning to avoid highly correlated draws
% thinning=1;%B/5000;
% thinningindex=1:thinning:length(sigma);
% beta=beta(thinningindex,:);
% sigma=sigma(thinningindex,1);
% invtau2=invtau2(thinningindex,:);
% lambda=lambda(thinningindex,1);


end
