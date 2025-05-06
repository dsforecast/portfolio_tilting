function results=BVAR(data,p,prior,vprior,h,repfor,N_save)

% Purpose: Estimate and predict VAR(p) model through a Gibbs sampler
% Inputs:
% data = (T x K) matrix of dependent variables
% p = scalar, number of lags in the VAR system
% prior = 1 for Indepependent Normal-Whishart Prior, 
%         = 2 for Indepependent Minnesota-Whishart Prior
% vprior = 1 for uninformative
%            = 0 for full model first equation and AR(1) in the following
% h = Number of forecast periods
% N_save = number of draws to save
% repfor = Number of times to obtain a draw from the predictive density, for each generated draw of the parameters                     
% Ouputs:
% to be defined

%% Introduction

% Test setting
if nargin==0
    Traw=60;
    M=13;
    data=randn(Traw,M);
    p=0;
    prior=1;
    repfor = 1;  % Number of times to obtain a draw from the predictive density, for each generated draw of the parameters                     
    h=1;
    vprior=1;
    N_save=100;                      % Final number of draws to save
end

% Dimensions
Yraw=data;
[Traw,M]=size(Yraw);

% Number of Gibbs draws
N_burn=0.1*N_save;        % Draws to discard (burn-in)
N=N_save+N_burn;             % Total number of draws

% Generate lagged Y matrix. This will be part of the X matrix
Ylag=mlag2(Yraw,p);

% Now define matrix X which has all the R.H.S. variables (constant, lags, exogenous regressors)
if p==0
    X=ones(Traw-p,1);
else
    X=[ones(Traw-p,1) Ylag(p+1:Traw,:)];
end

% Get size of final matrix X
K=size(X,2);

% Create the block diagonal matrix Z
Z=kron(eye(M),X);

% Form Y matrix accordingly Delete first "LAGS" rows to match the dimensions of X matrix
Y=Yraw(p+1:Traw,:); 

% T is the number of actual time series observations of Y and X (we lose the p-lags)
T=Traw-p;

% Containers for forecasting
Y_pred=zeros(N*repfor,M);    % Matrix to save prediction draws
PL=zeros(N,1);                              % Matrix to save Predictive Likelihood

% First get ML estimators
A_OLS=(X'*X)\(X'*Y); % This is the matrix of regression coefficients
a_OLS=A_OLS(:);         % This is the vector of parameters, i.e. it holds that a_OLS = vec(A_OLS)
SSE=(Y-X*A_OLS)'*(Y-X*A_OLS);   % Sum of squared errors
SIGMA_OLS=SSE./(T-K+1);

% Initialize Bayesian posterior parameters using OLS values
alpha=a_OLS;     
ALPHA=A_OLS;     
SSE_Gibbs=SSE;   
SIGMA=SIGMA_OLS;

% Storage space for posterior draws
alpha_draws=zeros(N_save,K*M);
ALPHA_draws=zeros(N_save,K,M);
SIGMA_draws=zeros(N_save,M,M);
Y_pred=zeros(N_save,M);

%% Prior set-up

L=K*M; % Total number of parameters (size of vector alpha)
% Define hyperparameters
if prior==1 % Normal-Wishart
    a_prior=0*ones(L,1);   
    if vprior==1
    V_prior=10*eye(L);
    elseif vprior==2
        V_prior=[];
        for iii=1:K-1
            if iii==1
                tmp=10*eye(K);
                tmp(2,2)=eps;
            else
                tmp=eps*eye(K);
                tmp(1,1)=10;
                tmp(iii+1,iii+1)=10;
            end
            V_prior=blkdiag(V_prior,tmp);
        end
    end
    v_prior=M;            
    S_prior=eye(M);         
    
elseif prior == 2 % Minnesota-Whishart
    % Prior mean
    A_prior=[zeros(1,M); 0.9*eye(M); zeros((p-1)*M,M)];
    a_prior=A_prior(:);
    
    % Minnesota Variance on VAR regression coefficients
    % First define the hyperparameters 'a_bar_i'
    a_bar_1 = 0.5;
    a_bar_2 = 0.5;
    a_bar_3 = 10^2;
    
    % Now get residual variances of univariate p-lag autoregressions. Here
    % we just run the AR(p) model on each equation, ignoring the constant
    % and exogenous variables (if they have been specified for the original
    % VAR model)
    sigma_sq=zeros(M,1); % vector to store residual variances
    for i = 1:M
        % Create lags of dependent variable in i-th equation
        Ylag_i = mlag2(Yraw(:,i),p);       
        Ylag_i=Ylag_i(p+1:Traw,:);
        % Dependent variable in i-th equation
        Y_i=Yraw(p+1:Traw,i);
        % OLS estimates of i-th equation
        alpha_i=(Ylag_i'*Ylag_i)\(Ylag_i'*Y_i);
        sigma_sq(i,1)=(1./(Traw-p+1))*(Y_i - Ylag_i*alpha_i)'*(Y_i - Ylag_i*alpha_i);
    end
    
    % Now define prior hyperparameters.
    % Create an array of dimensions K x M, which will contain the K diagonal
    % elements of the covariance matrix, in each of the M equations.
    V_i=zeros(K,M);
    
    % index in each equation which are the own lags
    ind=zeros(M,p);
    for i=1:M
        ind(i,:)=1+i:M:K;
    end
    for i=1:M  % for each i-th equation
        for j=1:K   % for each j-th RHS variable
                if j==1
                    V_i(j,i) = a_bar_3*sigma_sq(i,1); % variance on constant                
                elseif find(j==ind(i,:))>0
                    V_i(j,i)=a_bar_1./(p^2); % variance on own lags           
                else
                    for kj=1:M
                        if find(j==ind(kj,:))>0
                            ll=kj;                   
                        end
                    end                 % variance on other lags   
                    V_i(j,i)=(a_bar_2*sigma_sq(i,1))./((p^2)*sigma_sq(ll,1));      
                end
        end
    end
    
    % Now V is a diagonal matrix with diagonal elements the V_i
    V_prior=diag(V_i(:));  % this is the prior variance of the vector alpha
    
    % Hyperparameters on inv(SIGMA) ~ W(v_prior,inv(S_prior))
    v_prior=M;
    S_prior=eye(M);
    inv_S_prior=pinv(S_prior);   
end


%% MCMC Sampling
for irep=1:N  %Start the Gibbs "loop"
  
    VARIANCE=kron(inv(SIGMA),speye(T));
    V_post=(V_prior\eye(L)+Z'*VARIANCE*Z)\eye(L);
    V_post=tidy_cov_mat(V_post)+0.0001*eye(size(V_post));
    a_post=V_post*(V_prior\a_prior+Z'*VARIANCE*Y(:));
    alpha=a_post+chol(V_post)'*randn(L,1);    
    ALPHA=reshape(alpha,K,M);
    
    % Posterior of SIGMA|ALPHA,Data ~ iW(inv(S_post),v_post)
    v_post=T+v_prior;
    S_post=S_prior+(Y - X*ALPHA)'*(Y - X*ALPHA);
    S_post=tidy_cov_mat(S_post)+eps*eye(size(S_post));
    try
        SIGMA=iwishrnd(S_post,v_post);% Draw SIGMA
    catch
        SIGMA=iwishrnd(eye(size(S_post)),v_post);% Draw SIGMA
    end
    % Store results  
    if irep>N_burn     
        
        %% Forecasting
        % The usual way is to write the VAR(p) model in companion form, i.e. as VAR(1) model in order to estimate the
        % h-step ahead forecasts directly (this is similar to the code we use below to obtain impulse responses). Here we 
        % just iterate over h periods, obtaining the forecasts at T+1, T+2, ..., T+h iteratively.
                
        for ii = 1:repfor
                % Forecast of T+1 conditional on data at time T
                if p==0
                    X_fore=ones(M,1)';
                    Y_hat=X_fore.*ALPHA + randn(1,M)*chol(SIGMA);
                else
                    X_fore=[1 Y(T,:) X(T,2:M*(p-1)+1)];
                    Y_hat=X_fore*ALPHA + randn(1,M)*chol(SIGMA);
                end
                %Y_hat=X_fore*ALPHA + randn(1,M)*chol(SIGMA);
                Y_temp=Y_hat;
                X_new_temp=X_fore;
                for i = 1:h-1  % Predict T+2, T+3 until T+h                   
                    if i<=p && p>0
                        % Create matrix of dependent variables for predictions. Dependent on the horizon, use the previous                       
                        % forecasts to create the new right-hand side variables which is used to evaluate the next forecast.                       
                        X_new_temp=[1 Y_hat X_fore(:,2:M*(p-i)+1)];
                        % This gives the forecast T+i for i=1,..,p                       
                        Y_temp=X_new_temp*ALPHA+randn(1,M)*chol(SIGMA);                       
                        Y_hat=[Y_hat Y_temp];
                    elseif i>p && p>0
                        X_new_temp=[1 Y_hat(:,1:M*p)];
                        Y_temp=X_new_temp*ALPHA+randn(1,M)*chol(SIGMA);
                        Y_hat=[Y_hat Y_temp];
                    else
                        X_new_temp=ones(M,1)';
                        Y_temp=X_fore.*ALPHA + randn(1,M)*chol(SIGMA);
                        Y_hat=[Y_hat Y_temp];
                    end
                end %  the last value of 'Y_temp' is the prediction T+h
                Y_temp2(ii,:)=Y_temp;
            end
            % Matrix of predictions               
            Y_pred(((irep-N_burn)-1)*repfor+1:(irep-N_burn)*repfor,:) = Y_temp2;
            % Predictive likelihood
            SIGMA=tidy_cov_mat(SIGMA);
%             T+h
%             Yraw(T+h,:)
%             PL(irep-N_burn,:) = mvnpdf(Yraw(T+h,:),X_new_temp*ALPHA,SIGMA);
%             if PL(irep-N_burn,:) == 0
%                 PL(irep-N_burn,:)=1;
%             end
     
        %----- Save draws of the parameters
        alpha_draws(irep-N_burn,:)=alpha;
        ALPHA_draws(irep-N_burn,:,:)=ALPHA;
        SIGMA_draws(irep-N_burn,:,:)=SIGMA;
    end
end

%% Save all results
results.alpha=alpha_draws;
results.ALPHA=ALPHA_draws;
results.SIGMA=SIGMA_draws;
results.predictions=Y_pred;




end

