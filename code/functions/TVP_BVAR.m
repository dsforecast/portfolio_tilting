function results=TVP_BVAR(ydata,p,vprior,h,repfor,nrep)

% Purpose: Estimate and predict TVP-VAR(p) model through a Gibbs sampler
% This code implements the Homoskedastic TVP-VAR using the Carter and Kohn (1994)
% algorithm for state-space models.
% ************************************************************************************
% The model is:
%
%     Y(t) = B0(t) + B1(t)xY(t-1) + B2(t)xY(t-2) + u(t) 
% 
%  with u(t)~N(0,H).
% The state equation is
%
%            B(t) = B(t-1) + error
%
% where B(t) = [B0(t),B1(t),B2(t)]'.
%
% ************************************************************************************
%   NOTE: 
%      There are references to equations of Primiceri, "Time Varying Structural Vector
%      Autoregressions & Monetary Policy",(2005),Review of Economic Studies 72,821-852
%      for your convenience. The definition of vectors/matrices is also based on this
%      paper.
% ------------------------------------------------------------------------------------
% Inputs:
% data = (T x K) matrix of dependent variables
% p = scalar, number of lags in the VAR system
% prior = 1 for Indepependent Normal-Whishart Prior, 
%         = 2 for Indepependent Minnesota-Whishart Prior
% vprior = 1 for uninformative
%            = 0 for full model first equation and AR(1) in the following
% h = Number of forecast periods
% nrep = Number of replications
% repfor = Number of times to obtain a draw from the predictive density, for each generated draw of the parameters                     

% Ouputs:
% to be defined

%% Introduction

%warning off

% Test setting
if nargin==0
    t=60;
    M=12;
    ydata=randn(t,M);
    p=1; % p is number of lags in the VAR part
    repfor=1;  % Number of times to obtain a draw from the predictive density, for each generated draw of the parameters                     
    h=1;
    vprior=2;
    nrep = 1000;  % Number of replications
end

% % Demean and standardize data
%t2 = size(ydata,1);
%ydata = (ydata- repmat(mean(ydata,1),t2,1))./repmat(std(ydata,1),t2,1);
Y=ydata;

% Number of observations and dimension of X and Y
t=size(Y,1); % t is the time-series observations of Y
M=size(Y,2); % M is the dimensionality of Y

% Number of factors & lags:
tau=t/2; % tau is the size of the training sample

% ===================================| VAR EQUATION |==============================
% Generate lagged Y matrix. This will be part of the X matrix
ylag = mlag2(Y,p); % Y is [T x M]. ylag is [T x (Mp)]
% Form RHS matrix X_t = [1 y_t-1 y_t-2 ... y_t-k] for t=1:T
ylag = ylag(p+tau+1:t,:);

K = M + p*(M^2); % K is the number of elements in the state vector
% Create Z_t matrix.
Z = zeros((t-tau-p)*M,K);
for i = 1:t-tau-p
    ztemp = eye(M);
    for j = 1:p        
        xtemp = ylag(i,(j-1)*M+1:j*M);
        xtemp = kron(eye(M),xtemp);
        ztemp = [ztemp xtemp];  %#ok<AGROW>
    end
    Z((i-1)*M+1:i*M,:) = ztemp;
end

% Redefine FAVAR variables y
y = Y(tau+p+1:t,:)';
% Time series observations
t=size(y,2);   % t is now 215 - p - tau = 173

% Now define matrix X which has all the R.H.S. variables (constant, lags, exogenous regressors)
X=[ones(t,1) ylag];
k2=size(X,2);

%----------------------------PRELIMINARIES---------------------------------
% Set some Gibbs - related preliminaries
nburn = 0.1*nrep;   % Number of burn-in-draws

%========= PRIORS:
% To set up training sample prior a-la Primiceri, use the following subroutine
%[B_OLS,VB_OLS,A_OLS,sigma_OLS,VA_OLS]= ts_prior(Y,tau,M,p);

% % Or use uninformative values
 B_OLS = zeros(K,1);
 VB_OLS = eye(K);

%-------- Now set prior means and variances (_prmean / _prvar)
% This is the Kalman filter initial condition for the time-varying
% parameters B(t)
% B_0 ~ N(B_OLS, 4Var(B_OLS))
B_0_prmean = B_OLS;

% Note that for IW distribution I keep the _prmean/_prvar notation...
% Q is the covariance of B(t)
% Q ~ IW(k2_Q*size(subsample)*Var(B_OLS),size(subsample))
Q_prvar = tau;

% Sigma is the covariance of the VAR covariance, SIGMA
% Sigma ~ IW(I,M+1)
Sigma_prmean = eye(M);
Sigma_prvar = M+1;

%========= INITIALIZE MATRICES:
% Specify covariance matrices for measurement and state equations
consQ = 0.001;
%Qchol = sqrt(consQ)*eye(K);
Btdraw = zeros(K,t); %#ok<*PREALL>
Sigmadraw = 0.1*eye(M);

% vprior choice
if vprior==1
    B_0_prvar=4*VB_OLS;
    Q_prmean=((0.01).^2)*tau*VB_OLS;
    Qdraw=consQ*eye(K);
elseif vprior==2
    B_0_prvar=[];
    Q_prmean=[];
    Qdraw=[];
    for iii=1:M
        if iii==1
            tmpB=4*eye(k2);
            tmpQ=((0.01).^2)*tau*4*eye(k2);
            tmpQQ=consQ*eye(k2);
            tmpB(2,2)=eps;
            tmpQ(2,2)=eps;
            tmpQQ(2,2)=eps;
        else
            tmpB=eps*eye(k2);
            tmpQ=eps*eye(k2);
            tmpQQ=eps*eye(k2);
            tmpB(1,1)=4;
            tmpQ(1,1)=((0.01).^2)*tau*4;
            tmQQ(1,1)=consQ;
            tmpB(iii+1,iii+1)=4;
            tmpQ(iii+1,iii+1)=((0.01).^2)*tau*4;
            tmQQ(iii+1,iii+1)=consQ;
        end
        B_0_prvar=blkdiag(B_0_prvar,tmpB);
        Q_prmean=blkdiag(Q_prmean,tmpQ);
        Qdraw=blkdiag(Qdraw,tmpQQ);
    end
end

% Storage matrices for posteriors and stuff
Bt_postmean = zeros(K,t);
Qmean = zeros(K,K);
Sigmamean = zeros(M,M);

%----------------------------- END OF PRELIMINARIES ---------------------------

%====================================== START SAMPLING ========================================
%==============================================================================================

for irep = 1:nrep + nburn    % GIBBS iterations starts here
    % -----------------------------------------------------------------------------------------
    %   STEP I: Sample B_t from p(B_t|y,Sigma) (Drawing coefficient states, pp. 844-845)
    % -----------------------------------------------------------------------------------------
    [Btdraw,~] = carter_kohn_hom(y,Z,Sigmadraw,Qdraw,K,M,t,B_0_prmean,B_0_prvar);

    Btemp = Btdraw(:,2:t)' - Btdraw(:,1:t-1)';
    sse_2Q = zeros(K,K);
    for i = 1:t-1
        sse_2Q = sse_2Q + Btemp(i,:)'*Btemp(i,:);
    end

    Qinv = inv(sse_2Q + Q_prmean);%+eps*eye(size(sse_2Q,1));
    try
        Qinvdraw = wish(Qinv,t+Q_prvar);
    catch
        Qinvdraw = wish(10*eye(size(Qinv)),t+Q_prvar);
    end
    Qdraw = inv(Qinvdraw);
    %Qchol = chol(Qdraw);
    
    % -----------------------------------------------------------------------------------------
    %   STEP I: Sample Sigma from p(Sigma|y,B_t) which is i-Wishart
    % ----------------------------------------------------------------------------------------
    yhat = zeros(M,t);
    for i = 1:t
        yhat(:,i) = y(:,i) - Z((i-1)*M+1:i*M,:)*Btdraw(:,i);
    end
    
    sse_2S = zeros(M,M);
    for i = 1:t
        sse_2S = sse_2S + yhat(:,i)*yhat(:,i)';
    end
    
    Sigmainv = inv(sse_2S + Sigma_prmean);
    Sigmainvdraw = wish(Sigmainv+eps*eye(size(Sigmainv)),t+Sigma_prvar);
    Sigmadraw = inv(Sigmainvdraw);
    %Sigmachol = chol(Sigmadraw);
    
    %----------------------------SAVE AFTER-BURN-IN DRAWS AND IMPULSE RESPONSES -----------------
    if irep > nburn;
        % Save only the means of B(t), Q and SIGMA. Not memory efficient to
        % store all draws (at least for B(t) which is large). If you want to
        % store all draws, it is better to save them in a file at each iteration.
        % Use the MATLAB command 'save' (type 'help save' in the command window
        % for more info)
        Bt_postmean = Bt_postmean + Btdraw;
        %Qmean = Qmean + Qdraw;
        Sigmamean = Sigmamean + Sigmadraw;
        
                %% Forecasting
        % The usual way is to write the VAR(p) model in companion form, i.e. as VAR(1) model in order to estimate the
        % h-step ahead forecasts directly (this is similar to the code we use below to obtain impulse responses). Here we 
        % just iterate over h periods, obtaining the forecasts at T+1, T+2, ..., T+h iteratively.
                
        ALPHA=reshape(Btdraw(:,end),k2,M);
        SIGMA=Sigmadraw;
        T=t;
        for ii = 1:repfor
                % Forecast of T+1 conditional on data at time T
                X_fore=[1 Y(T,:) X(T,2:M*(p-1)+1)];
                Y_hat=X_fore*ALPHA + randn(1,M)*chol(SIGMA);
                Y_temp=Y_hat;
                X_new_temp=X_fore;
                for i = 1:h-1  % Predict T+2, T+3 until T+h                   
                    if i<=p
                        % Create matrix of dependent variables for predictions. Dependent on the horizon, use the previous                       
                        % forecasts to create the new right-hand side variables which is used to evaluate the next forecast.                       
                        X_new_temp=[1 Y_hat X_fore(:,2:M*(p-i)+1)];
                        % This gives the forecast T+i for i=1,..,p                       
                        Y_temp=X_new_temp*ALPHA+randn(1,M)*chol(SIGMA);                       
                        Y_hat=[Y_hat Y_temp];
                    else
                        X_new_temp=[1 Y_hat(:,1:M*p)];
                        Y_temp=X_new_temp*ALPHA+randn(1,M)*chol(SIGMA);
                        Y_hat=[Y_hat Y_temp];
                    end
                end %  the last value of 'Y_temp' is the prediction T+h
                Y_temp2(ii,:)=Y_temp;
            end
            % Matrix of predictions               
            Y_pred(((irep-nburn)-1)*repfor+1:(irep-nburn)*repfor,:) = Y_temp2;
            % Predictive likelihood
            SIGMA=tidy_cov_mat(SIGMA);
            PL(irep-nburn,:) = mvnpdf(ydata(T+h,:),X_new_temp*ALPHA,SIGMA);
            if PL(irep-nburn,:) == 0
                PL(irep-nburn,:)=1;
            end
        
            %----- Save draws of the parameters
        ALPHA_draws(irep-nburn,:,:)=ALPHA;
        SIGMA_draws(irep-nburn,:,:)=SIGMA;
        
    end % END saving after burn-in results 
end %END main Gibbs loop (for irep = 1:nrep+nburn)

%% Save all results
results.ALPHA=ALPHA_draws;
results.SIGMA=SIGMA_draws;
results.predictions=Y_pred;

