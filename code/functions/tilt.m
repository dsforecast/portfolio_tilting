function results = tilt(dat,w,g,gbar,tol,nsim,base_method)

% Purpose: Entropic tilting modifies a simulated forecast distribution such that it satisfies certain moment conditions of interest. See the paper by KCR15 for details, and the example below for an illustration from that paper.
% Inputs:
% dat                  = Matrix of simulated data, where columns represent simulation draws (typically a few thousand) and rows represent different variables or forecast horizons.
% w                     = Weights attached to observations. This is a vector, with length equal to the number of columns of dat. Defaults to NULL, which implies equal weights.
% g                     = Indictaor Function which characterizes tilting, see Section 4.1 in KCR15. Defaults to identity function. NOTE: Inputs to g are the columns of dat, the output of g is a vector with length equal to gbar below.
% gbar                = Vector which characterizes the tilting condition, see Section 4.1 in KCR15.
% tol	                = Tolerance level; the function issues a warning if the target specified in gbar is missed by more than tol, in absolute terms.
% nsim                = Number of desired output draws from tilted distribution. Defauls to zero.
% base_method = Optimizer to be used for numerical optimization via optim. Defaults to BFGS method.
%
% Outputs:
% gamma             = vector which solves the tilting problem (see Section 4.1 in KCR15)
% w                       = resulting vector of tilting weights
% klic                    = Kullback-Leibler divergence between original and tilting weights
% nu                     = amount of shrinkage (penalty for L2 norm of gamma) applied to regularize the problem. nu = 0 implies no shrinkage.
% sim                   = simulation draws from tilting distribution (if any).
% time                 = computation time.
% convergence   = convergence code for the numerical optimization problem. 0 indicates normal convergence.
%
% Reference: Krueger, F., T.E. Clark, and F. Ravazzolo (2015): “Using Entropic Tilting to Combine BVAR Forecasts with External Nowcasts”, Journal of Business and Economic Statistics, forthcoming.
gradi=1;
if nargin<1
    gradi=1;
    m=1;
    n=1000;
    dat=randn(m,n);
    w=ones(1,n)./n;
    %g=@(x) [x];
    %gbar=[zeros(m,1)];
    g=@(x) [x; x.^2];
    gbar=[1.*ones(m,1);3.*ones(m,1)];
end

tic
% Dimension
[~,n]=size(dat);
k=size(gbar,1);

% Check imputs
if not(ismember('w',who)), w=ones(1,n)./n; end
if not(ismember('tol',who)), tol=0.1; end
if not(ismember('nsim',who)),nsim=0; end
if not(ismember('base_method',who)),base_method='bfgs'; end

% Define function handles
if k==1
    of = @(l) sum(w.*exp(l'*(g(dat)-repmat(gbar,1,n))));
    options = optimoptions('fminunc','Display','off','Algorithm','quasi-newton','HessUpdate',base_method);
    [th,pen,cc,~]=fminunc(of,zeros(1,k)',options);
else  
    try
        of=@(l) deal(sum(w.*exp(l'*(g(dat)-repmat(gbar,1,n)))) , repmat(exp(l'*(g(dat)-repmat(gbar,1,n))),k,1).*(g(dat)-repmat(gbar,1,n))*w');       
        options= optimoptions('fminunc','Display','off','Algorithm','quasi-newton','HessUpdate',base_method,'GradObj','on');
        [th,pen,cc,~]=fminunc(of,zeros(1,k)',options);
    catch
        of=@(l) sum(w.*exp(l'*(g(dat)-repmat(gbar,1,n))));
        options= optimoptions('fminunc','Display','off','Algorithm','quasi-newton','HessUpdate',base_method,'GradObj','off');
        [th,pen,cc,~]=fminunc(of,zeros(1,k)',options);
    end
end
ws=w.*exp(th'*g(dat));
ws=ws./sum(ws);
%plot(ws)

nu=0;

if any(isnan(ws))==1 || cc~=1
    disp('Original problem failed - use some shrinkage');
    shrink=linspace(1e-10,0.2,20);
    ok=0;
    ct=1;
    while ok==0
        if k==1
            of2=@(l) sum(w.*exp(l'*(g(dat)-repmat(gbar,1,n)))) +shrink(ct).*pen.*sum(l.^2);
            options = optimoptions('fminunc','Display','off','Algorithm','quasi-newton','HessUpdate',base_method);
            th2=fminunc(of2,zeros(1,k)',options);
        else
            if gradi==1
                of2=@(l) deal(sum(w.*exp(l'*(g(dat)-repmat(gbar,1,n))))+shrink(ct).*pen.*sum(l.^2), repmat(exp(l'*(g(dat)-repmat(gbar,1,n))),k,1).*(g(dat)-repmat(gbar,1,n))*w');
                options= optimoptions('fminunc','Display','off','Algorithm','quasi-newton','HessUpdate',base_method,'GradObj','on');
                th2=fminunc(of2,zeros(1,k)',options);
            else
                of2=@(l) sum(w.*exp(l'*(g(dat)-repmat(gbar,1,n))))+shrink(ct).*pen.*sum(l.^2);
                options= optimoptions('fminunc','Display','off','Algorithm','quasi-newton','HessUpdate',base_method,'GradObj','off');
                th2=fminunc(of2,zeros(1,k)',options);
            end
        end
        ws=w.*exp(th2'*g(dat));
        ws=ws./sum(ws);
        ok=any(isnan(ws));
        if ok==0
            ct=ct+1;
        end
    end
    nu=shrink(ct);
end

tt=repmat(ws,k,1).*g(dat);
if k==1
    tt=sum(tt);
else
    tt=sum(tt,2);
end

if mean(abs(tt - gbar)) > tol
    disp('Tolerance exceeded:')
     disp('Empirical vs. target:')
     [tt,gbar]
end

if nsim > 0
    aux=datasample(1:n,nsim,'Replace',true,'Weights',ws);
    sim=dat(:,aux);
else
    sim=0;
end
toc

%plot(ws,'o')

% Outputs
results.gamma=th;
results.draws=ws.*dat;
results.w=ws;
results.klic=sum(ws.*log(ws/w));
results.nu=nu;
results.sim=sim;
results.convergence=cc;
results.time=tic-toc;

end


% Fabians R code
% function (dat, w = NULL, g = identity, gbar, tol = 0.1, nsim = 0, 
%     base_method = "BFGS") 
% {
%     t0 <- Sys.time()
%     if (!is.matrix(dat)) 
%         dat <- matrix(dat, 1, length(dat))
%     m <- dim(dat)[1]
%     n <- dim(dat)[2]
%     k <- length(gbar)
%     if (is.null(w)) 
%         w <- rep(1/n, n)
%     of <- function(l) {
%         return(sum(w * apply(dat, 2, function(z) exp(t(l) %*% 
%             (g(z) - gbar)))))
%     }
%     d1 <- function(l) {
%         return(apply(dat, 2, function(z) exp(t(l) %*% (g(z) - 
%             gbar)) * (g(z) - gbar)) %*% matrix(w))
%     }
%     if (k == 1) {
%         th <- optim(0, of, method = "BFGS")
%     }
%     else {
%         th <- optim(par = rep(0, k), fn = of, gr = d1, method = base_method)
%     }
%     cc <- th$convergence
%     pen <- th$value
%     th <- th$par
%     ws <- w * apply(dat, 2, function(z) exp(t(th) %*% g(z)))
%     ws <- ws/sum(ws)
%     nu.final <- 0
%     if (any(is.na(ws)) | (cc != 0)) {
%         warning("Original problem failed - use some shrinkage")
%         nu.shrink <- seq(from = 1e-10, to = 0.2, length.out = 20)
%         ok <- 0
%         ct <- 1
%         while (ok == 0) {
%             of2 <- function(l) sum(w * apply(dat, 2, function(z) exp(t(l) %*% 
%                 (g(z) - gbar)))) + nu.shrink[ct] * pen * sum(l^2)
%             th2 <- optim(rep(0, k), of2, method = "BFGS")$par
%             ws <- w * apply(dat, 2, function(z) exp(t(th2) %*% 
%                 g(z)))
%             ws <- ws/sum(ws)
%             ok <- !any(is.na(ws))
%             if (ok == 0) 
%                 ct <- ct + 1
%         }
%         nu.final <- nu.shrink[ct]
%     }
%     tt <- (apply(as.matrix(1:n), 1, function(z) ws[z] * g(dat[, 
%         z])))
%     if (k == 1) 
%         tt <- sum(tt)
%     else tt <- apply(tt, 1, sum)
%     if (mean(abs(tt - gbar)) > tol) 
%         warning(paste0("Tolerance exceeded: empirical = ", tt, 
%             " target = ", gbar))
%     if (nsim > 0) {
%         aux <- sample(1:n, nsim, replace = TRUE, prob = ws)
%         sim <- dat[, aux]
%     }
%     else {
%         sim <- NULL
%     }
%     t1 <- Sys.time()
%     list(gamma = th, w = ws, klic = sum(ws * log(ws/w)), nu = nu.final, 
%         sim = sim, time = t1 - t0, convergence = cc, w_orig = w, 
%         data = dat)
% }
