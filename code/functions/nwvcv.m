function [out_] = nwvcv(e_,x_)
%     Input:  e, error term vector obtained from some TS model
%             x, regressors
%     Output: Newey-West (1987) estimate of vcv matrix
%     !!! Note: Truncation lag chosen as floor( 4*(0.01*t)^0.25 ), as suggested by L?tkepohl and Kr?tzig (2004, p.57)
t_ = size(e_,1);
trunc_ = floor( 4*(0.01*t_)^(0.25) );
aux_ = e_.*x_;
s_ = aux_'*aux_;
for z_=1:trunc_
    aux1_ = e_(1:t_-z_).*x_(1:t_-z_,:);
    aux2_ = e_(z_+1:t_).*x_(z_+1:t_,:);
    w_ = 1-z_/(trunc_+1);
    s_ = s_+w_*(aux1_'*aux2_+aux2_'*aux1_);
end
out_ = inv(x_'*x_)*s_*inv(x_'*x_);
 