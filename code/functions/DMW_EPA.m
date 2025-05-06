% //-----------------------------------------------------------------------------------------------------------------------------------
% // DMW_EPA(_forecastErrors, _nested) - test for population level equal predictive ability (cp. West(2006), p.128f)
% //-----------------------------------------------------------------------------------------------------------------------------------
% // Inputs: 
% // - _forecastErrors  - vector [Tx2] of forecast errors
% // - _nested = 1 - models are nested (!!!: 1st column of _forecastErrors -> nested model)
% //-----------------------------------------------------------------------------------------------------------------------------------
% // NOTE: 
% // - p-values refer to one-sided alternative hypothesis 
% //		(test with nested models is one-sided. HA: in population larger model has lower MSFE 
% //		- if nested model is correct, estimated coeff. is zero -> forecasts are identical)
% // - validity of critical values of nested test relies on West(2006),6.2:
% //		- MSFE criterion, NLS estimator
% //		- direct not iterated forecasts
% //		- one step ahead predictions and conditionally homoscedastic prediction errors, or
% //		- the number of additional regressor in the larger model is exactly 1
% //-----------------------------------------------------------------------------------------------------------------------------------
function [out_] = DMW_EPA(fe_, nested_,compact_) 
%      fe_ = randn(100,2);
%      nested_ = 1;
     
    nobs_ = size(fe_,1);
	x_ = ones(nobs_,1);
	sfe_ = fe_.*fe_;

	if nested_ == 0
		y_ = sfe_(:,1) - sfe_(:,2);
    else
        corr_ = mean((fe_(:,1)-fe_(:,2)).^2);
		y_ = sfe_(:,1)-(sfe_(:,2)-corr_);
    end
	
	b_ = inv(x_'*x_)*x_'*y_;  			%auxiliary regression estimated coefficient.
	e_ = y_-x_*b_;
	se_ = sqrt(nwvcv(e_,x_));      %HAC S.E. of auxiliary regression parameter estimate.
	
	% Compute test statistic  
		stat_ = b_/se_;
        if nested_ == 0
			out_ = [mean(sfe_(:,2))/mean(sfe_(:,1)) ; stat_ ; normcdf(-abs(stat_),0,1)];
        else
            out_ = [mean(sfe_(:,2))/mean(sfe_(:,1)) ; stat_ ; normcdf(-stat_,0,1)];
        end
        
        if compact_==1
            out_ = out_(3);
        end
        
        
