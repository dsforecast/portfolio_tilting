function draws = Draw_InvGaus(mu,lambda)

% Purpose: Draw Random Numbers from Inverse Gausian Distribution
% Input:  mu     = scalar, location parameter
%         lambda = scalar, scale parameter(T x N-1) matrix of right-handside centered variables
% Output: draw = random number

v0=randn.^2;
draws=mu+(.5*(mu.^2)*v0)/lambda-(.5*mu/lambda)*sqrt(4*mu*lambda*v0+(mu.^2)*(v0.^2));

if rand>=mu/(mu+draws)
	draws=(mu.^2)/draws;
end

end

