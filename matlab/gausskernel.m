function K = gausskernel(X1,X2, ell, sigma, omega)
% gaussian kernel
	
	if ~exist('ell','var')
		ell = 1;
	end
	if ~exist('sigma','var')
		sigma = 1;
	end
	if ~exist('omega','var')
		omega = 0;
	end

	if length(X1) == length(X2)
		K = sigma^2 * exp(-0.5* pdist2(X1,X2).^2 / ell^2) + omega^2*eye(size(X1,1));
	else
		K = sigma^2 * exp(-0.5* pdist2(X1,X2).^2 / ell^2);
	end
end

