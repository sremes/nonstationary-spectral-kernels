function [l,g] = nlogp_gibbs(hyp, u, x, hyp_kernel)
% Compute the negative log posterior and its gradient, used for optimizing 
% hyperparameters i.e. latent functions mu(x), ell(x) and sigma(x)).
% hyp: kernel hyperparameters (latent functions mu(x), ell(x) and sigma(x))
% u: observed outputs u(x)
% x: input points
% hyp_kernels: kernels for latent functions mu(x), ell(x), sigma(x)
N = size(x,1);
A = length(hyp.log_w);

% unwhiten
for a = 1:A
    hyp.log_mu{a} = hyp_kernel.Lmu * hyp.log_mu{a};
    hyp.log_w{a} = hyp_kernel.Lw * hyp.log_w{a};
    hyp.log_sigma{a} = hyp_kernel.Lsigma * hyp.log_sigma{a};
end

if nargout == 1
    K = inputdep_gibbs(x, x, hyp);
else
    [K,dK] = inputdep_gibbs(x, x, hyp);
end

l = 0.5 * (logdet(K) + u'*(K\u));
% add prior terms
for a = 1:A
    l = l - sum(logmvnpdf(hyp.log_mu{a}', zeros(1,N), hyp_kernel.K_mu)) ...
            - logmvnpdf(hyp.log_sigma{a}', zeros(1,N), hyp_kernel.K_sigma) ...
            - logmvnpdf(hyp.log_w{a}', zeros(1,N), hyp_kernel.K_w);
end

if nargout > 1
    a = K\u;
    R = -0.5*(a*a'-inv(K));
    g = dK(R);
    
    % add prior terms and whitening
    for a = 1:A
        g.log_w{a} = hyp_kernel.Lw' * (g.log_w{a} + hyp_kernel.Kw_inv * hyp.log_w{a});
        g.log_mu{a} = hyp_kernel.Lmu' * (g.log_mu{a} + hyp_kernel.Kmu_inv * hyp.log_mu{a});
        g.log_sigma{a} = hyp_kernel.Lsigma' * (g.log_sigma{a} + hyp_kernel.Ksigma_inv * hyp.log_sigma{a});
    end
%     g = unwrap(g);
%     g(g<-10) = -10; g(g>10) = 10;
%     g = rewrap(hyp,g);
end
