function [ustar] = inputdep_kron_predict(hyp, u, x, xstar, hyp_kernel)
% Compute predictions from u(x) -> ustar(xstar).
% hyp: kernel hyperparameters (latent functions mu(x), ell(x) and sigma(x))
% hyp_kernels: kernels for latent functions mu(x), ell(x), sigma(x)
% Note: consumes a lot of memory if xstar has too many samples! 
%    (may want to do predictions in batches)

[P,A] = size(hyp.log_w);

% compute kernels
noise = exp(2*hyp.log_noise);
hyp.log_noise = -inf; % without noise
K = cell(P,1); Q = cell(P,1); V=Q;Kstar=Q;Qt=Q; % initialize
eig_vals = 1;
for p = 1:P
    % unwhiten
    for a = 1:A
        hyp.log_mu{p,a} = hyp_kernel{p}.Lmu * (hyp.log_mu{p,a}) + (hyp_kernel{p}.mu_mu);
        hyp.log_w{p,a} = hyp_kernel{p}.Lw * (hyp.log_w{p,a}) + (hyp_kernel{p}.mu_w);
        hyp.log_sigma{p,a} = hyp_kernel{p}.Lsigma * (hyp.log_sigma{p,a}) + (hyp_kernel{p}.mu_sigma);
    end
    
    hyp_p = hyp;
    hyp_p.log_mu = hyp.log_mu(p,:);
    hyp_p.log_w = hyp.log_w(p,:);
    hyp_p.log_sigma = hyp.log_sigma(p,:);
    
    K{p} = inputdep_gibbs(x{p}, x{p}, hyp_p);
    K{p} = (K{p} + K{p}') / 2 + 1e-8*eye(numel(x{p}));
    [Q{p}, V{p}] = eig(K{p}); Qt{p} = Q{p}';
    eig_vals = kron(eig_vals, diag(V{p}));
    
    Kstar{p} = inputdep_gibbs(x{p}, xstar{p}, hyp_p, hyp_kernel{p});
end
% ustar = Kstar' * (K+sigma^2*I)^-1 * u
alpha = kron_mv(Q, kron_mv(Qt,u(:)) ./ (eig_vals+noise));
if P == 2 % using identity vec(AXB) = (B' \kron A)vec(X)
    alpha = reshape(alpha,[length(x{2}) length(x{1})]);
    ustar = Kstar{2}'*alpha*Kstar{1}; ustar = ustar(:);
else
    Ks = 1;
    for p=1:P
        Ks = kron(Ks,Kstar{p});
    end
    ustar = Ks'*alpha;
end
