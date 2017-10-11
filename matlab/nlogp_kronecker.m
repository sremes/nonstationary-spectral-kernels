function [l,g,K] = nlogp_kronecker(hyp, u, x, hyp_kernel)
% marginal likelihood and gradients for the generalized spectral mixture product (GSM-P) kernel
% using kronecker inference on a multidimensional grid
% x: cell array of length P containing the input points along all P axes
% u: P-dimensional array
% hyp: {P,A} cell array for parameters of each P dimensions and A mixture
% components

[P,A] = size(hyp.log_w);

% compute kernels
noise = exp(2*hyp.log_noise);
hyp.log_noise = -inf; % without noise
K = cell(P,1); dK = cell(P,1);
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
    if nargout == 1
        K{p} = inputdep_gibbs(x{p}, x{p}, hyp_p);
    else
        [K{p},~,dK{p}] = inputdep_gibbs(x{p}, x{p}, hyp_p);
    end
    K{p} = (K{p} + K{p}') / 2 + 1e-8*eye(numel(x{p}));
end

% compute MLL = log N(vec(u)|0, K{1} x ... x K{P} + sigma^2 I)
% following notation of GPatt of Wilson (2014) / Saatchi (2011)
Q = cell(P,1); V = cell(P,1); Qt = Q; %Vinv = V;
eig_vals = 1;
for p = 1:P
    [Q{p}, V{p}] = eig(K{p} + 1e-8*eye(numel(x{p})));
    Qt{p} = Q{p}';
    assert(all(isreal(V{p})),'non-real eigen values');
    assert(all(isreal(Q{p})),'non-real eigen vectors');
    eig_vals = kron(eig_vals, diag(V{p}));
end
eig_vals = real(eig_vals + noise);

Kinv_u = kron_mv(Q, kron_mv(Qt,u(:)) ./ eig_vals);
l = 0.5 * (sum(log(eig_vals)) + u(:)'*Kinv_u(:));

% Kinv_u2 = (kron(K{1},K{2}) + noise*eye(numel(u)))\u(:);
% l2 = 0.5*(logdet(kron(K{1},K{2})+noise*eye(numel(u))) + u(:)'*Kinv_u2);
% add prior terms
for p = 1:P 
    for a = 1:A
        l = l - sum(logmvnpdf(hyp.log_mu{p,a}', hyp_kernel{p}.mu_mu*ones(1,length(x{p})), hyp_kernel{p}.K_mu)) ...
                - logmvnpdf(hyp.log_sigma{p,a}', hyp_kernel{p}.mu_sigma*ones(1,length(x{p})), hyp_kernel{p}.K_sigma) ...
                - logmvnpdf(hyp.log_w{p,a}', hyp_kernel{p}.mu_w*ones(1,length(x{p})), hyp_kernel{p}.K_w);
    end
end

% GRADIENTS (Saatci's Thesis)
if nargout > 1
    diag_QtKQs = cell(P,1);
    for p = 1:P % precompute
        diag_QtKQs{p} = diag(Qt{p} * K{p} * Q{p});
    end
    for p = 1:P
        d_kernel = K;
        d_diag = diag_QtKQs;
        for a = 1:A
            g.log_w{p,a} = zeros(length(x{p}),1);
            g.log_mu{p,a} = zeros(length(x{p}),1);
            g.log_sigma{p,a} = zeros(length(x{p}),1);
            for n = 1:length(x{p})
                % log_w
                d_kernel{p} = dK{p}.log_w{a}(:,:,n);
                d_diag{p} = diag(Qt{p} * d_kernel{p} * Q{p});
                g.log_w{p,a}(n) = kron_deriv(d_kernel, d_diag, Kinv_u, eig_vals);
                
                % log_mu
                d_kernel{p} = dK{p}.log_mu{a}(:,:,n);
                d_diag{p} = diag(Qt{p} * d_kernel{p} * Q{p});
                g.log_mu{p,a}(n) = kron_deriv(d_kernel, d_diag, Kinv_u, eig_vals);
                
                % log_sigma
                d_kernel{p} = dK{p}.log_sigma{a}(:,:,n);
                d_diag{p} = diag(Qt{p} * d_kernel{p} * Q{p});
                g.log_sigma{p,a}(n) = kron_deriv(d_kernel, d_diag, Kinv_u, eig_vals);
            end
            % add prior terms and whitening
            g.log_w{p,a} = hyp_kernel{p}.Lw' * (g.log_w{p,a} + hyp_kernel{p}.Kw_inv * (hyp.log_w{p,a} - hyp_kernel{p}.mu_w));
            g.log_mu{p,a} = hyp_kernel{p}.Lmu' * (g.log_mu{p,a} + hyp_kernel{p}.Kmu_inv * (hyp.log_mu{p,a} - hyp_kernel{p}.mu_mu));
            g.log_sigma{p,a} = hyp_kernel{p}.Lsigma' * (g.log_sigma{p,a} + hyp_kernel{p}.Ksigma_inv * (hyp.log_sigma{p,a} - hyp_kernel{p}.mu_sigma));
        end
    end
    g.log_noise = 0.5*(-Kinv_u'*Kinv_u + sum(1./eig_vals)) * (2*noise);
end

function g = kron_deriv(d_kernel, d_diag, alpha, eig_vals)
kron_diag = 1;
for d = 1:length(d_kernel) %fliplr(1:length(d_kernel))
    kron_diag = kron(kron_diag, d_diag{d});
end
trace_term = sum(kron_diag ./ eig_vals);
norm_term = alpha' * kron_mv(d_kernel, alpha);
g = -0.5*norm_term + 0.5*trace_term;
% g = -g;



% function logdet = kron_logdet(V,noise_var)
% eigen_values = diag(V{1});
% for p=2:length(V)
%     v2 = diag(V{p});
%     tmp = eigen_values*v2';
%     eigen_values = tmp(:);
% end
% logdet = sum(log(eigen_values + noise_var));
%
% function mv = kron_mvprod(K,v)
% P = length(K);
% mv = reshape(v(:),size(K{P},1),[])' * K{P};
% for p = fliplr(1:P-1)
%     mv = reshape(mv(:),size(K{p},1),[])' * K{p};
% end
% mv = mv(:);
