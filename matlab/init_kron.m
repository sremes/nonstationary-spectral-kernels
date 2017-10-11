function hyp = init_kron(x,A,hyp_kernel,noise,max_freq)
P = length(x);
for a=1:A
    for p=1:P
        N = size(x{p},1);
        Fs = N ./ (x{p}(end,:) - x{p}(1,:)); Fn = Fs/2;
        freq = min(0.99*Fn,(1e-6 + max_freq*rand));
        mu = freq*ones(size(x{p})); 
        hyp.log_mu{p,a} = hyp_kernel{p}.Lmu \ (log(mu ./ (Fn-mu)) - hyp_kernel{p}.mu_mu);
        ell = 0.1 + 0.9*rand;
        hyp.log_sigma{p,a} = hyp_kernel{p}.Lsigma \ (log(ell*ones(size(x{p}))) - hyp_kernel{p}.mu_sigma);
        hyp.log_w{p,a} = hyp_kernel{p}.Lw \ (log(1/A*ones(size(x{p}))) - hyp_kernel{p}.mu_w);
    end
end
hyp.log_noise = log(noise);
