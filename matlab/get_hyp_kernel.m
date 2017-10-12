function hyp_kernel = get_hyp_kernel(x, ell, sigma, omega)
% Returns hyperparameter kernels each with lengthscale "ell", 
% signal variance "sigma" and jitter/noise "omega".
hyp_kernel.ell = ell; 
hyp_kernel.sigma = sigma; 
hyp_kernel.omega = omega;

hyp_kernel.K_w = gausskernel(x, x, ell, sigma, omega);
hyp_kernel.Kw_inv = inv(hyp_kernel.K_w); 
hyp_kernel.Lw = chol(hyp_kernel.K_w)';

hyp_kernel.K_mu = gausskernel(x, x, ell, sigma, omega);
hyp_kernel.Kmu_inv = inv(hyp_kernel.K_mu); 
hyp_kernel.Lmu = chol(hyp_kernel.K_mu)';

hyp_kernel.K_sigma = gausskernel(x, x, ell, sigma, omega);
hyp_kernel.Ksigma_inv = inv(hyp_kernel.K_sigma); 
hyp_kernel.Lsigma = chol(hyp_kernel.K_sigma)';
