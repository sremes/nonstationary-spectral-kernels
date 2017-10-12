# Non-Stationary Spectral Kernels
Matlab implementation for the following paper
 * S. Remes, M. Heinonen, S. Kaski (2017). *Non-stationary Spectral Kernels*. Accepted for NIPS 2017. Preprint: https://arxiv.org/abs/1705.08736

## Example

To use the GSM kernel, we need to first define the kernels for the latent functions: frequencies *mu(x)*, lengthscales *ell(x)* and mixture weights *w(x)*. This is also provided in a helper function *get_hyp_kernel.m*.
```matlab
ell = 1; sigma = 1; omega = 1e-4; % some standard values
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
```

Next we need to define the latent functions for *A* mixture components, let's use some simple constant functions for this example:
```matlab
hyp.log_sigma = cell(A,1);
hyp.log_mu = cell(A,1);
hyp.log_noise = log(1e-1);
hyp.log_w = cell(A,1);
freq = [3.5,8];
w = [2 1];
for a = 1:A
    hyp.log_mu{a} = logit(freq(a)*ones(size(x)), Fn); 
    hyp.log_sigma{a} = -1*ones(size(x));
    hyp.log_w{a} = log(w(a)*ones(size(x))); 
end
hyp.log_noise = log(1);
```
A function *init_inputdep.m* is also provided for initializing the hyperparameters, which is based on fitting a GMM on the spectrogram of the data.

Now we can use the defined hyperparameters to compute the GSM kernel:
```matlab
K = inputdep_gibbs(x, x, hyp);
Kxy = inputdep_gibbs(x, y, hyp, hyp_kernel); 
% hyp_kernel needed to interpolate latent functions to new inputs "y"
```

To learn the latent functions from data, we use *minimize_v2.m*. First whiten the latent functions, and unwhiten them after optimization.
```matlab
hyp = inputdep_whiten_vars(hyp, hyp_kernel);
hyp = minimize_v2(hyp, @nlogp_gibbs, -100, u, x, hyp_kernel);% u is the data with inputs x
hyp = inputdep_unwhiten_vars(hyp, hyp_kernel);
```

With Kronecker inference, we define our inputs *x* as a cell array:
```matlab
x = {x1, x2, x3}; % for a 3d input grid
% u contains output values with length |x1|*|x2|*|x3|
hyp = init_kron(x, A, hyp_kernel, noise_var, max_freq); % random initialization with A components
hyp = minimize_v2(hyp, @nlogp_kronecker, -100, u, x, hyp_kernel);
```
