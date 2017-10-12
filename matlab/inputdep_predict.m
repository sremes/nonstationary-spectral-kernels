function [y_pred,y_var] = inputdep_predict(x,y,z,hyp,hyp_kernels)
%% Predict output at z, given training input x and output y
% hyp: kernel hyperparameters (latent functions mu(x), ell(x) and sigma(x))
% hyp_kernels: kernels for latent functions mu(x), ell(x), sigma(x)

% Compute kernels:
Kxx = inputdep_gibbs(x,x,hyp);
% Kxz = inputdep_gibbs(x,z,hyp); % is not directly computable

% Compute the kernel Kxz and Kzz
A = length(hyp.log_w);
Kvar_xz = gausskernel(x,z,hyp_kernels.ell,hyp_kernels.sigma,hyp_kernels.omega);
hypz = hyp;
Nx = size(x,1); Nz = size(z,1);
Kxz = zeros(Nx,Nz);
Kzz = zeros(Nz,Nz);
for a = 1:A
    % predict the latent functions mu, sigma, w first
    
    hypz.log_mu{a} = Kvar_xz' * hyp_kernels.Kmu_inv * hyp.log_mu{a};
    hypz.log_sigma{a} = Kvar_xz' * hyp_kernels.Ksigma_inv * hyp.log_sigma{a};
    hypz.log_w{a} = Kvar_xz' * hyp_kernels.Kw_inv * hyp.log_w{a};

    l = exp(hyp.log_sigma{a}); lz = exp(hypz.log_sigma{a});
    Fs = Nx ./ (max(x(:)) - min(x(:))); Fn = Fs/2;
    mu = Fn./(1+exp(-hyp.log_mu{a})); muz = Fn./(1+exp(-hypz.log_mu{a}));
    w = exp(hyp.log_w{a}); wz = exp(hypz.log_w{a});
    l2 = l.^2*ones(Nz,1)' + ones(Nx,1)*lz.^2';
    D = pdist2(x,z,'squaredeuclidean');
    E = sqrt(2*(l*lz')./(l2)).*exp(-D./l2);
    Dz = pdist2(z,z,'squaredeuclidean');
    l2z = lz.^2*ones(Nz,1)'+ones(Nz,1)*lz.^2';
    Ez = sqrt(2*(lz*lz')./l2z).*exp(-Dz./l2z);
    phi1 = [cos(2*pi*sum(mu.*x,2))  1*sin(2*pi*sum(mu.*x,2))];
    phi2 = [cos(2*pi*sum(muz.*z,2))  1*sin(2*pi*sum(muz.*z,2))];
    Kxz = Kxz + (w*wz') .* E .* (phi1*phi2');
    Kzz = Kzz + (wz*wz') .* Ez .* (phi2*phi2');
end

Kzz = 0.5*(Kzz + Kzz') + 1e-4 * eye(Nz); % ensure PSD'ness

y_pred = Kxz' * (Kxx \ y);
y_var = Kzz - Kxz'*(Kxx\Kxz);
y_var = 0.5 * (y_var + y_var') + 1e-4*eye(Nz);
