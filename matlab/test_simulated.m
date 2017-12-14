%% runs GSM on a simulated time series which contains a decreasing frequency trend

rng(17)
clear

%% define inputs and hyperparameter kernels
N = 200;
x = linspace(-1,1,N)';
y = x;
A = 1;
ell = 1; sigma = 1; omega = 1e-4;
hyp2.ell = ell; hyp2.sigma = sigma; hyp2.omega = omega;
hyp2.K_w = gausskernel(x,y,ell,sigma,omega);
hyp2.K_mu = gausskernel(x,y,ell,sigma,omega);
hyp2.K_sigma = gausskernel(x,y,ell,sigma,omega); % this is the the length-scale \ell in the paper!
hyp2.Kw_inv = inv(hyp2.K_w); hyp2.Lw = chol(hyp2.K_w)';
hyp2.Kmu_inv = inv(hyp2.K_mu); hyp2.Lmu = chol(hyp2.K_mu)';
hyp2.Ksigma_inv = inv(hyp2.K_sigma); hyp2.Lsigma = chol(hyp2.K_sigma)';

%% define hyperparameters (i.e. the latent functions)
clear hyp
hyp.log_sigma = cell(A,1);
hyp.log_mu = cell(A,1);
hyp.log_noise = log(1e-1);
hyp.log_w = cell(A,1);
freq = [3.5,8];
w = [2 1];
for a = 1:A
    hyp.log_mu{a} = log(freq(a)*ones(size(x))); 
    hyp.log_sigma{a} = -1*ones(size(x));
    hyp.log_w{a} = log(w(a)*ones(size(x))); 
end

%% compute kernel and generate data
Fn = 0.5 * (max(x)-min(x)) / (x(2)-x(1));
hyp.log_mu{1} = flipud(logit(1+(x+1).^2,Fn));
[K] = inputdep_gibbs(x,x,hyp);
u = mvnrnd(zeros(N,1), K)';

figure(1)
imagesc(x,x,K),colorbar

figure(2)
plot(x,u)

%% optimize
p.length = -300;
p.verbosity = 10;
p.method = 'LBFGS';
p.SIG = 1-1e-4;
A = 1;
hyp_rand = init_inputdep(u,x,A,1); 

for a = 1:A
    hyp_rand.log_sigma{a} = hyp_rand.log_sigma{a} - 2; % it's useful to start from a more diagonal kernel
end
hyp_rand.log_noise = log(1e-1);

hyp_rand = inputdep_whiten_vars(hyp_rand, hyp2);
D = numel(unwrap(hyp_rand));

figure(3), clf
hyp_opt = hyp_rand; f_opt = nlogp_gibbs(hyp_opt,u,x,hyp2);
for iter = 1:3% try multiple restarts
    hyp_rand2 = rewrap(hyp_rand,unwrap(hyp_rand)+1e-2*randn(D,1)); % add some noise
    [hyp_tmp,f_tmp] = minimize_v2(hyp_rand2, @nlogp_gibbs, p, u, x, hyp2);
    if f_tmp(end) < f_opt
        hyp_opt = hyp_tmp;
        f_opt = f_tmp(end);
    end
end

hyp_opt = inputdep_unwhiten_vars(hyp_opt,hyp2);
hyp_rand = inputdep_unwhiten_vars(hyp_rand,hyp2);

%% plot kernel
figure(1),clf
subplot(121)
Ktrue = inputdep_gibbs(x,x,hyp);
imagesc(x,x,Ktrue),colorbar
title('True kernel')

subplot(122)
Kopt = inputdep_gibbs(x,x,hyp_opt);
imagesc(x,x,Kopt),colorbar
title('Learned kernel')

%% plot data and posterior using the learned kernel
figure(4),clf
xt = linspace(-1.5,1.1,500)';
[uhat,uvar] = inputdep_predict(x,u,xt,hyp_opt,hyp2);
uvar = sqrt(max(-inf,diag(uvar)) + exp(hyp_opt.log_noise));
h = fill([xt; flip(xt)], [uhat+2*uvar; flip(uhat-2*uvar)], 'red','facealpha', 0.20);
set(h, 'EdgeColor','none');
hold on;
plot(x,u,'o', xt,uhat,'-k')

legend('Confidence interval','Observation','Posterior mean','Location','SouthEast')
title('Simulated time series with GSM kernel')
axis auto
xlim([-1.5,1.1])
xlabel('$x$')
