function hyp = init_inputdep(u,x,A,ell)
% Init the GSM kernel by fitting GMM's on the spectrogram of the data.
% u: signal values
% x: input points (regularly spaced!)
% A: number of mixture components in GSM
% ell: length-scale of gaussian kernel to be used for interpolating from spectrogram -> x


N = length(x); dt = max(x) - min(x);
Fs = N/dt;

% compute spectrogram at frequencies F and time points T
[S,F,T] = spectrogram(u,[],[],[],Fs);
idx = (F < 0.5 | F > (Fs/4)); % remove very small/big freqs
S = S(~idx,:); F = F(~idx);
spectrogram(u,[],[],[],Fs)

% find A peaks at the first time point, and find the closest peaks at next
% the time points, interpolate linearly between the time points
[mu(:,1),sigma(:,1),w(:,1),prev] = fit_gmm_spec_density(F,S(:,1),A);
for t = 2:length(T)
    [mu(:,t),sigma(:,t),w(:,t),prev] = fit_gmm_spec_density(F,S(:,t),A,prev);
end

Kxt = gausskernel(x,T'-1,ell);
Ktt = gausskernel(T'-1,T'-1,ell,1,1e-1);
for a = 1:A
    hyp.log_mu{a} = Kxt*(Ktt \ logit(mu(a,:)',Fs/2));
    hyp.log_sigma{a} = Kxt*(Ktt \ log(2./sqrt(sigma(a,:)')));
    hyp.log_w{a} = Kxt*(Ktt \ log(std(u)*sqrt(w(a,:)')));
end
hyp.log_noise = 0;

function [mu,sigma,w,gm] = fit_gmm_spec_density(F,S,A,prev)
%% fit GMM on a spectral density

% create a fake dataset from the density
lS = max(0,log(abs(S(:,1)).^2));
area = trapz(F,lS);
cdf = cumtrapz(F,lS) / area;
nsamp = 1e5;
[cdf,idx] = unique(cdf);
X = interp1(cdf,F(idx),rand(nsamp,1),'linear',0);

% fit gmm
if ~exist('prev','var')
    gm = fitgmdist(X,A);% plot((0:.01:50),pdf(gm,(0:.01:50)'));
else
    gm = fitgmdist(X,A,'Start',prev);% plot((0:.01:50),pdf(gm,(0:.01:50)'));
end
mu = gm.mu; sigma = gm.Sigma(:); w = gm.ComponentProportion;
gm = struct(gm);
