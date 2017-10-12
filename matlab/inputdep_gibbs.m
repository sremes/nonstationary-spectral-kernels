function [K,dhyp,dKdt] = inputdep_gibbs(x, y, hyp, hyp_kernels)
%% Generalized spectral mixture (GSM) kernel
% x, y: input points
% hyp: kernel hyperparameters (latent functions mu(x), ell(x) and sigma(x))
% hyp_kernels: kernels for latent functions mu(x), ell(x), sigma(x)

K = zeros(size(x,1),size(y,1));
A = length(hyp.log_w);
N = size(x,1);
Ny = size(y,1);
P = size(x,2);
for a = 1:A
    l = exp(hyp.log_sigma{a}); l_y = l;
    % limit mu by half the Nyquist frequency
    Fs = N ./ (max(x(:)) - min(x(:))); Fn = Fs/2;
    mu = Fn ./ (1+exp(-hyp.log_mu{a})); mu_y = mu;    
    w = exp(hyp.log_w{a}); w_y = w;
    
    if nargin == 4 % test data case, interpolate the latent functions
        Kxy = gausskernel(x,y,hyp_kernels.ell,hyp_kernels.sigma,hyp_kernels.omega);
        l_y = exp(hyp_kernels.mu_sigma+Kxy'*(hyp_kernels.K_sigma\(hyp.log_sigma{a}-hyp_kernels.mu_sigma)));
        mu_y = Fn ./ (1+exp(-hyp_kernels.mu_mu-Kxy'*(hyp_kernels.K_mu\(hyp.log_mu{a}-hyp_kernels.mu_mu))));
        w_y = exp(hyp_kernels.mu_w+Kxy'*(hyp_kernels.K_w\(hyp.log_w{a}-hyp_kernels.mu_w)));
    end
    
    l2 = l.^2*ones(Ny,1)' + ones(N,1)*l_y.^2';
    D = pdist2(x,y,'squaredeuclidean');
    E = sqrt(2*(l*l_y')./(l2)).*exp(-D./l2);
    phi1 = [cos(2*pi*sum(mu.*x,2))  1*sin(2*pi*sum(mu.*x,2))];
    phi2 = [cos(2*pi*sum(mu_y.*y,2))  1*sin(2*pi*sum(mu_y.*y,2))];
    Ka = (w*w_y') .* E .* (phi1*phi2');
    K = K + Ka;
    if nargout > 1 % compute gradients as well
        % w
        oneN = ones(N,1); %oneN(idx) = 0; oneN = ~oneN;
        tmp = (oneN*w' + w*oneN') .* E .* (phi1*phi2');
        dK.log_w{a} = @(R)  diag(R * tmp) .* w; % Seems OK (checkgrad)
        if nargout > 2
            dKdt.log_w{a} = zeros([ N N N ]);
            for n = 1:N
                n1 = zeros(N,1); n1(n) = 1;
                dKdt.log_w{a}(:,:,n) = (oneN*n1' + n1*oneN') .* tmp * w(n) * .5;
            end
        end

        % mu
        const = (w*w') .* E;
        temp_funs = cell(N,P);
        phi1 = sparse(phi1);
        phi2 = sparse(phi2);
        for d = 1:P
            dphi1 = [-2*pi*x(:,d).*sin(2*pi*sum(mu.*x,2)),  2*pi*x(:,d).*cos(2*pi*sum(mu.*x,2))];
            dKdt.log_mu{a} = zeros([N N N]);
            for n=1:N
                dphi = sparse(N,2); dphi(n,:) = dphi1(n,:);
                tmp = full(const .* (dphi*phi2' + phi1*dphi'));
                temp_funs{n,d} = @(R) sum(R(:) .* tmp(:)) * mu(n,d) * (1-mu(n,d)/Fn); 
                if nargout > 2
                    dKdt.log_mu{a}(:,:,n) = tmp * mu(n,d) * (1-mu(n,d)/Fn);
                end
            end
            
        end
        dK.log_mu{a} = @(R) cellfun(@(f) f(R), temp_funs); % seems good!
        phi1 = full(phi1);
        phi2 = full(phi2);

        % sigma
        const = (w*w') .* (phi1*phi2');
        temp_funs = cell(N,1);
        
%         tic
%         [XX,YY] = meshgrid(l);
%         tmp_grad = -YY.*(XX.^4 - YY.^4 -4*XX.^2.*D)./(sqrt(2*XX.*YY./(XX.^2+YY.^2)).*(XX.^2+YY.^2).^3).*E;
%         oneN = ones(N,1);
        dKdt.log_sigma{a} = zeros([N N N]);
        
%         L = sqrt(2*(l*l') ./ (l.^2 + l'.^2)).^(-3);
%         LE = L.*E;
%         L1 = repmat(l', N,1);
%         L4 = l.^4 - l.^4';
%         L2 = 4 * l.^2 * ones(1,N);
%         dL = -L1.*( L4 -L2.*D) .* LE;
%         for n = 1:N
%             cross = sparse(N,N);
%             cross(n,1) = 1; cross(1,n) = 1;
%             tmp = full(dL .* cross);
%             temp_funs{n} = @(R) sum(R(:).*const(:).*tmp(:)) * (l(n));
%             if nargout > 2
%                 dKdt.log_sigma{a}(:,:,n) =  const .* tmp * l(n);
%             end
%         end
        for n = 1:N
            tmp = zeros(N,N);
            for i = 1:N % TODO: why is the full loop version faster than partly vectorized?
                for j = 1:N
                    if (i==n) || (j==n)
                        XX = l(i); YY = l(j);
                        tmp(i,j) = -YY.*(XX.^4 - YY.^4 -4*XX.^2.*D(i,j))./(sqrt(2*XX.*YY./(XX.^2+YY.^2)).*(XX.^2+YY.^2).^3).*E(i,j);
                        tmp(j,i) = tmp(i,j);
                    end
                end
            end
            temp_funs{n} = @(R) sum(R(:).*const(:).*tmp(:)) * (l(n));
            if nargout > 2
                    dKdt.log_sigma{a}(:,:,n) =  const .* tmp * l(n);
            end
        end
        dK.log_sigma{a} = @(R) cellfun(@(f) f(R), temp_funs);
    end
end
if nargin < 4
    K = K + exp(hyp.log_noise)*eye(N);
end
if nargout > 1
    dK.log_noise = @(R) sum(R(:)) * exp(hyp.log_noise);
    dhyp = @(R) dirder(R,dK);
end

function dhyp = dirder(R,dK)
A = length(dK.log_w);
for field = fieldnames(dK)'
    tmp = cell(A,1);
    fs = dK.(field{1});
    if iscell(fs)
        for a = 1:A
            f = fs{a};
            tmp{a} = f(R);
        end
        dhyp.(field{1}) = tmp;
    else
        dhyp.(field{1}) = fs(R);
    end
end
