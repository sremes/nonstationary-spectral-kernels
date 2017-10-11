function hyp = inputdep_unwhiten_vars(hypw,hyp_kernels)
hyp = hypw;
A = length(hyp.log_w);
for a = 1:A
    hyp.log_sigma{a} = hyp_kernels.Lsigma * hypw.log_sigma{a};
    hyp.log_w{a} = hyp_kernels.Lw * hypw.log_w{a};
    hyp.log_mu{a} = hyp_kernels.Lmu * hypw.log_mu{a};
end