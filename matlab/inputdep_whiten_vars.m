function hypw = inputdep_whiten_vars(hyp,hyp_kernels)
hypw = hyp;
A = length(hyp.log_w);
for a = 1:A
    hypw.log_sigma{a} = hyp_kernels.Lsigma \ hyp.log_sigma{a};
    hypw.log_w{a} = hyp_kernels.Lw \ hyp.log_w{a};
    hypw.log_mu{a} = hyp_kernels.Lmu \ hyp.log_mu{a};
end