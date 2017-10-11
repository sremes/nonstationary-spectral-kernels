function [ld] = logdet(A)
E = eig(A);
ld = sum(log(E(E>0)));
%ld = 2 * sum(log(diag(chol(A))));