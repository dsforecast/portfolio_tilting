function cov_mat = tidy_cov_mat(m)

% Purpose: Returns positive definite and symmetric covariance matrix

cov_mat = tril(m, -1);
cov_mat = cov_mat + diag(diag(m)) + cov_mat';
[V,D] = eig(cov_mat);
d=diag(D);
d(d<=eps) = eps * 10;
a = V * diag(d) * V';
cov_mat = tril(a, -1);
cov_mat = cov_mat + diag(diag(a)) + cov_mat';