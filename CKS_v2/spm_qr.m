function R = spm_qr(X)


[Q R] = qr(X,0);

R     = real(R);
D     = diag(sign(diag(R)));
R     = D*R;
R     = R';
