function [dx dq] = expmall2(J,f,t,xt,K,EP0,Q,n,Jm,xind)

EP   = EP0;
dx   = [];

for i=1:size(f,2)/K,
    EP(EP0==1) = J(:,(i-1)*K+1:i*K);
    EP(EP0==2) = f(:,(i-1)*K+1:i*K);
    dx0        = expm(EP*t)*xt;
    dx         = [dx, dx0(~xt)];
end

dx = reshape(dx,size(f,1),size(f,2));

Jm(:) = J;
JJ = kron(speye(n/(size(Jm,1)/n)),ones(size(Jm,1)/n));
Jm = mean(Jm,2);
JJ(JJ==1) = Jm; 

C  = expm([-JJ,Q(xind,xind);zeros(n),JJ']*t);
G  = C(1:n,n+1:end);
F  = C(n+1:end,n+1:end);
dq = diag(diag(F'*G));