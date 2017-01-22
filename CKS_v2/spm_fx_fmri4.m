function [y] = spm_fx_fmri3(x,u,P,N)
% state equation for a dynamic [bilinear/nonlinear/Balloon] model of fMRI
% responses
% FORMAT [y] = spm_fx_fmri(x,u,P,M)

% x      - state vector
%   x(:,1) - excitatory neuronal activity             ue
%   x(:,2) - vascular signal                          s
%   x(:,3) - rCBF                                  ln(f)
%   x(:,4) - venous volume                         ln(v)
%   x(:,5) - deoyxHb                               ln(q)
%  [x(:,6) - inhibitory neuronal activity             ui]
%
% y      - dx/dt
%
%___________________________________________________________________________
%
% References for hemodynamic & neuronal state equations:
% 1. Buxton RB, Wong EC & Frank LR. Dynamics of blood flow and oxygenation
%    changes during brain activation: The Balloon model. MRM 39:855-864,
%    1998.
% 2. Friston KJ, Mechelli A, Turner R, Price CJ. Nonlinear responses in
%    fMRI: the Balloon model, Volterra kernels, and other hemodynamics.
%    Neuroimage 12:466-477, 2000.
% 3. Stephan KE, Kasper L, Harrison LM, Daunizeau J, den Ouden HE,
%    Breakspear M, Friston KJ. Nonlinear dynamic causal models for fMRI.
%    Neuroimage 42:649-662, 2008.
% 4. Marreiros AC, Kiebel SJ, Friston KJ. Dynamic causal modelling for
%    fMRI: a two-state model.
%    Neuroimage. 2008 Jan 1;39(1):269-78.
%__________________________________________________________________________

% Copyright (C) 2008 Wellcome Trust Centre for Neuroimaging

% Karl Friston & Klaas Enno Stephan
% $Id: spm_fx_fmri.m 3888 2010-05-15 18:49:56Z karl $

N = size(x,2)/size(P,2);

% number of regular states

% implement differential state equation y = dx/dt (neuronal)
%--------------------------------------------------------------------------

X       = x;
A       = kron(speye(size(P,2)),ones(N));
A(A==1) = spm_vec(P(1:N*N,:));
C       = spm_vec(P(N*N+1:N*N+N,:));
transit = spm_vec(P((N*N+1*N+1):(N*N+2*N),:))';
decay   = spm_vec(P((N*N+2*N+1):(N*N+3*N),:))';
areg    = spm_vec(P((N*N+3*N+1):(N*N+4*N),:))';
extfr   = spm_vec(P((N*N+4*N+1):(N*N+5*N),:))';
alpha   = spm_vec(P((N*N+5*N+1):(N*N+6*N),:))';
%----------------------------------------------------------------------
if isempty(u)
    y(1,:)  = (A*X(1,:)')';  
   % y(1,y(1,:)<0) = 0;
else
    y(1,:)  = (A*X(1,:)' + C*u(:))';
end

% Hemodynamic motion
%==========================================================================

% hemodynamic parameters
%--------------------------------------------------------------------------
%   H(1) - signal decay                                   d(ds/dt)/ds)
%   H(2) - autoregulation                                 d(ds/dt)/df)
%   H(3) - transit time                                   (t0)
%   H(4) - exponent for Fout(v)                           (alpha)
%   H(5) - resting oxygen extraction                      (E0)
%   H(6) - ratio of intra- to extra-vascular components   (epsilon)
%          of the gradient echo signal
%--------------------------------------------------------------------------
% H        = [0.65 0.41 2.00 0.32 0.34];
H        = [0.64 0.32 2.0 0.32 0.32];
%
%
% % exponentiation of hemodynamic state variables
% %--------------------------------------------------------------------------

X(3:5,:) = exp(X(3:5,:));

% % signal decay
% %--------------------------------------------------------------------------
sd       = H(1).*exp(decay);

% % autoregulation
% %--------------------------------------------------------------------------
ar       = H(2).*exp(areg);     

% % transit time
% %--------------------------------------------------------------------------
tt       = H(3).*exp(transit);

% % alpha
% %------------------------------------------------------------------------
al       = H(4).*exp(alpha);

% % E0
% %------------------------------------------------------------------------
ef       = H(5).*exp(extfr);


% % Fout = f(v) - outflow
% %--------------------------------------------------------------------------
fv       = X(4,:).^(1./al);


% % e = f(f) - oxygen extraction
% %--------------------------------------------------------------------------
ff       = (1 - (1 - ef).^(1./X(3,:)))./ef;

%
% % implement differential state equation y = dx/dt (hemodynamic)
% %--------------------------------------------------------------------------
y(2,:)   = X(1,:) - sd.*X(2,:) - ar.*(X(3,:) - 1);
y(3,:)   = X(2,:)./X(3,:);
y(4,:)   = (X(3,:) - fv)./(tt.*X(4,:));
y(5,:)   = (ff.*X(3,:) - fv.*X(5,:)./X(4,:))./(tt.*X(5,:));

