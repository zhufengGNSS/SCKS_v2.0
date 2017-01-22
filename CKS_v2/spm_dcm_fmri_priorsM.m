function [pE,pC,x,pW] = spm_dcm_fmri_priorsM(A,B,C,D,options)
% Returns the priors for a two-state DCM for fMRI.
% FORMAT:[pE,pC,x] = spm_dcm_fmri_priors(A,B,C,D,options)
%
%   options.two_state:  (0 or 1) one or two states per region
%   options.endogenous: (0 or 1) exogenous or endogenous fluctuations
%
% INPUT:
%    A,B,C,D - constraints on connections (1 - present, 0 - absent)
%
% OUTPUT:
%    pE     - prior expectations (connections and hemodynamic)
%    pC     - prior covariances  (connections and hemodynamic)
%    x      - prior (initial) states
%__________________________________________________________________________
%
% References for state equations:
% 1. Marreiros AC, Kiebel SJ, Friston KJ. Dynamic causal modelling for
%    fMRI: a two-state model.
%    Neuroimage. 2008 Jan 1;39(1):269-78.
%
% 2. Stephan KE, Kasper L, Harrison LM, Daunizeau J, den Ouden HE,
%    Breakspear M, Friston KJ. Nonlinear dynamic causal models for fMRI.
%    Neuroimage 42:649-662, 2008.
%__________________________________________________________________________
% Copyright (C) 2008 Wellcome Trust Centre for Neuroimaging
 
% Karl Friston
% $Id: spm_dcm_fmri_priors.m 4112 2010-11-05 16:12:21Z karl $
% modified by Martin 
% number of regions
%--------------------------------------------------------------------------
n = length(A);
if nargin < 5
    options = struct( 'two_state',0, 'endogenous',0);
end
   
 
% check options and D (for nonlinear coupling)
%--------------------------------------------------------------------------
try , options.two_state;  catch, options.two_state  = 0; end
try, options.endogenous;  catch, options.endogenous = 0; end
try, D;                   catch, D = zeros(n,n,0);       end
 
 
% prior (initial) states and shrinkage priors on A for endogenous DCMs
%--------------------------------------------------------------------------
if options.two_state,  x = sparse(n,6); else, x = sparse(n,5); end
if options.endogenous, a = 128;         else, a = 8;           end
 
 
% connectivity priors
%==========================================================================
if options.two_state
   
    % enforce optimisation of intrinsic (I to E) connections
    %----------------------------------------------------------------------
    A     = (A + eye(n,n)) > 0;
 
    % prior expectations and variances
    %----------------------------------------------------------------------
    pE.A  =  A*32 - 32;
    pE.B  =  B*0;
    pE.C  =  C*0;
    pE.D  =  D*0;
 
    % prior covariances
    %----------------------------------------------------------------------
    pC.A  =  A/4;
    pC.B  =  B/4;
    pC.C  =  C*4;
    pC.D  =  D/4;
 
else
 
    % enforce self-inhibition
    %----------------------------------------------------------------------
    A     =  A > 0;
    A     =  A - diag(diag(A));
 
    % prior expectations
    %----------------------------------------------------------------------
    pE.A  =  A/(64*n) - eye(n,n)/2;
    pE.B  =  B*0;
    pE.C  =  C*0;
    pE.D  =  D*0;
   
    % prior covariances
    %----------------------------------------------------------------------
    pC.A  =  A*a/4 + eye(n,n)/(64); % 8*4
    pC.B  =  B;
    pC.C  =  C;
    pC.D  =  D;
    
       
    pW.A  =  A*a/4 + eye(n,n)/(8*4);
    pW.B  =  B;
    pW.C  =  C;
    pW.D  =  D;
    
    
        
end
 
% and add hemodynamic priors
%==========================================================================
pE.transit = sparse(n,1);
pE.decay   = sparse(n,1);
pE.areg    = sparse(n,1);
pE.extfr   = sparse(n,1);
pE.alpha   = sparse(n,1);
pE.epsilon = sparse(1,1);
 
pC.transit = sparse(n,1) + exp(-2);           % just to know the order...
pC.decay   = sparse(n,1) + exp(-2.001);
pC.areg    = sparse(n,1) + exp(-2.003);
pC.extfr   = sparse(n,1) + exp(-2.005);
pC.alpha   = sparse(n,1) + exp(-2.007);
pC.epsilon = sparse(1,1) + exp(-5.012);

pW.transit = sparse(n,1) + exp(-4);
pW.decay   = sparse(n,1) + exp(-4);
pW.areg    = sparse(n,1) + exp(-4);
pW.extfr   = sparse(n,1) + exp(-4);
pW.alpha   = sparse(n,1) + exp(-4);
pW.epsilon = sparse(1,1) + exp(-6);

pC          = diag(spm_vec(pC));
pW          = diag(spm_vec(pW)); 
return
