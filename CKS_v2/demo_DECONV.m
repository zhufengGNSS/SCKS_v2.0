% DEMO for deconvolution using SCKS algorithm
% Created by Martin Havlicek 2011, Albuquerque
%==================================================================

clear all; close all;

% 1. Here add path to spm8   !!!!!!!!

% addpath(genpath('C:\Backup\spm8\'));


% 2. Load your data:
% data (=time course) should be for example the first eigen-variate from the
% ROI. Data should be already corrected for slow fluctuations and motion.
% We further assume data having zero-mean (or even better adjusted to zero baseline in rest)

load data.mat
DCM.Y = data;
 
T   = length(DCM.Y);                             % number of time point

% 3. Here write the TR of your data:
TR  = 2;                                         % sampling period

% scale data to have particular peak to peak range e.g. 2 % signal
% change
scale = peak2peak(DCM.Y');

pk2pk = 2;   
if scale > pk2pk
    scale   = pk2pk/max(scale,pk2pk);
    DCM.Y   = DCM.Y*scale;
else
    scale   = pk2pk/min(scale,pk2pk);
    DCM.Y   = DCM.Y*scale;
end

% Specify generative model for inversion (DCM)
% =========================================================================
% set inversion parameters
% -------------------------------------------------------------------------

DCM.M(2).v    = 0;
DCM.M(1).E.TR = TR; 
DCM.M(1).E.dt = 1; 
n             = 1;
% Specify hyper-priors on precisions
% -------------------------------------------------------------------------
W             = exp(spm_vec(sparse(1:n,1,(6 - 16),n,5) + 16));
DCM.M(1).xP   = exp(6);
DCM.M(1).V    = 1;             % prior log precision (noise)
DCM.M(1).W    = diag(W);       % fixed precision (hidden-state)
DCM.M(2).V    = exp(8);        % fixed precision (hidden-cause) (not used)


% Full connectivity inversion
%==========================================================================
F  = ones(n,n);
B  = zeros(n,n,0);
C  = zeros(n,1);
D  = zeros(n,n,0);
options.endogenous = 0;  
[pE pC ill pW]     = spm_dcm_fmri_priorsM(F,B,C,D,options);   % generating parameter priors for model inversion
                                                              % 
                                                              
DCM.M(1).pE = pE;
DCM.M(1).pC = pC;
DCM.M(1).cb = [];
DCM.M(1).Nc = n^2;

DCM.M(1).W  = diag(W)/TR*5;        %!!! precision on neuronal (and hemodynamic) state noise - IMPORTANT
                                   % might need to be adjustment to your (it defines smoothness of the estimate)
                                  
DCM.M(2).V  = eye(n);
DCM.M(1).V  = eye(n)*DCM.M(1).V;
DCM.M(2).v  = 0;
DCM.M(1).x  = zeros(n,5);

% defining paramters (their covariances) for estimation:
nA   = ones(n);                                  % full connectivity matrix
nA   = diag(nA(:));
hP   = diag(repmat([0 1 0 1 0]',n,1));           % select hemodynamic paremeters for estimation;
DCM.M(1).pC = pC.*blkdiag(nA,eye(n),hP,1);       % prior covariance on parameters

DCM.M(1).pE = pE;
DCM.M(1).pP = pW*1e-6/3;  % 

DCM.M(1).wP = pC*1e-2;
DCM.M(1).uP = 0; 
DCM.M(1).xP = blkdiag(eye(n)*1e-1^2,eye(4*n)*1e-1^2)/3; %

pE          = spm_vec(pE);
ip          = [1:length(pE)]; 

DCM.M(1).ip = ip;
DCM.M(1).l  = n;
DCM.M(1).Q  = {speye(DCM.M(1).l,DCM.M(1).l)};   % if Q is specified then algorithm performs
                                                % estimation of measurement noise covariance 
%DCM.M(1).Q  = [];                              % if presion on measurement noise
                                                % is known then Q = [];
DCM.M(1).E.nN    = 20;                   % max number of iteration of SCKF-SCKS algorithm
DCM.M(1).E.Itol  = 1e-5;                 % convergence tolerance value for SCKF_SCKS algorithm
DCM.M(1).E.nD    = 5;                   %numeber of integration steps.!!! set to match the TR but it can be also higher number (but it might need other adjustments)
DCM.M(1).E.RM    = [1e2 1e8];
DCM.M(1).VB.N    = 5;                    % number of VB iteration during one SCKF-SCKS run
DCM.M(1).VB.l    = 1 - exp(-6);          % scaling parameter for VB algorithm, 
                                         % controls dynamics

fx = ['@(x,v,P) spm_fx_fmri4(x,v,P)*',num2str(TR)]; % functions of hemodynamic model for model inversion by SCKS 
gx = ['@(x,v,P) spm_gx_fmri4(x,v,P)'];

DCM.M(1).f    = eval(fx);
DCM.M(1).g    = eval(gx);
DCM.pU        = [];
DCM.pU.x{1}   = [];

%==========================================================================
% Do model inverison
%--------------------------------------------------------------------------
FULL = spm_SCKS_sDCM(DCM);

%==================================================================
%  Plot results for single time course estimates
%==================================================================
% Display State results:
f5 = spm_figure('Create','Graphics','CKF estimates');
set(f5,'RendererMode','auto','Renderer','painter');
clf(f5);
for p = 1:3
    subplot(3,1,p),
    hax = gca;
    si    = spm_invNcdf(1 - 0.05);

    if p == 1,
        xxfig = FULL.qU.r{1}(1:n,1:end);
        Sfig  = zeros(n,size(xxfig,2));
        tit   = 'BOLD prediction';
    elseif p == 2
        xxfig = FULL.qU.x{1}(1:n,1:end);
        Sfig  = FULL.qU.S{1}(1:n,1:end);
        tit   = 'Neural Estimate';
    else
        xxfig = FULL.qU.x{1}(n+1:end,1:end);
        Sfig  = FULL.qU.S{1}(n+1:end,1:end);
        tit   = 'States Estimate';
    end

    s = abs(Sfig);

    % conditional covariances
    %------------------------------------------------------------------

    j       = [1:size(xxfig,1)];
    ss      = si*s(j,:);
    s(j,:)  = [];
    [ill indss] = sort(mean(ss,2),'descend');
    pf = plot(1:T,xxfig,'linewidth',1.5);
    set(hax,'xlim',[1,T],'nextplot','add')
    for ic = 1:size(xxfig,1)
        col0 = get(pf(indss(ic)),'color');
        col = (ones(1,3)-col0)*0.65 + col0;
        fill([(1:T) fliplr(1:T)],[(xxfig(indss(ic),:) + ss(indss(ic),:)) fliplr((xxfig(indss(ic),:) - ss(indss(ic),:)))],...
            'r',...
            'FaceColor',col,...
            'EdgeColor',col);

        hold on
        COL{ic} = col0;
    end
    for ic = 1:size(xxfig,1)
        plot(xxfig(indss(ic),:),'color',COL{ic},'linewidth',0.75);
    end
    try
     if p == 1   
        plot(FULL.Y','Color',[0 0 0],'linewidth',1);
        try
          plot(SIM.pU.r{1}(1:n,:)' - SIM.pU.z{1}(1:n,:)' ,'Color','r','linewidth',1);    
        catch
        end
     elseif p == 2
        plot(SIM.pU.x{1}(1:n,:)','Color',[0 0 0],'linewidth',1);   
     else
        plot(SIM.pU.x{1}(n+1:end,:)','Color',[0 0 0],'linewidth',1);    
     end
    catch
    end
    title(tit);
    grid(hax,'on')
    axis(hax,'tight')
    set(hax,'box','on','Layer','top');
    set(hax,'tickdir','out')
   
end
    