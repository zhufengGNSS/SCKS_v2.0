function [SCKS Xit] = spm_SCKS(SCKS)
% FORMAT SCKS  = spm_SCKS(SCKS,fig)
%__________________________________________________________________________
% Square-root Cubature Kalman Filters [2] & Square-root Rauch-Tang-Striebel
% Smoother (SCKF-SCKS [1]).
%==========================================================================
% This function performs joint estimation of the states, input and parameters
% of the model that is described as a stochastic continuous-discrete 
% state-space in terms of nonlinear blind deconvolution. The state equations
% must have the form of ordinary differential equations, where the 
% discretization is performed through local-linearization scheme [3]. 
% Additionally, the parameter noise covariance is estimated online via 
% stochastic Robbins-Monro approximation method [4], and the measurement noise 
% covariance is estimated online as well, by using combination of varitional
% Bayesian (VB) approach with nonlinear filter [5].
%__________________________________________________________________________
%
% SCKS.M  - model structure (based on DEM [6] in SPM8 toolbox)
% SCKS.Y  - response variable, output or data
%    fig  - 1 = display estimates, 0 = do not display
%__________________________________________________________________________
%
% generative model:
%--------------------------------------------------------------------------
%   M(1).f  = dx/dt = f(x,v,P)    {inline function, string or m-file}
%   M(1).g  = y(t)  = g(x,v,P)    {inline function, string or m-file}
%   
%   M(1).xP = state error covariance matrix
%   M(1).uP = input error variance
%   M(1).wP = parameter error covariance matrix
%
%   M(1).pE = prior expectation of p model-parameters
%   M(1).pC = prior covariances of p model-parameters
%   M(1).pP = prior process covariance of p model-parameters
%   M(1).ip = parameter indices
%   M(1).cb = constrain on parameters [lower, upper];
%
%   M(1).Q  = precision components on observation noise
%   M(1).V  = fixed precision (input noise)
%   M(1).W  = precision on state noise (approximated by annealing)
%
%   M(i).m  = number of inputs v(i + 1);
%   M(1).n  = number of states x(i);
%   M(1).l  = number of output v(i);
%
%   M(1).Qf      = form of measarument noise cov estimate:
%                  'auto'(=default),'min','mean'
%   M(1).E.nN    = number of SCKF-SCKS algorithm iterations
%   M(1).E.Itol  = tolerance value for SCKF-SCKS convergence 
%   M(1).E.nD    = number of integration step between observations
%   M(1).VB.N    = number of VB algorithm iterations
%   M(1).VB.Itol = tolerance value for VB convergence 
%   M(1).VB.l    = VB scaling factor;
%
%   conditional moments of model-states - q(u)
%--------------------------------------------------------------------------
%   qU.x{1}  = Conditional expectation of hidden states (backward estimate)
%   qU.x{2}  = Conditional expectation of hidden states (forward estimate)
%   qU.v{1}  = Conditional expectation of input (backward estimate)
%   qU.v{2}  = Conditional expectation of input (forward estimate)
%   qU.r{1}  = Conditional prediction of response 
%   qU.z{1}  = Conditional prediction error 
%   qU.S{1}  = Conditional covariance: cov(x) (states - backward estimate)
%   qU.S{2}  = Conditional covariance: cov(x) (states - forward estimate)
%   qU.C{1}  = Conditional covariance: cov(u) (input - backward estimate)
%   qU.C{2}  = Conditional covariance: cov(u) (input - forward estimate)
%
% conditional moments of model-parameters - q(p)
%--------------------------------------------------------------------------
%   qP.P    = Conditional expectation
%   qP.C    = Conditional covariance
%
%      F    = negative log-likelihood
%__________________________________________________________________________
% Copyright (c) Brno University of Technology (2010), 
% Martin Havlicek 05-12-2010
% 
% References:
% [1] Havlicek, M., et al. (2011) Dynamic estimation of neuronal responses 
%     from fMRI using cubature Kalman filtering, NeuroImage, In Press.
% [2] Arasaratnam, I., Haykin, S. (2009) Cubature Kalman Filters. IEEE
%     Transactions on Automatic Control 54, 1254-1269.
% [3] Jimenez, J.C. (2002) A simple algebraic expression to evaluate the
%     local linearization schemes for stochastic differential equations* 
%     1. Applied Mathematics Letters 15, 775-780.
% [4] Van der Merwe, R., 2004. Sigma-point Kalman filters for probabilistic
%     inference in dynamic state-space models. Ph.D.thesis, Oregon Graduate 
%     Institute of Science and Technology.
% [5] Sarkka, S., Hartikainen, J. (2011) Extension of VB-AKF to Estimation
%     of Full Covariance and Non-Linear Systems. In Press.
% [6] Friston, K.J., et al. (2008) DEM: a variational treatment of dynamic
%     systems. Neuroimage 41, 849-885.
%__________________________________________________________________________

% check model specification
%--------------------------------------------------------------------------
M  = SCKS.M;
M(1).f  = fcnchk(M(1).f,'x','v','P');
M(1).g  = fcnchk(M(1).g,'x','v','P');

% get integration step dt:
dt      = 1;%M(1).E.dt;    % default 1    
nD      = M(1).E.nD;    
dt      = dt/(nD);        % integration step
TR      = M(1).E.TR;
% INITIALISATION:
% =========================================================================
% interpolate observation according to integration step
%--------------------------------------------------------------------------
y     = SCKS.Y;            % observations
if size(y,1)>size(y,2)     % check the dimensions
    y = y';
end
% intrepolate if dt < 1:
y    = interp1([1:size(y,2)],y',[1:1/nD:size(y,2)],'linear')';
if size(y,1)>size(y,2)       % check dimensions again
    y = y';
end
T    = size(y,2);            % number of time points 

% initial condition:
%--------------------------------------------------------------------------
x       = M(1).x;            % states

N       = size(x,1);         % number of nodes
u       = M(2).v;            % input
pE      = spm_vec(M(1).pE);  % model parameter
ip      = M(1).ip;           % parameter indices (must be specified or empty [] to avoid parameter estimation)


try cb  = M(1).cb;               catch, cb = []; end; % parameter constrains  
try tE  = spm_vec(SCKS.pP.P{1}); catch, tE = []; end; % true paramters for display (if available)
                            
% covariances (square-roots)
%--------------------------------------------------------------------------
sR      = cell(1,T);
[sR{:}] = deal(sparse(((M(1).V))));   % observation noise variance
               
% process error covariances (square-roots)
%--------------------------------------------------------------------------
Sx      = sparse(real(spm_sqrtm(M(1).xP)));
sQ      = sparse(real(spm_sqrtm(inv(M(1).W))));         % hidden state noise variance

if trace(M(1).uP)~=0
    Su  = sparse(real(spm_sqrt(M(1).uP)));
    sV  = sparse(real(spm_sqrtm(inv(M(2).V))));       % input noise variance
else
    Su  = [];
    sV  = [];
    u   = [];
end

if ~isempty(ip)
    qp.u0   = spm_svd(M(1).pC,exp(-32));           % basis for parameters
    M(1).p  = size(qp.u0,2);                       % number of qp.p
    theta   = sparse(M(1).p,1);                    % initial deviates
    Sw      = sparse(real(spm_sqrtm(qp.u0'*M(1).wP*qp.u0)));
    PB      = (qp.u0'*M(1).pC*qp.u0);              % prior covariance
    sW      = cell(1,T-1);
    [sW{:}] = deal(sparse(real(spm_sqrtm(qp.u0'*M(1).pP*qp.u0))));   % parameter noise variance
    dv      = diag(sW{1});
else
    Sw = [];
    sW = [];
end

ap1 = [1e2 1e8];
ap2 = [5e2 1e8];
% number of states, inputs and parameters:
%--------------------------------------------------------------------------
nx      = size(sQ,1);             % number of states
nu      = size(sV,1);             % number of states
nw      = size(sW{1},1);          % number of paramters
no      = size(sR{1},1);          % number of observations

% concatenate state vector and square-root error covariance:
%--------------------------------------------------------------------------
xc      = [x(:); u(:); theta(:)];
xx      = zeros(nx+nu+nw,T);
xx(:,1) = xc;
Sc      = cell(1,T);
[Sc{:}] = deal(sparse(nx+nu+nw,nx+nu+nw));
Sc{1}   = blkdiag(Sx,Su,Sw);
 
% get vector indices for components of concatenated state vector
xmask   = [ones(1,nx),ones(1,nu)*2,ones(1,nw)*3];
xind    = find(xmask==1);
uind    = find(xmask==2);
wind    = find(xmask==3);

xind2   = spm_vec((repmat([1:nx/N:length(xind)],nx/N,1) + repmat([0:1:nx/N-1]',1,N))');
xind1   = spm_vec((repmat([1:N:length(xind)],N,1) + repmat([0:1:N-1]',1,nx/N))');
clear xmask;

% Precalculate cubature points: 
%--------------------------------------------------------------------------
n          = nx + nu + nw;                % total state vector dimension
nPts       = 2*n;                         % number of cubature points
CubPtArray = sqrt(n)*[eye(n) -eye(n)];    % cubature points array

% setting for VB: observation noise estimation:
if ~isempty(M(1).Q)
    try iter  = M(1).VB.N;    catch, iter = 3;          end
    try lambda = M(1).VB.l;   catch, lambda = 1-exp(-2); end
    beta    = eye(no);
    alpha   = 1;
    [sR{:}] = deal(sqrt(beta./alpha));
else
    iter  = 1;
    RR    = repmat(diag(sR{1}),1,T-1);
end

% augment paramter matrix by number of cubature points:
pE0  = sparse(pE(:,ones(1,nPts)));
qp.u = kron(speye(nPts),qp.u0);

% prepare matrix template for integration by Local linearization scheme:
%--------------------------------------------------------------------------
r      = [1:12];
EXstep = (N*nPts)./r;
EXstep = r(mod(EXstep,1)==0);
EXstep = EXstep(end);
EXPm   = sparse([ones(nx/N),2*ones(nx/N,1);zeros(1,nx/N+1)]);
EXPm   = kron(speye(EXstep),EXPm);
xt     = repmat([zeros(1,nx/N) 1]',EXstep,1);

OnesNpts = ones(1,nPts);
XiR      = zeros(nx/N,nPts*N);
dx       = zeros(nx,nPts);
dx0      = zeros(nx,1);
xPred    = zeros(n,nPts);
yPred    = zeros(N,nPts);
Xf       = cell(1,T-1);
[Xf{:}]  = deal(sparse(nx+nu+nw,nPts));
x1f      = zeros(nx+nu+nw,T-1);
Jm       = zeros(((nx/N)^2)*N,nPts);
% Initialize display:
%--------------------------------------------------------------------------
try M(1).nograph; catch, M(1).nograph = 0; end
if ~M(1).nograph
    f1 = spm_figure('Create','Graphics','SCKF-SCKS estimates');
    movegui(f1,'northeast'); drawnow;
    f2 = spm_figure('Create','Graphics','SCKF-SCKS estimates');
    movegui(f2,'northwest'); drawnow;
end

% ==================================================================
% Main iteration scheme:
% ==================================================================
% get maximum number of iterations and tolerance:
try  Itol   = M(1).E.Itol;   catch,  Itol   = 1e-3;      end
try  RUN    = M(1).E.nN;     catch,  RUN    = 60;        end

MLdiff0  = 1e-4;
mloglik0 = 0;
ML       = [];
VBrun    = RUN; 
EXEC     = 0;
t0  = tic;


UU = [];
ww = [];
qq = [];
lastwarn('');

% =========================================================================
% Iteration loop (until convergence)
% =========================================================================

dq2 = diag(sQ);
sQ0 = sQ; 

RRR = [];


tt  =1;
for run = 1:RUN
     t1 = tic;
     mloglik    = 0;
     SSSb = [];
     SS = [];
    % ==================================================================
    %   Forward pass:
    % ==================================================================
    for t = 2:T,
       
        % in the case of VB observation noise estimation:
        %---------------------------------------------------------
        % Dynamic Inverse Gamma distribution model:
        if ~isempty(M(1).Q)
            if nD>1
                alpha     = lambda.*alpha + 1/sqrt(nD) + 0.05;
            elseif (TR==1 && nD==1)
                alpha     = lambda.*alpha + 1/sqrt(1.5);
            elseif (TR>1 && nD==1)
                alpha     = lambda.*alpha + 1/sqrt(1.2);
            elseif (TR<1 && nD==1)
                alpha     = lambda.*alpha + sqrt(TR);
            elseif  (TR>3)
                alpha     = lambda.*alpha + 1/sqrt(3);
            end
            beta      = lambda.*beta;
            beta0     = beta;
        end

        
        S = Sc{t-1};
        Xi            =  xc(:,OnesNpts) + S*CubPtArray;
        %------------------------------------------------------------------
        % PREDICTION STEP:
        %------------------------------------------------------------------
        xPred(uind,:) = Xi(uind,:);   % 
        xPred(wind,:) = Xi(wind,:);   % 
        
        % parameter constrain:
        if ~isempty(cb) && ~isempty(ip)
            xPred(wind,:) = min(cb(:,2*OnesNpts),xPred(wind,:)); % upper constrain
            xPred(wind,:) = max(cb(:,1*OnesNpts),xPred(wind,:)); % lower constrain
        end

        pE                = pE0 + (spm_unvec(qp.u*spm_vec(xPred(wind,:)),pE0));
   
        % propagation of cubature points through nonlinear function:
        %------------------------------------------------------------------

        XiR(:)        = Xi(xind1,:);
        f             = M(1).f(XiR,xPred(uind,:),pE);
 
        % integration by local-linearization scheme:
        %------------------------------------------------------------------
        dfdx           = spm_diff_all(M(1).f,XiR,xPred(uind,:),pE,1);
        [dx(:) dq]     = expmall2(dfdx,f,dt,xt,EXstep,EXPm,sQ0*sQ0',nx,Jm,xind1);
        xPred(xind,:)  = Xi(xind,:) + dx(xind2,:);
      
        % mean prediction:
        %------------------------------------------------------------------
        x1            = sum(xPred,2)/nPts;
        X0            = (xPred-x1(:,OnesNpts))/sqrt(nPts);
        Xf{t-1}       = X0;  % store for the backwards run (then no need to propaget through the nonlinear fcn)
        x1f(:,t-1)    = x1;  % store for the backwards run
        sQ0           = spm_sqrtm(dq(xind2,xind2));
        sQ(N+1:end,N+1:end) = sQ0(N+1:end,N+1:end); 
    
    
        S             = spm_qr([X0 blkdiag(sQ,sV,sW{t-1})]');
      
        Xi            = x1(:,OnesNpts) + S*CubPtArray;
        X             = (Xi-x1(:,OnesNpts))/sqrt(nPts);
        
        %------------------------------------------------------------------
        % UPDATE STEP:
        %------------------------------------------------------------------
        pE            = pE0 + spm_unvec(qp.u*spm_vec(Xi(wind,:)),pE0);
        XiR(:)        = Xi(xind1,:);
        % propagate cubature points through observation function:
        yPred(:)      = M(1).g(XiR,xPred(uind,:),pE);
        y1            = sum(yPred,2)/nPts;
        Y             = (yPred-y1(:,OnesNpts))/sqrt(nPts);

        resid         = y(:,t) - y1;         % innovations 
        RES(:,t)    = resid;
 
        for it = 1:iter,   
            % VB part - update of square-root measurement noise covarinace:
            if ~isempty(M(1).Q)
                Rtype = 'mean';
                switch(Rtype)
                    case('full')
                        sR{t-1} = spm_sqrtm(beta./alpha);
                        rr      = diag(diag(beta./alpha));
                    case('diag')
                        sR{t-1} = diag(sqrt(diag(beta./alpha)));
                        rr      = diag(diag(beta./alpha));
                    case('mean')
                        sR{t-1} = mean(sqrt(diag(beta./alpha)))*eye(no);
                        rr      = diag(diag(beta./alpha));
                end
            end
             
            SA  = spm_qr([Y sR{t-1};  X zeros(nx+nu+nw,N)]');
        
            
            Sy  = diag(diag(SA(1:no,1:no)));
            Sxy = SA(no+1:end,1:no);
            S   = SA(no+1:end,no+1:end);
            K   = Sxy/Sy;
           
            xc  = x1 + K*(resid);

            
            % VB part:
            if ~isempty(M(1).Q) 
                Xi0      = xc(:,OnesNpts) + S*CubPtArray;
                XiR(:)   = Xi0(xind1,:);
                yPred(:) = M(1).g(XiR,xPred(uind,:),pE); 
                D        = (y(:,t*OnesNpts)-yPred)/sqrt(nPts);
                beta     = beta0 + (D*D');
            end
            
        end
        
          
        if ~isempty(ip)
            d      = 1/ap1(1);
            subKG  = K(wind,:);
            try
                dv     = sqrt((1-d)*(dv.^2) + d*(diag(subKG*(subKG*(rr))')));
            catch
                dv     = sqrt((1-d)*(dv.^2) + d*(diag(subKG*(subKG*(resid*resid'))')));
            end
            sW{t}  = diag(dv);
            ap1(1) = min(ap1(1)+0.001,ap1(2));
        end
        
        d2     = 1/ap2(1);
        subKG  = K(xind,:);
        try
            dq2    = sqrt((1-d2)*(dq2.^2) + d2*(diag(subKG*(subKG*(rr))') ));
        catch
            dq2    = sqrt((1-d2)*(dq2.^2) + d2*(diag(subKG*(subKG*(resid*resid'))') ));
        end
        sQ0(N+1:end,N+1:end)   = diag((dq2(N+1:end)));
        ap2(1)                 = min(ap2(1)+0.001,ap2(2));
            
        Sc{t}    = S;
        xx(:,t)  = xc;

        
        % Maximum log-Likelihood
        %------------------------------------------------------------------
        mloglik  =  mloglik - log(2.*pi).*(no/2)- log(det(Sy*Sy'))/2 - resid'/(Sy*Sy')*resid/2; 
        %------------------------------------


        if ~isempty(M(1).Q)
            RR(:,t-1) = diag(sR{t-1});
        end
        
        % stop the loop if warning appears
        if ~isempty(lastwarn), error(lastwarn); 
          disp('ERROR')
          return;
        end
        
    end
 
    sW{1}  = sW{T};
    xxf = xx;
    Sf  = Sc;

   

    %----------------------------------------------------------------------
    % END of forward pass
    % ---------------------------------------------------------------------
    if run>4, mxw = mean(xx(wind,:),2); mxwi = mxw>(max(abs(mxw))*0.5);
        xx(wind(mxwi),end) = mxw(mxwi);
    end
    
    smoother = 1;
    if smoother
        % ==================================================================
        %   Backward pass:
        % ==================================================================
        for t = T-1:-1:1
            
            % Square-root Cubature Rauch-Tung-Striebel smoother
            %------------------------------------------------------------------
            % evaluate cubature points:
            Xi            =  xx(:,t*OnesNpts) + Sc{t}*CubPtArray;
            
            x01           = xx(:,t);
            X01           = (Xi - x01(:,OnesNpts))/sqrt(nPts);
            
            S             = spm_qr([Xf{t} blkdiag(sQ*0.9,sV,sW{T-1})]');
             
            Pxy           = X01*Xf{t}';      % cross covariance
            K             = (Pxy/S')/S;      % Kalman gain
            
            % smoothed estimate of the states (input,parameters)
            % and process error covariance:
            
            resid         = xx(:,t+1) - x1f(:,t);
            xx(:,t)       = xx(:,t) + K*resid;
            S             = spm_qr([X01 - K*Xf{t}, K*blkdiag(sQ,sV,sW{T-1}), K*Sc{t+1}]');
         
            Sc{t}         = S;
     
        end
    end
    
    xxb = xx;
    Sb  = Sc;

    figure(f2)
    clf
    for ii = 1:N
       if N>1, subplot(N,1,ii),  else subplot(3,1,ii), end
       try, l1 = plot(SCKS.pU.x{1}(ii,:),'k'); tit1 = 'neuronal TRUE'; hold on; catch, l1 = []; tit1 = []; end
       l2 = plot(xxb(ii,1:nD:end),'r'); tit2 = 'neuronal estimate'; hold on
       l3 = plot(y(ii,1:nD:end)/15,'g'); tit3 = 'BOLD signal';
       xlim([1 T/nD]); hold off;
                       
    end
    try
    legend([l1,l2,l3],{tit1,tit2,tit3});
    catch
    legend([l2,l3],{tit2,tit3});    
    end
        


    %----------------------------------------------------------------------
    % END of backward pass
    %----------------------------------------------------------------------
    str{1} = sprintf('SCKS: %i (1:%i)',run,iter);
    %----------------------------------------------------------------------
    % log-likelihood difference:
   
    MLdiff(run) = (mloglik-mloglik0);
    ML(run)     = mloglik;
    
    timed  = toc(t1);
    str{2} = sprintf('F:%.4e',ML(end));
    str{3} = sprintf('dF:%.4e',MLdiff(end));
    str{4} = sprintf('(%.2e sec)',timed);
    str{5} = sprintf(datestr(now,14));
    fprintf('%-16s%-16s%-16s%-16s%-16s\n',str{:})
    
    if (MLdiff(run)<0 && run>1)
        EXEC  = 1;
    else
        XXf = xxf;
        XXb = xxb;
        SSf = Sf;
        SSb = Sb;

        % plot estimates:
        %----------------------------------------------------------------------
        if ~M(1).nograph
            doplotting(M,XXf,XXb,SSf,SSb,ML,f1,T,wind,ip,run,RR,VBrun,tE,pE0,qp,N);
        end
    end

    %----------------------------------------------------------------------
    % stopping condition:
    if RUN>1 && (~isempty(ip) || ~isempty(M(1).Q))
        if run==2
            MLdiff0 = MLdiff(run);
        elseif run>2
            if MLdiff0<MLdiff(run),
                MLdiff0 = MLdiff(run);
            end
        end
        if ( (abs(MLdiff(run)/MLdiff0)<Itol || run==RUN) || ...
                (isempty(ip) && MLdiff(run)<Itol) || EXEC ),
            
            
            timed = toc(t0);
            if (run~=RUN || EXEC)
                fprintf('Converged (in %2.2e sec)\n',timed);
            else
                fprintf('Reached the maximum of iterations (in %2.2e sec)\n',timed);
            end
            if ~isempty(ip)
                pr(1:2:length(ip)*2,1) = ip';
                pr(2:2:length(ip)*2+1,1) = pE0(:,1) + qp.u0*mean(XXb(wind,:),2);
                fprintf('Estimated parameter %i. (mean): %4.2f\n',pr);
            end
            pE(:,1)  = pE0(:,1) + qp.u0*mean(XXb(wind,:),2);
            yy       = reshape(M(1).g(reshape(XXb(xind1,:),nx/N,N*T),XXb(uind,:),repmat(pE(:,1),1,T)),no,T);
            res      = y - yy;
            
            try SCKS = rmfield(SCKS,'qU'); end
            try SCKS = rmfield(SCKS,'qP'); end
            try SCKS = rmfield(SCKS,'qH'); end
            
            % save results into structure:
            SCKS.qU.x{2} = XXf(xind,1:nD:end);
            SCKS.qU.x{1} = XXb(xind,1:nD:end);
            SCKS.qU.v{2} = XXf(uind,1:nD:end);
            SCKS.qU.v{1} = XXb(uind,1:nD:end);
            SCKS.qU.r{1} = yy(:,1:nD:end);
            SCKS.qU.z{1} = res(:,1:nD:end);
            
            SCKS.F       = ML;
            ii = 0;
            for i = 1:nD:T
                ii = ii + 1;
                SCKS.qU.S{1}(:,ii) = diag(SSb{i}(xind,xind));
                SCKS.qU.S{2}(:,ii) = diag(SSf{i}(xind,xind));
                try
                    SCKS.qU.C{1}(:,ii) = diag(SSb{i}(uind,uind));
                    SCKS.qU.C{2}(:,ii) = diag(SSf{i}(uind,uind));
                catch
                    SCKS.qU.C{1} = [];
                    SCKS.qU.C{2} = [];
                end
                if ~isempty(ip)
                    SCKS.qP.P{1}(:,ii) = pE0(:,1) + qp.u0*(XXb(wind,i));
                    SCKS.qP.P{2}(:,ii) = pE0(:,1) + qp.u0*(XXf(wind,i));
                    SCKS.qP.C{1}{ii}   = qp.u0*(SSb{i}(wind,wind))*qp.u0';
                    SCKS.qP.C{2}{ii}   = qp.u0*(SSf{i}(wind,wind))*qp.u0';
                    SCKS.qP.W{1}{ii}   = qp.u0*sW{i}*qp.u0';
                 
                end
                if ~isempty(M(1).Q)
                    SCKS.qR           = RRR;
                end
            end
         
            return
        end
        
       
        
        mloglik0  = mloglik;
        xc        = [xx([xind,uind],1); mean(xx(wind,:),2);];
        xx(:,1)   = xc;
        Sc{1}     = Sc{end-1};
        
      
        
    else
        pE(ip,1) = mean(XXb(wind,:),2);
        yy       = M(1).g(XXb(xind,:),XXb(uind,:),pE(:,1));
        res      = y - reshape(yy,no,T);
        
        try SCKS = rmfield(SCKS,'qU'); end
        try SCKS = rmfield(SCKS,'qP'); end
        try SCKS = rmfield(SCKS,'qH'); end
        
        % save results into structure:
        SCKS.qU.x{2} = XXf(xind,:);
        SCKS.qU.x{1} = XXb(xind,:);
        SCKS.qU.v{2} = XXf(uind,:);
        SCKS.qU.v{1} = XXb(uind,:);
        SCKS.qU.r{1} = yy;
        SCKS.qU.z{1} = res;
        SCKS.qP.P{1} = XXb(wind,:);
        SCKS.qP.P{2} = XXf(wind,:);
        SCKS.F       = ML;
        
        for i = 1:T
            SCKS.qU.S{1}(:,i) = diag(SSb{i}(xind,xind));
            SCKS.qU.S{2}(:,i) = diag(SSf{i}(xind,xind));
            try
                SCKS.qU.C{1}(:,i) = diag(SSb{i}(uind,uind));
                SCKS.qU.C{2}(:,i) = diag(SSf{i}(uind,uind));
            catch
                SCKS.qU.C{1} = [];
                SCKS.qU.C{2} = [];
            end
            if ~isempty(ip)
                SCKS.qP.C{1}{i} = qp.u0*(SSb{i}(wind,wind))*qp.u0';
                SCKS.qP.C{2}{i} = qp.u0*(SSf{i}(wind,wind))*qp.u0';
            end
        end
        return
    end
end

%==========================================================================

%--------------------------------------------------------------------------
% Plot estimates at each iteration:
%--------------------------------------------------------------------------
function doplotting(M,xxf,xxb,Sf,Sb,ML,f1,T,wind,ip,run,RR,VBrun,tE,pE0,qp,N)

figure(f1);
set(f1,'RendererMode','auto','Renderer','painter');
clf(f1);
for p=1:2
    subplot(3,3,[1:3]+3*(p-1)),
    hax = gca;
    si    = spm_invNcdf(1 - 0.05);
    s     = [];
    if p == 1,
        xxfig = xxf;
        Sfig  = Sf;
        tit   = 'SCKF - forward pass';
    else
        xxfig = xxb;
        Sfig  = Sb;
        tit   = 'SCKS - backward pass';
    end
    for i = 1:T
        s = [s abs(diag(Sfig{i}))];
    end

    % conditional covariances
    %------------------------------------------------------------------
    j           = [1:size(xxfig(:,:),1)];
    ss          = si*s(j,:);
    [ill, indss] = sort(full(mean(ss,2)),'descend');

    pf = plot(1:T,xxfig,'linewidth',1.5);
    set(hax,'xlim',[1,T],'nextplot','add')
    box(hax,'on')
    for ic = 1:size(xxfig,1)
        col0 = get(pf(indss(ic)),'color');
        col  = (ones(1,3)-col0)*0.65 + col0;
        fill([(1:T) fliplr(1:T)],[(xxfig(indss(ic),:) + ss(indss(ic),:)) fliplr((xxfig(indss(ic),:) - ss(indss(ic),:)))],...
            'r',...
            'FaceColor',col,...
            'EdgeColor',col);
        hold on;
        COL{ic} = col0;
    end
    for ic = 1:size(xxfig,1)
        plot(xxfig(indss(ic),:),'color',COL{ic},'linewidth',0.75);
    end
    title(tit);
    grid(hax,'on');
    axis(hax,'tight');
end

subplot(3,3,7)
h = plot([1:length(ML)],ML);
if ~isempty(M(1).Q)
    AYlim = get(gca,'Ylim');
    if VBrun>=run
        bkg = ones(1,run)*max(AYlim);
    else
        bkg = [ones(1,VBrun)*max(AYlim),ones(1,abs(VBrun-run))*min(AYlim)];
    end
    a = area(bkg,min(AYlim));
    axis([1 run+1 AYlim(1) AYlim(2)]); hold on;
    set(a(1),'FaceColor',ones(1,3)*0.6,'EdgeColor',ones(1,3)*0.6)
    h = plot([1:length(ML)],ML);
    axis([1 run+1 AYlim(1) AYlim(2)]); hold on;
    if VBrun>=run
        text(run/2+1,mean(AYlim),'VB-SCKS','HorizontalAlignment','center','VerticalAlignment','top');
    else
        text(VBrun/2+1,mean(AYlim),'VB-SCKS','HorizontalAlignment','center','VerticalAlignment','top');
        text(VBrun+(abs(VBrun-run)/2)+1,mean(AYlim),'SCKS','HorizontalAlignment','center','VerticalAlignment','top');
    end
    set(gca,'Layer','top')
    hold off
end
set(h,'color','k','Marker','o','Markersize',4,'MarkerFaceColor','k','linewidth',1);
title('Log-Likelihood');

if ~isempty(ip)
 opt = 1;
 subplot(3,3,8)
  if opt   
   
    pE     = pE0(:,1) + spm_unvec(qp.u0*mean(xxb(wind,:),2),pE0(:,1));  
    if isempty(tE)
        b1 = bar(1:length(pE),pE');
        set(b1,'BarWidth',0.5,'FaceColor','k');
    else
        b1 = bar(1:N*N,tE(1:N*N));
        set(b1,'BarWidth',0.5,'FaceColor',ones(1,3)*0.5,'EdgeColor',ones(1,3)*0.5); hold on;
        b2 = bar(1:N*N,pE(1:N*N)');
        set(b2,'BarWidth',0.6/2,'FaceColor','k'); hold off;
    end
    title('Parameters');
    if isempty(M(1).Q)
         subplot(3,3,9)
         plot([1:T],xxb(wind,:)); 
         xlim([1 T]);
         title('Parameter Modes');  
    end   
 else
   plot([1:T],xxb(wind,:)); 
   xlim([1 T]);
   title('Parameter Modes');
 end
 

end

if ~isempty(M(1).Q)
    subplot(3,3,9)
    if VBrun>=run
        plot(RR'); hold on;
    else
        plot(RR'); hold on;
        switch(lower(M(1).Qf))
            case('all')
                sR  = RR';
            case('mean')
                sR  = repmat(mean(RR,2),1,T)';
            case('mean-all')
                sR  = ones(3,T)*(mean(mean(RR,2)));
            case('min')
                RRs = sort(RR,2,'descend');
                sR  = repmat(mean(RRs(:,round(T*0.90):end),2),1,T)';
            case('auto')
                dlim    = min(RR,[],2);
                ulim    = max(RR,[],2);
                if all(ulim./dlim<4)
                    sR  = RR';
                else
                    RRs  = sort(RR,2,'descend');
                    sR  = repmat(mean(RRs(:,round(T*0.90):end),2),1,T)';
                end
        end
        plot([1:T-1],sR,'r','linewidth',2);
    end
    axis(gca,'tight');
    title('Std(R)');
    hold off;
end
drawnow;