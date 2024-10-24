function [ bnfxp] = nfxpdygam(aobs, zobs, aobs_1, zval, ptranz, theta, v, bdisc, miniter, maxiter)

% ---------------
% Some constants
% ---------------
eulerc = 0.5772;
myzero = 1e-16;
nobs = size(aobs, 1);
nplayer = size(aobs, 2);
naction = 2;
numa = 2^nplayer;
numz = size(zval, 1);
numx = numz*numa;
kparam = nplayer + 3;
times = [];
varb = zeros(kparam, kparam);
llike = -1e300;

% ----------------------------------------------
% Matrix with values of states of (s[t],a[t-1])
% ----------------------------------------------
aval = zeros(numa, nplayer);
for i = 1:nplayer
    aval(:,i) = kron(ones(2^(i-1),1), kron([ 0; 1 ], ones(2^(nplayer-i),1)));
end
mstate = zeros(numx, nplayer+1);
mstate(:,1) = kron(zval, ones(numa, 1));
mstate(:,2:nplayer+1) = kron(ones(numz, 1), aval);

% ------------------------------------------------
% Matrix with observed indexes of state variables
% ------------------------------------------------
indzobs = (repmat(zobs, 1,numz)==repmat(zval', nobs, 1))*[ 1:numz ]';
twop = kron(ones(nobs, 1), 2.^(nplayer-[1:nplayer]));
indobs = sum(aobs_1.*twop, 2);
indobs = (indzobs-1).*(2^nplayer) + indobs + 1; % 1 to 160, for the states of the game
indobs_orig = indobs;
for i = 2:nplayer
    indobs = [ indobs; indobs_orig + (i-1)*numx ];
end

dummy_mat=(reshape(aobs,nobs,1,nplayer)==...
    reshape(0:naction-1,1,naction,1));%nobs*naction*nplayer

dummy_count=zeros(numx*nplayer,naction);

for i=1:(numx*nplayer)
    dummy_count(i,:)=sum(reshape(permute(dummy_mat,[1,3,2]),[],naction).*(indobs==i),1);
end

pchoice = zeros(numx, nplayer);
% Calculate initial choice probabilities
for i = 1:nplayer
    vdiff = v(:,i,2) - v(:,i,1);
    expvdiff = exp(vdiff);
    pchoice(:,i) = expvdiff ./ (1 + expvdiff);
end

ptranz_kron=(kron(ptranz, ones(numa,numa)));%%%

%% NFXP
global v_global pchoice_global
v_global=[]; % hot start
pchoice_global=[]; % hot start
opt_options = optimset('GradObj','off','MaxIter',100,...
            'Display','off','TolFun',1e-6,'Algorithm','quasi-newton',...
            'FinDiffType','central',MaxFunEvals = 1000);%%%%%%%

%%log_likelihood_func_NFXP(theta,dummy_count,v,zval,ptranz,bdisc,aval,mstate,ptranz_kron)

theta_init=theta;

[bnfxp,fval_NFXP,exitflag_NFXP,output_opt_NFXP]=...
            fminunc(@log_likelihood_func_NFXP,theta_init,opt_options,...
                dummy_count,v,pchoice,zval,ptranz,bdisc,aval,mstate,ptranz_kron);



end
