% ----------------------------------------------------------------------------
% NPLDYGAM       Procedure that estimates the structural parameters
%                of dynamic game of firms' entry/exit
%                using a Nested Pseudo-Likelihood (NPL) algorithm
%
% Based on the NPL Gauss code by Victor Aguirregabiria.
% Converted from Gauss to Matlab by Jason Blevins.
% Modified to include CCPs and (un-normalized) choice-specific value
% functions from 1-step NPL along with return values.
%
% ----------------------------------------------------------------------------
%
% FORMAT:
% [bnpl,varb,pchoice,v1npl,vnpl,iter,err,times] = npldygam(aobs,zobs,aobs_1,zval,ptranz,pchoice,bdisc,miniter,maxiter)
%
% INPUTS:
%     aobs    - (nobs x nplayer) matrix with observations of
%               firms' activity decisions (1=active ; 0=no active)
%     zobs    - (nobs x 1) vector with observations of market
%               exogenous characteristics.
%     aobs_1  - (nobs x nplayer) matrix with observations of
%               firms' initial states (1=incumbent; 0=potential entrant)
%     zval    - (numz x 1) vector with values of market characteristics
%     ptranz  - (numz x numz) matrix of transition probabilities
%               of market characteristics.
%     pchoice - ((numz*2^nplayer) x nplayer) matrix of players'
%               choice probabilities used to initialize the procedure.
%     bdisc   - Discount factor
%     miniter - Minimum number of NPL iterations
%     maxiter - Maximum number of NPL iterations
%
% OUTPUTS:
%     bnpl    - (kparam x niter) matrix with NPL estimates as rows:
%               (theta_fc_1; theta_fc_2; ... ; theta_fc_n; theta_rs; theta_rn; theta_ec).
%     varb    - (kparam x kparam) variance matrix from clogit estimation.
%     pchoice - ((numz*2^nplayer) x nplayer) matrix of players'
%               choice probabilities consistent with estimates.
%     v1npl   - (nplayer x 2 x (numz*2^nplayer)) matrix of players'
%               choice-specific value functions consistent with estimates
%               from the first iteration of NPL.
%     vnpl    - (nplayer x 2 x (numz*2^nplayer)) matrix of players'
%               choice-specific value functions consistent with the
%               converged NPL estimates.
%     iter    - Number of NPL iterations completed.
%     err     - Error flag indicator
%     times   - Vector of computational time (sec.) per iteration
%
% ----------------------------------------------------------------------------

function [ bnpl, varb, pchoice, v1npl, vnpl, llike, iter, err, times ] = ...
    npldygam1(aobs, zobs, aobs_1, zval, ptranz, pchoice, bdisc, miniter, maxiter)
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
  critconv = (1e-2)*(1/kparam);
  bcnpl = zeros(kparam, 1);
  bnpl = [];
  times = [];
  varb = zeros(kparam, kparam);
  v1npl = zeros(numx, nplayer, naction);
  vnpl = zeros(numx, nplayer, naction);
  llike = -1e300;


    global scale_ll output_opt
    
opt_options = optimset('GradObj','on','MaxIter',100,...
            'Display','off','TolFun',1e-6,'Algorithm','quasi-newton',...
            'FinDiffType','central','TolX',1e-6,...
            MaxFunEvals = 100);%%%%%%%

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
  indobs = (indzobs-1).*(2^nplayer) + indobs + 1;
  indobs_orig=indobs;

  indobs_rep=indobs;
  for i = 2:nplayer
    indobs_rep = [ indobs_rep; indobs_orig + (i-1)*numx ];
  end

  dummy_mat=(reshape(aobs,nobs,1,nplayer)==...
    reshape(0:naction-1,1,naction,1));%nobs*naction*nplayer

  dummy_count=zeros(numx*nplayer,naction);

  for i=1:(numx*nplayer)
    dummy_count(i,:)=sum(reshape(permute(dummy_mat,[1,3,2]),[],naction).*(indobs_rep==i),1);
  end

  
  % -------------
  % NPL algorithm
  % -------------
  aobs = 1 + reshape(aobs, [nobs*nplayer,1]);
  v = zeros(numx,nplayer,naction);

  
  ptranz_kron=(kron(ptranz, ones(numa,numa)));%%%

  % Iterate until convergence criterion met
  criter = 1000;
  iter = 1;
  while ((criter > critconv) && (iter <= maxiter))
    %fprintf('\n');
    fprintf('-----------------------------------------------------\n');
    fprintf('NPL ESTIMATOR: K = %d\n', iter);
    fprintf('-----------------------------------------------------\n');
    fprintf('\n');
    tic;


    out_e_u=...
        compute_e_u_from_pchoice_func(pchoice,bcnpl,zval, ptranz,  bdisc,...
        aval,mstate,ptranz_kron);

    e0=out_e_u{1};
    e1=out_e_u{2};
    u0=out_e_u{3};
    u1=out_e_u{4};

    %%% eobs0,eobs1,uobs0,uobs1 %%%%
    % ------------------------------------------
    %  (d) Pseudo Maximum Likelihood Estimation
    % ------------------------------------------
    %tic
    %%[ thetaest, varb, llike, cl_iter, err ] = clogit(aobs, [uobs0, uobs1], [eobs0, eobs1], 100, bcnpl);
    [ thetaest, varb, llike, cl_iter, err ] = clogit0(dummy_count,[u0,u1],[e0,e1],...
        100, bcnpl);

    %toc

    %if err > 0
    %    return;
    %end

   %tic

   %[thetaest2,llike_negative,exitflag,output_fminunc]=...
   %     fminunc(@log_likelihood_func0,bcnpl,opt_options,...
   % dummy_count,out_e_u{1:4});
   
   %llike2=(-1)*llike_negative*scale_ll;
   %toc
   %thetaest=thetaest2;

   %err=0;%%%%%%
    %%%%%%varb


    % ----------------------------
    %  (e) Updating probabilities
    % ----------------------------
    for i = 1:nplayer
      v(:,i,2) = u1((i-1)*numx+1:i*numx,:)*thetaest + e1((i-1)*numx+1:i*numx,:);
      v(:,i,1) = u0((i-1)*numx+1:i*numx,:)*thetaest + e0((i-1)*numx+1:i*numx,:);
      buff = v(:,i,2) - v(:,i,1);
      pchoice_est(:,i) = exp(buff)./(1+exp(buff));
    end

    % Check for convergence after minimum number of iterations
    if (iter > miniter)
        criter1 = max(abs(thetaest - bcnpl));
        criter2 = max(max(abs(pchoice_est - pchoice)));
        criter = max(criter1, criter2);
    end

    % Save values for next iteration
    bcnpl = thetaest;
    vnpl = v;
    pchoice = pchoice_est;

    % Collect estimates
    bnpl = [ bnpl; thetaest' ];

    % Collect times
    times = [ times; toc ];

    % Separately save value function from first iteration
    if iter == 1
        v1npl = v;
    end

    disp('theta = ')
    disp(bcnpl')
    disp('llike = ')
    disp(llike)

    % Proceed to the next iteration
    iter = iter + 1;
  end

  % Undo final iteration increment
  iter = iter - 1;
end
