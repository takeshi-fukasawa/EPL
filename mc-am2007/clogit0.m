% -------------------------------------------------------------------------------------
% CLOGIT -    Maximum Likelihood estimation of McFadden's Conditional Logit
%
%                  Optimization algorithm: Newton's method with analytical
%                  gradient and hessian
%
% by Victor Aguirregabiria
%
% Converted from Gauss to Matlab by Jason Blevins.
%
% -------------------------------------------------------------------------------------
%
% Format      [ best, varest, iter, err ] = clogit(ydum, x, restx, maxiter, xinit)
%
% Input        ydum    - (nobs x 1) vector of observations of dependet variable
%                        Categorical variable with values: {1, 2, ..., nalt}
%
%              x       - (nobs x (k * nalt)) matrix of explanatory variables
%                        associated with unrestricted parameters.
%                        First k columns correspond to alternative 1, and so on
%
%              restx   - (nobs x nalt) vector of the sum of the explanatory
%                        variables whose parameters are restricted to be
%                        equal to 1.
%
%  Output      best    - (k x 1) vector with ML estimates.
%
%              varest  - (k x k) matrix with estimate of covariance matrix
%
% -------------------------------------------------------------------------------------

function [ b0, Avarb, llike, iter, err ] = clogit0(...
    dummy_count,x,restx,...
    maxiter, xinit)
    
  cconvb = 1e-6;
  myzero = 1e-16;
  nobs = sum(dummy_count(:), 1);
  nalt = size(dummy_count,2);
  npar = size(x, 2) / nalt;
  err = 0;
  lsopts.SYM = true;
  lsopts.POSDEF = true;


  % x: (n_state*nPlayer)*(npar*nalt)
  % 

  xysum=sum(reshape(x,[],npar,nalt).*reshape(dummy_count,[],1,nalt),[1,3]);%1*npar
  dummy_count_sum=sum(dummy_count,2);

  iter = 1;
  criter = 1000;
  llike = 0;
  b0 = xinit;
  while (criter > cconvb && iter <= maxiter)
    % fprintf('\n');
    % fprintf('Iteration                = %d\n', iter);
    % fprintf('Log-Likelihood function  = %12.4f\n', llike);
    % fprintf('Norm of b(k)-b(k-1)      = %12.4f\n', criter);
    % fprintf('\n');

    % Computing probabilities

    phat=reshape(sum(reshape(x,[],npar,nalt).*reshape(b0,1,npar,1),2),[],nalt)+ restx;
    phatmax = max(phat, [], 2);
    phat = phat - phatmax;
    phat = exp(phat) ./ sum(exp(phat), 2);%[]*nalt


    llike=sum(log(phat).*dummy_count,[1,2]);


    %xxm=sum(reshape(phat,[],1,1,nalt).*reshape(x,[],npar,1,nalt).*...
    %    reshape(x,[],1,npar,nalt).*reshape(dummy_count_sum,[],1,1,1),[1,4]);
    %xxm=reshape(xxm,npar,npar);


    % Computing gradient

    mean_x_temp=sum(reshape(phat,[],1,nalt).*reshape(x,[],npar,nalt),3);%[]*npar

    diff_x_temp=reshape(x,[],npar,nalt)-mean_x_temp;

    mean_x=sum(reshape(mean_x_temp,[],npar).*...
        reshape(dummy_count_sum,[],1),1);%1*npar

    d1llike=xysum-mean_x;


    % Computing hessian
    d2llike=-sum(reshape(phat,[],1,1,nalt).*reshape(diff_x_temp,[],npar,1,nalt).*...
        reshape(diff_x_temp,[],1,npar,nalt).*reshape(dummy_count_sum,[],1,1,1),[1,4]);
    d2llike=reshape(d2llike,npar,npar);


     %%%% Added %%%%
     llike=llike./nobs;
     d1llike=d1llike./nobs;
     d2llike=d2llike./nobs;
     %%%%%%%%%%%%%%%
     
    % Gauss iteration
    try
        b1 = b0 + linsolve(-d2llike, d1llike', lsopts);
    catch ME
        err = 1;
        break;
    end
    criter = sqrt( (b1-b0)'*(b1-b0) );
    b0 = b1;
    iter = iter + 1;

    % fprintf('Coefficients  = ');
    % b1'
    % fprintf('\n');
  end

  Avarb  = inv(-d2llike);
  % sdb    = sqrt(diag(Avarb));
  % tstat  = b0./sdb;
  % numyj  = sum(kron(ones(1,nalt), ydum)==kron(ones(nobs,1),(1:nalt)));
  % logL0  = sum(numyj.*log(numyj./nobs));
  % lrindex = 1 - llike/logL0;
end
