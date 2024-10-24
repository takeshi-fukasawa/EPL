function [out,other_vars]=pchoice_update_func(pchoice,theta,...
e0,e1,u0,u1,...
zval,ptranz,bdisc,aval,mstate,ptranz_kron)

  nplayer = size(pchoice, 2);
  numa = 2^nplayer;
  numz = size(zval, 1);
  numx = numz*numa;
  naction = 2;

   if isempty(e0)==1 % If middle vars are empty, compute using functions 

        out=...
        compute_e_u_from_pchoice_func(pchoice,theta, zval, ptranz, bdisc,...
        aval,mstate,ptranz_kron);

        e0=out{1};
        e1=out{2};
        u0=out{3};
        u1=out{4};
   end

    % ----------------------------
    %  (e) Updating probabilities
    % ----------------------------
    v = zeros(numx,nplayer,naction);
    pchoice_updated=pchoice;

    for i = 1:nplayer
      v(:,i,2) = u1((i-1)*numx+1:i*numx,:)*theta + e1((i-1)*numx+1:i*numx,:);
      v(:,i,1) = u0((i-1)*numx+1:i*numx,:)*theta + e0((i-1)*numx+1:i*numx,:);
      buff = v(:,i,2) - v(:,i,1);
      pchoice_updated(:,i) = exp(buff)./(1+exp(buff));
    end

out={pchoice_updated};
other_vars=[];

end
