function [out,other_vars]=v_update_func(v,theta,...
e0,e1,u0,u1,...
zval,ptranz,bdisc,aval,mstate,ptranz_kron)


  nplayer = size(v, 2);
  numa = 2^nplayer;
  numz = size(zval, 1);
  numx = numz*numa;

   if isempty(e0)==1 % If middle vars are empty, compute using functions 

        out=...
        compute_e_u_from_v_func(v,theta,zval, ptranz, bdisc,...
        aval,mstate,ptranz_kron);

        e0=out{1};
        e1=out{2};
        u0=out{3};
        u1=out{4};
   end

    % ----------------------------
    %  (e) Updating probabilities
    % ----------------------------
    v_updated=v;
    
    for i = 1:nplayer
      v_updated(:,i,2) = u1((i-1)*numx+1:i*numx,:)*theta + e1((i-1)*numx+1:i*numx,:);
      v_updated(:,i,1) = u0((i-1)*numx+1:i*numx,:)*theta + e0((i-1)*numx+1:i*numx,:);      
    end

out={v_updated};
other_vars=[];

end
