function resid=G_v_func(v,theta,...
zval,ptranz,bdisc,aval,mstate,ptranz_kron)

%%% v: vector; not array %%%

  nplayer = size(aval, 2);
  numa = 2^nplayer;
  numz = size(zval, 1);
  numx = numz*numa;
  naction=2;
  v=reshape(v,numx,nplayer,naction);

   [out,other_vars]=v_update_func(v,theta,...
[],[],[],[],...
zval,ptranz,bdisc,aval,mstate,ptranz_kron);

   v_updated=out{1};
    resid=-v_updated(:)+v(:);
end
