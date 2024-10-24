function out=compute_e_u_from_v_func...
    (v,param, zval, ptranz, bdisc,...
    aval,mstate,ptranz_kron)

%%% The following code does not depend on param 
  % ---------------
  % Some constants
  % ---------------
  eulerc = 0.5772;
  myzero = 1e-16;
  nplayer = size(aval, 2);
  numa = 2^nplayer;
  numz = size(zval, 1);
  numx = numz*numa;
  kparam = nplayer + 3;

    u0 = zeros(numx*nplayer,kparam);
    u1 = zeros(numx*nplayer,kparam);
    e0 = zeros(numx*nplayer,1);
    e1 = zeros(numx*nplayer,1);

    pchoice=zeros(numx, nplayer);
    for i = 1:nplayer
        vdiff = v(:,i,2) - v(:,i,1);
        expvdiff = exp(vdiff);
        pchoice(:,i) = expvdiff ./ (1 + expvdiff);
    end
    V=log(sum(exp(v),3))+eulerc;

        % -----------------------------------------------------------
    % (a) Matrix of transition probabilities Pr(a[t]|s[t],a[t-1])
    % -----------------------------------------------------------
    ptrana = ones(numx, numa);
    for i = 1:nplayer;
      mi = aval(:,i)';
      ppi = pchoice(:,i);
      ppi1 = repmat(ppi, 1, numa);
      ppi0 = 1 - ppi1;
      mi1 = repmat(mi, numx, 1);
      mi0 = 1 - mi1;
      ptrana = ptrana .* (ppi1 .^ mi1) .* (ppi0 .^ mi0);
    end

    % -----------------------------------------
    %  (c) Construction of explanatory variables
    % -----------------------------------------
    
    for i = 1:nplayer
      % --------------------------------------------
      %  (c.1) Matrices Pr(a[t] | s[t],a[t-1], ai[t])
      % --------------------------------------------
      mi = aval(:,i)';
      ppi = pchoice(:,i);
      ppi= (ppi>=myzero).*(ppi<=(1-myzero)).*ppi ...
         + (ppi<myzero).*myzero ...
         + (ppi>(1-myzero)).*(1-myzero);
      ppi1 = repmat(ppi, 1, numa);
      ppi0 = 1 - ppi1;
      mi1 = repmat(mi, numx, 1);
      mi0 = 1 - mi1;
      ptrani = ((ppi1 .^ mi1) .* (ppi0 .^ mi0));
      ptranai0 = ptrana .* (mi0 ./ ptrani);
      ptranai1 = ptrana .* (mi1 ./ ptrani);

      % ------------------------------------
      %  (c.2) Computing hi = E(ln(Sum(aj)+1))
      % ------------------------------------
      hi = aval;
      hi(:,i) = ones(numa, 1);
      hi = ptranai1 * log(sum(hi, 2));

      % ---------------------------
      %  (c.3) Creating U0 and U1
      % ---------------------------
      umat0 = zeros(numx, nplayer+3);
      umat1 = eye(nplayer);
      umat1 = umat1(i,:);
      umat1 = [ repmat(umat1, numx, 1), mstate(:,1), (-hi), (mstate(:,i+1)-1) ];

      % ----------------------------------
      %  (c.6) Creating utilda and etilda
      % ----------------------------------
      utilda0=umat0;
      utilda1=umat1;

      %I_numa=ones(numa,numa);
      %%ptranz_kron=(kron(ptranz, I_numa));


      %ptranz_kron=sparse(ptranz_kron);
      %ptranai0=sparse(ptranai0);
      %ptranai1=sparse(ptranai1);
   
      %ones_numz=ones(1,numz);

      %%%%ptranai0_temp =  ptranz_kron.* kron(ones_numz, ptranai0);

      ptranai0_temp=reshape(reshape(ptranz_kron,numx*numa,numz).*...
          reshape(ptranai0,numx*numa,1),numx,numa*numz);

      etilda0=bdisc*(ptranai0_temp*V(:,i));

      ptranai1_temp=reshape(reshape(ptranz_kron,numx*numa,numz).*...
          reshape(ptranai1,numx*numa,1),numx,numa*numz);

      %%%%ptranai1_temp = ptranz_kron .* kron(ones_numz, ptranai1);
      etilda1=bdisc*(ptranai1_temp*V(:,i));
   
      % -------------------------------------------
      %  (c.7) Creating observations uobs and eobs
      % -------------------------------------------
      u0((i-1)*numx+1:i*numx,:) = utilda0;
      u1((i-1)*numx+1:i*numx,:) = utilda1;
      e0((i-1)*numx+1:i*numx,:) = etilda0;
      e1((i-1)*numx+1:i*numx,:) = etilda1;

          
    end

    out={e0,e1,u0,u1};

end
