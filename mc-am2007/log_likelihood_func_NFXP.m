function [llike,d1llike]=log_likelihood_func_NFXP(...
    b0,dummy_count,...
    v_initial0,pchoice_initial0,zval,ptranz,bdisc,aval,mstate,ptranz_kron)

  global v_global pchoice_global

  if isempty(v_global)==1
      v_global=v_initial0;
  end
  if isempty(pchoice_global)==1
      pchoice_global=pchoice_initial0;
  end

  myzero = 1e-16;
  nobs = sum(dummy_count(:), 1);
  nalt = size(dummy_count,2);
  npar = size(b0(:), 1);

  scale_ll=1;

    % Computing probabilities
   spec=[];
   spec.ITER_MAX=1000;
   spec.TOL=1e-12;


   spec.Anderson_acceleration=1;

   if 1==0
       spec.ITER_MAX=100;
       [output_spectral,~,iter_info_x]=...
            spectral_func(@pchoice_update_func,spec,{pchoice_global},b0,[],[],[],[],...
        zval,ptranz,bdisc,aval,mstate,ptranz_kron);
    
       phat=reshape(output_spectral{1},[],1);
       phat=[1-phat,phat];

   else
       [output_spectral,~,iter_info_x]=...
            spectral_func(@v_update_func,spec,{v_global},b0,[],[],[],[],...
        zval,ptranz,bdisc,aval,mstate,ptranz_kron);
    
        v_global=output_spectral{1};
        v=reshape(output_spectral{1},[],nalt);
    
        vmax = max(v, [], 2);
        v = v - vmax;
        phat = exp(v) ./ sum(exp(v), 2);%[]*nalt  
   end

    llike=sum(log(phat).*dummy_count,[1,2]);

     llike=llike./nobs;

    llike=llike*(-1);%%%%%%%

   llike=llike./scale_ll;

   d1llike=[];

   %b0

end % function


