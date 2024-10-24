function y = JV_APPROX_func(v, F,varargin)

    % v: vector

    x=varargin{1};
    fixed_point_spec=varargin{end-1};
    Fx=varargin{end};
   
    v=reshape(v,size(x));
 
    central_difference_spec=1;
    

    if central_difference_spec==0
        eps_val=sqrt(eps)./max(max(abs(v(:))),1e-8);
    else
        eps_val=(eps.^(1/3))./max(max(abs(v(:))),1e-8);
        %eps_val=eps.^(1/3);
    end
    
    if isempty(Fx)==1 & central_difference_spec==0
       Fx = F(varargin{1:end-2}); % unperturbed residual
       if fixed_point_spec==1
          Fx=x-Fx;
       end
    end
    
    x_eps_plus=x+eps_val*v;
    Fx_per_plus = F(x_eps_plus,varargin{2:end-2}); % perturbed residual

    if fixed_point_spec==1
       Fx_per_plus=x_eps_plus-Fx_per_plus;
    end
    
    if central_difference_spec==0
        y = (Fx_per_plus - Fx) / eps_val; % approximation of jacobian action on krylov vector
    else
        x_eps_minus=x-eps_val*v;
        Fx_per_minus = F(x_eps_minus,varargin{2:end-2}); % perturbed residual

        if fixed_point_spec==1
          Fx_per_minus=x_eps_minus-Fx_per_minus;
        end

        y = (Fx_per_plus - Fx_per_minus) / (2*eps_val); % approximation of jacobian action on krylov vector
    end

    y=y(:);% vector

end