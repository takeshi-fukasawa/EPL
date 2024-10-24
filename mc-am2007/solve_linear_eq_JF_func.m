function [vec_sol,flag_vec,relres_vec,iter_cell,resvec_cell]=...
    solve_linear_eq_JF_func(b,spec_gmres,varargin)

    %%% Solve (dG/dx)^(-1) vec=b for vec.
    %%% Jacobian-free methods.
    %%% If fixed_point_spec==1, solve (I - dPhi/dx)^(-1) vec=b.
    
    %fun=varargin{1};
    %x=varargin{2};
    
    K=size(b,2);% b can be a matrix
    
    flag_vec=zeros(1,K);
    relres_vec=zeros(1,K);
    iter_cell=[];
    resvec_cell=[];


    %fixed_point_spec=varargin{end};
    
    Fx=[];%%%
    j_vec_approx=@(vec)JV_APPROX_func(vec,varargin{:},Fx);
    
    v_init=[];
    vec_sol=b;

    if isfield(spec_gmres,'TOL')==0
        TOL=1e-6;
    else
        TOL=spec_gmres.TOL;
    end

    if isfield(spec_gmres,'ITER_MAX')==0
        ITER_MAX=30;
    else
        ITER_MAX=spec_gmres.ITER_MAX;
    end
    


    for i=1:K
        [vec_sol(:,i),flag_vec(i),relres_vec(i),iter_cell{i},resvec_cell{i}]=...
            gmres(...
            j_vec_approx,b(:,i),[],TOL/norm(b(:,i)),ITER_MAX,[],[],v_init);
    end

end
