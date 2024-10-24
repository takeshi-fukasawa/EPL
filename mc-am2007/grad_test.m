
param=bcnpl;
[obj,grad]=log_likelihood_func0(param,dummy_count, [u0, u1], [e0, e1])
param_eps=param;
eps_val=1e-6;
param_eps(1)=param(1)+eps_val;
obj_eps=log_likelihood_func0(param_eps,dummy_count, [u0, uobs1], [e0, e1])
diff=(obj_eps-obj)./eps_val;
