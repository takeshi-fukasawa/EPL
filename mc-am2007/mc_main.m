% mc_main.m -- Control program for running Monte Carlo experiments.


clear
global selexper
global nobs
global compute_jacobian_spec krylov_spec
global flag_vec relres_cell iter_gmres_cell resvec_cell

warning('off', 'all')

addpath('./spectral')
addpath('./results')

save_spec=0;
nrepli = 100;%100;%1000  % Number of Monte carlo replications
nplayer = 5;    %  Number of players

selexper = 3;
k=0;
nobs=1600;

%% Comparison of EPL-analytical, EPL-krylov, and EPL-JF
run_nfxp_spec=0;%%%

            %% Analytical (Jacobian-based)
            compute_jacobian_spec=1;%%%%%
            krylov_spec=0;%%%%%
            mc_epl
            ww_array_stack_analytical=ww_array_stack;
            bmat_1epl_analytical=bmat_1epl; 
            bmat_2epl_analytical=bmat_2epl;
            bmat_3epl_analytical=bmat_3epl;
            bmat_cepl_analytical=bmat_cepl;
            table_analytical=table;

            %% Krylov (Jacobian-based)
            compute_jacobian_spec=1;%%%%%
            krylov_spec=1;%%%%%
            mc_epl
            ww_array_stack_krylov=ww_array_stack;
            bmat_1epl_krylov=bmat_1epl; 
            bmat_2epl_krylov=bmat_2epl;
            bmat_3epl_krylov=bmat_3epl;
            bmat_cepl_krylov=bmat_cepl;
            table_krylov=table;
            
            %% JF (Jacobian-free)
            compute_jacobian_spec=0;%%%%%
            krylov_spec=1;%%%%%
            mc_epl
            ww_array_stack_JF=ww_array_stack;
            bmat_1epl_JF=bmat_1epl; 
            bmat_2epl_JF=bmat_2epl;
            bmat_3epl_JF=bmat_3epl;
            bmat_cepl_JF=bmat_cepl;
            table_JF=table;

            writematrix(table_analytical,'./results/table_analytical.csv')
            writematrix(table_krylov,'./results/table_krylov.csv')
            writematrix(table_JF,'./results/table_JF.csv')
                        
            Time_comparison=[table_analytical(end-3:end,3),...
                table_krylov(end-3:end,3),table_JF(end-3:end,3),...
                table_analytical(end-3:end,5),...
                table_krylov(end-3:end,5),table_JF(end-3:end,5),...
                table_analytical(end-3:end,7),...
                table_krylov(end-3:end,7),table_JF(end-3:end,7),...
                table_analytical(end-3:end,9),...
                table_krylov(end-3:end,9),table_JF(end-3:end,9)];
            Time_comparison=round(Time_comparison,3);

            writematrix(Time_comparison,'./results/time_comparison.csv')
            

            difference_cepl_JF=abs(bmat_cepl_JF-bmat_cepl_analytical);
            difference_cepl_krylov=abs(bmat_cepl_krylov-bmat_cepl_analytical);
            log10_difference_cepl_krylov=log10(difference_cepl_krylov(:));
            log10_difference_cepl_JF=log10(difference_cepl_JF(:));
            difference_cepl_table=[[mean(log10_difference_cepl_krylov);max(log10_difference_cepl_krylov)],...
                [mean(log10_difference_cepl_JF);max(log10_difference_cepl_JF)]];

            difference_1epl_JF=abs(bmat_1epl_JF-bmat_1epl_analytical);
            difference_1epl_krylov=abs(bmat_1epl_krylov-bmat_1epl_analytical);
            log10_difference_1epl_krylov=log10(difference_1epl_krylov(:));
            log10_difference_1epl_JF=log10(difference_1epl_JF(:));
            difference_1epl_table=[[mean(log10_difference_1epl_krylov);max(log10_difference_1epl_krylov)],...
                [mean(log10_difference_1epl_JF);max(log10_difference_1epl_JF)]];

            difference_2epl_JF=abs(bmat_2epl_JF-bmat_2epl_analytical);
            difference_2epl_krylov=abs(bmat_2epl_krylov-bmat_2epl_analytical);
            log10_difference_2epl_krylov=log10(difference_2epl_krylov(:));
            log10_difference_2epl_JF=log10(difference_2epl_JF(:));
            difference_2epl_table=[[mean(log10_difference_2epl_krylov);max(log10_difference_2epl_krylov)],...
                [mean(log10_difference_2epl_JF);max(log10_difference_2epl_JF)]];

            difference_3epl_JF=abs(bmat_3epl_JF-bmat_3epl_analytical);
            difference_3epl_krylov=abs(bmat_3epl_krylov-bmat_3epl_analytical);
            log10_difference_3epl_krylov=log10(difference_3epl_krylov(:));
            log10_difference_3epl_JF=log10(difference_3epl_JF(:));
            difference_3epl_table=[[mean(log10_difference_3epl_krylov);max(log10_difference_3epl_krylov)],...
                [mean(log10_difference_3epl_JF);max(log10_difference_2epl_JF)]];

            difference_table=round([[NaN;NaN],difference_1epl_table,...
                [NaN;NaN],difference_2epl_table,...
                [NaN;NaN],difference_3epl_table,...
                [NaN;NaN],difference_cepl_table],3);


            writematrix(difference_table,'./results/difference_table.csv')
            
         
%% Comparison of EPL-JF and NFXP-numer-diff
run_nfxp_spec=1;
nrepli = 3;%1000  % Number of Monte carlo replications
nplayer = 5;    %  Number of players
compute_jacobian_spec=0;%%%%%
krylov_spec=1;%%%%%
mc_epl

diff_bmat=abs(bmat_cepl-bmat_nfxp);
log_diff=log10(diff_bmat(:));
table1=[[NaN;NaN],[mean(log_diff(:));max(log_diff(:))]];
table2=[[mean(time_cepl/nrepli);std(time_cepl/nrepli)],...
    [mean(time_nfxp/nrepli);std(time_nfxp/nrepli)]];


writematrix(round([table1;table2],3),'./results/summary_comparison_NFXP.csv')
