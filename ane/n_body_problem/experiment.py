from n_body_problem.dml import exec_dml


mae, rmse, corr = exec_dml(idx_dataset=10,
                           init_lr_emb=5e-4,
                           l2_emb=5e-6,
                           bs_emb=32,
                           dim_emb=32,
                           init_lr_pred=5e-4,
                           l2_pred=5e-6,
                           bs_pred=32)
print('eval', mae, rmse, corr)
