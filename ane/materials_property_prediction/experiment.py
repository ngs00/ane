from materials_property_prediction.dml import exec_dml


# Index of target materials property
# 2: formation energy
# 3: band gap
# 4: shear modulus
# 5: bulk modulus
idx_target = 2


mae, rmse, r2 = exec_dml(dataset_name='mps',
                         idx_target=idx_target,
                         gnn_model='mpnn',
                         init_lr_emb=5e-4,
                         l2_emb=0,
                         bs_emb=32,
                         dim_emb=32,
                         init_lr_pred=5e-4,
                         l2_pred=1e-6,
                         bs_pred=64)
print(mae, rmse, r2)
