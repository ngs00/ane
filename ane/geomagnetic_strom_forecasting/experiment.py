from geomagnetic_strom_forecasting.dml import exec_dml


mae, rmse, corr, precision, recall, f1 = exec_dml(dataset_name='mnn_a',
                                                  rnn_model='gru',
                                                  init_lr_emb=5e-4,
                                                  l2_emb=1e-6,
                                                  bs_emb=32,
                                                  dim_emb=24,
                                                  init_lr_pred=1e-3,
                                                  l2_pred=1e-6,
                                                  bs_pred=32)
print(mae, rmse, corr, precision, recall, f1)
