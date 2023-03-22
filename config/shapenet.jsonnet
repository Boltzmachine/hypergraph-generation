local hidden_dim = 64;
local T = 500;
local batch_size = 2;


{
  steps: {
    train: {
      type: "train",
      trainer: {
        max_epochs: 2000,
        accelerator: "auto",
        check_val_every_n_epoch: 100,
      },
      model: {
        type: 'diffusion',
        model: {
          type: "hyper",
          hidden_dim: hidden_dim,
          position_encoder: {
            type: "mlp",
            d_model: hidden_dim,
            max_len: T
          },
          backbone: "pyg"
        },
        node_criterion: {
          type: "mse",
          reduction: "none"
        },
        edge_criterion: {
          type: "bce",
          reduction: "none",
          pos_weight: 20,#283.38,
        },
        learning_rate: 3e-4,
        transition: {
          T: T,
          node_scheduler: {
            type: "identity_continuous",
            beta_schedule: "linear_beta_schedule",
            // n_classes: 256
          },
          edge_scheduler: {
            type: "uniform_discrete",
            beta_schedule: "linear_beta_schedule",
            n_classes: 2,
          }
        },
        sample_bs: 1
      },
      data_module: {
        type: "shapenet",
        data_path: "data/shapenet/",
        subset: "04379243",
        batch_size: batch_size,
        num_workers: 0
      },
      run_name: "shapenet"
    },
  },
}
