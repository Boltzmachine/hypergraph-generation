local hidden_dim = 256;
local T = 300;
local batch_size = 512;

{
  steps: {
    train: {
      type: "train",
      trainer: {
        max_epochs: 200,
        accelerator: "auto",
        check_val_every_n_epoch: 10,
      },
      model: {
        type: 'diffusion',
        model: {
          type: "gb_tf",
          position_encoder: {
            type: "sin",
            d_model: hidden_dim,
            max_len: T
          },
        },
        node_criterion: {
          type: "mse",
          reduction: "none"
        },
        edge_criterion: {
          type: "ce",
          reduction: "none"
        },
        learning_rate: 1e-3,
        transition: {
          T: T,
          node_scheduler: {
            type: "gaussian_continuous",
            beta_schedule: "linear_beta_schedule",
            // n_classes: 256
          },
          edge_scheduler: {
            type: "uniform_discrete",
            beta_schedule: "linear_beta_schedule",
            n_classes: 2,
          }
        },
        visualizer: {
          type: "matplotlib"
        },
        sample_bs: 64,
      },
      data_module: {
        type: "cuboid",
        batch_size: batch_size,
      },
      run_name: "cuboid-edge"
    },
  },
}
