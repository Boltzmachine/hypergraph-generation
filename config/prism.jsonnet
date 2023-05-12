local hidden_dim = 256;
local T = 1000;
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
          type: "transformer",
          hidden_dim: hidden_dim,
          position_encoder: {
            type: "sin",
            d_model: hidden_dim,
            max_len: T
          },
          // n_layer: 6,
        },
        node_criterion: {
          type: "mse",
          reduction: "none"
        },
        face_criterion: {
          type: "bce",
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
            beta_schedule: "cosine_beta_schedule",
            n_classes: 2,
          }
        },
        visualizer: {
          type: "blender"
        },
        sample_bs: 16,
      },
      data_module: {
        type: "prism",
        batch_size: batch_size,
      },
      run_name: "prism"
    },
  },
}
