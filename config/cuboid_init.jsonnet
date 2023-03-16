local hidden_dim = 256;
local T = 500;
local batch_size = 512;

{
  steps: {
    train: {
      type: "train",
      trainer: {
        max_epochs: 200,
        accelerator: "auto",
      },
      model: {
        type: 'diffusion',
        model: {
          type: "hyper_initial",
          hidden_dim: hidden_dim,
          position_encoder: {
            type: "position",
            d_model: hidden_dim,
            max_len: T
          },
          backbone: "pyg"
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
            type: "identity_discrete",
            beta_schedule: "cosine_beta_schedule",
            n_classes: 2,
          }
        },
        visualizer: {
          type: "blender"
        }
      },
      data_module: {
        type: "cuboid",
        batch_size: batch_size,
      },
      run_name: "with init positions"
    },
  },
}
