local hidden_dim = 512;
local T = 500;
local batch_size = 1;


{
  steps: {
    train: {
      type: "train",
      trainer: {
        max_epochs: 1000,
        accelerator: "auto",
      },
      model: {
        type: 'diffusion',
        model: {
          type: "hyper",
          hidden_dim: hidden_dim,
          position_encoder: {
            type: "position",
            d_model: hidden_dim,
            max_len: T
          }
        },
        learning_rate: 1e-3,
        transition: {
          T: T,
          node_scheduler: {
            type: "gaussian_discrete",
            beta_schedule: "linear_beta_schedule",
            n_classes: 256
          },
          edge_scheduler: {
            type: "uniform_discrete",
            beta_schedule: "cosine_beta_schedule",
            n_classes: 2,
          }
        }
      },
      data_module: {
        type: "shapenet",
        data_path: "data/shapenet/",
        subset: "04379243",
        batch_size: batch_size,
        num_workers: 5
      },
    },
  },
}
