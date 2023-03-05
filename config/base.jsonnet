{
  steps: {
    construct_dataset: {
      type: "construct_dataset",
        dataset: {
          type: "cuboid"
        },
    },
    train: {
      type: "train",
      trainer: {
        max_epochs: 100,
        accelerator: "auto",
      },
      model: {
        type: 'diffusion',
        model: {
          type: "hyper"
        },
        learning_rate: 1e-3,
        transition: {
          T: 500,
          node_scheduler: {
            type: "dummy_discrete",
            n_classes: 256
          },
          edge_scheduler: {
            type: "uniform_discrete",
            n_classes: 2,
          }
        }
      },
      dataloader: {
        dataset: {
          type: "ref",
          ref: "construct_dataset"
        },
        batch_size: 512,
        shuffle: true
      },
    },
  },
}
