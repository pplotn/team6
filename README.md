# team6

Hello, it is the repository to get familiar with my code, of course it contains small dataset.
Sorry for long code.
The key part, which I want to edit are strings 180-184.

My purpose of paricipating in workshop utilize data-parallelism in gpu training, based on existing TF tools (tf.distribute.Strategy) or maybe Horovod. 
If it is needed I will rewrite code on torch.

Training on 1 Volta takes 15 hours for 80 epochs (200000 samples, network parameters 1500000).
I use tensorflow-gpu. 
And I use standard model.fit function to make the training.

Can you share any solution which is easy to use to utilize data parallelism with tensorflow?
I tried to scale code to 2 gpus with tensorflow  tf.distribute.Strategy long time ago, but training take double time, comparing to single gpu.

Single-gpu time
{'epochs': 80, 'batch_size': 2,
Epoch 00001: val_loss improved from inf to 97857.06732, saving model to ./keras_models/weights_best904.hdf5
199283/199283 - 695s - loss: 104483.9045 - MAPE: 846.2272 - coeff_determination: 0.5929 - val_loss: 97857.0673 - val_MAPE: 319.8510 - val_coeff_determination: 0.6211
Epoch 2/80
