# team6

Hello, it is the repository to get familiar with my code, of course it contains small dataset.
Sorry for long code.
The key part, which I want to edit are lines 180-184.

My purpose of paricipating in workshop utilize data-parallelism in gpu training, based on existing TF tools (tf.distribute.Strategy) or maybe Horovod. 
If it is needed I will rewrite code on torch.

Training on 1 Volta takes 15 hours for 80 epochs (200000 samples, network parameters 1500000).
I use tensorflow-gpu. 
And I use standard model.fit function to make the training.

