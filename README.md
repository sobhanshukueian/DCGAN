# DCGAN 

GANs are generative models: they create new data instances that resemble your training data.

The **Generator** tries to produce data that come from some probability distribution. That would be you trying to reproduce the party’s tickets.

The **Discriminator** acts like a judge. It gets to decide if the input comes from the generator or from the true training set. That would be the party’s security comparing your fake ticket with the true ticket to find flaws in your design.

**DCGAN**, or **Deep Convolutional GAN, is a generative adversarial** **network** architecture. It uses a couple of guidelines, in particular:

* Replacing any pooling layers with strided convolutions (discriminator) and fractional-strided convolutions (generator).
* Using batchnorm in both the generator and the discriminator.
* Removing fully connected hidden layers for deeper architectures.
* Using ReLU activation in generator for all layers except for the output, which uses tanh.
* Using LeakyReLU activation in the discriminator for all layer.

![image](https://user-images.githubusercontent.com/47561760/191743142-4fc35339-5e1f-4026-8221-724cbb7794fd.png)

The goal of the generator is to fool the discriminator, so the generative neural network is trained to maximise the final classification error (between true and generated data)

The goal of the discriminator is to detect fake generated data, so 
the discriminative neural network is trained to minimise the final classification error

# Dataset
The MNIST database of handwritten digits has a training set of 60,000 examples and a test set of 10,000 samples.
I used pytorch datasets for downloading dataset : 
```
train_dataset = datasets.MNIST('mnist/', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('mnist/', train=False, download=True, transform=transform)
```
# Model

**DCGAN**, or **Deep Convolutional GAN, is a generative adversarial** **network** architecture. It uses a couple of guidelines, in particular:

* Replacing any pooling layers with strided convolutions (discriminator) and fractional-strided convolutions (generator).
* Using batchnorm in both the generator and the discriminator.
* Removing fully connected hidden layers for deeper architectures.
* Using ReLU activation in generator for all layers except for the output, which uses tanh.
* Using LeakyReLU activation in the discriminator for all layer.

# Train
Trainer class Does the main part of code which is training model, plot the training process and save model each n epochs.

I Defined `Adam` Optimizer with learning rate 0.0002.

Each generative model training step occurse in `train_generator` function, descriminator model training step in `train_descriminator` and whole trining process in 
`train` function.

## Some Configurations
 
*   You can set epoch size : `EPOCHS` and batch size : `BATCH_SIZE`.
*   Set `device` that you want to train model on it : `device`(default runs on cuda if it's available)
*   You can set one of three `verboses` that prints info you want => 0 == nothing || 1 == model architecture || 2 == print optimizer || 3 == model parameters size.
*   Each time you train model weights and plot(if `save_plots` == True) will be saved in `save_dir`.
*   You can find a `configs` file in `save_dir` that contains some information about run. 

# Results

Trained 200 epochs:

![epoch-190-loss-plot](https://user-images.githubusercontent.com/47561760/191743929-a52ec953-9efb-46bc-9609-f727c0096d7b.png)
![DC-GAN](https://user-images.githubusercontent.com/47561760/191743855-e79b798b-c668-49ea-a51d-40f93f05e8d4.gif)
