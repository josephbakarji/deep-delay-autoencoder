# Description

This code discovers an analytical model using a combination of time-delay embedding, autoencoders, and sparse identification of differential equations (SINDy). The details can be found in the paper [Discovering Governing Equations from Partial Measurements with Deep Delay Autoencoders](https://arxiv.org/abs/2201.05136).

### Abstract:
A central challenge in data-driven model discovery is the presence of hidden, or latent, variables that are not directly measured but are dynamically important. Takens’ theorem provides conditions for when it is possible to augment these partial measurements with time delayed information, resulting in an attractor that is diffeomorphic to that of the original full-state system. However, the coordinate transformation back to the original attractor is typically unknown, and learning the dynamics in the embedding space has remained an open challenge for decades. Here, we design a custom deep autoencoder network to learn a coordinate transformation from the delay embedded space into a new space where it is possible to represent the dynamics in a sparse, closed form. We demonstrate this approach on the Lorenz, Rossler, and Lotka-Volterra systems, learning dynamics from a single measurement variable. As a challenging example, we learn a Lorenz analogue from a single scalar variable extracted from a video of a chaotic waterwheel experiment. The resulting modeling framework combines deep learning to uncover effective coordinates and the sparse identification of nonlinear dynamics (SINDy) for interpretable modeling. Thus, we show that it is possible to simultaneously learn a closedform model and the associated coordinate system for partially observed dynamics.

# Code

The code builds on [SindyAutoencoders](https://github.com/kpchamp/SindyAutoencoders) with a Tensorflow 2 upgrade. Here's a brief description of the main files 

- `src` contains the heart of the code.
	- `net_config.py` contains the `SindyAutoencoder` which defines the deep network and SINDy architecture to be trained
	- `analyze.py` contains functions for reading and visualizing the results
	- `training.py` contains training and testing functions
- `examples` contains test cases and run files
	- `basic_params.py` contains the basic parameters that can be defined in SINDy autoencoders
	- `basic_run.py` has a function for hyperparameter optimization and runs the code with the basic parameters
	- `lorenz.py`, `predprey.py`, `rossler.py` and `waterlorenz.py` all generate data for training. 
- `testcases` contains some examples of changing the input parameters and training the model.
- `analyze` contains notebook that will read multiple results and visualize them.

The main assumption in the code is that the first dimension is taken as the 'measurement' and the algorithm attempts to recover it.