# robohearts

**Akash Kwatra and Lucas Kabela**

_Using reinforcement learning for the game of hearts. <3_

<a href="http://www.youtube.com/watch?feature=player_embedded&v=YOUTUBE_VIDEO_ID_HERE
" target="_blank"><img src="http://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg" 
alt="Video of project description" width="240" height="180" border="10" /></a>

---
## TODO:
 - Rename alpha parameter models based on learning rate, upload other models and weights for the final writeup
 - Clean mc_monte_carlo and policy_grad notebooks
 - Clean code we have added (and specify)
 - Add data to writeup
 - Review Write up (Into Rough draft by tonight --12pm)
 - Record video and add link
 - Executables (?)
 - Fill in ReadMe


## References:
This Opengym AI environment is a slightly modified version of https://github.com/zmcx16/OpenAI-Gym-Hearts


## Getting Started

### Prerequisites
The following base packages are required to run the repository:

 - [Python](https://www.python.org/) - 3.6+
 - [Gym](https://gym.openai.com/) - 0.15.4+
 - [Numpy](https://numpy.org/) - 1.16.4+
 - [PyTorch](https://pytorch.org/) - Lastest (1.2.0+)
 - [TensorBoard](https://www.tensorflow.org/tensorboard) - 2.0+
 - [TQDM](https://tqdm.github.io/) - 4.32.1+

### First Steps
This repository contains code for running repeatable experiments on the utility of various subsets of raw feature state for value approximation.  We have provided 4 notebooks to run repeatable expirements for this environment:

 - [run_hearts.ipynb](./run_hearts.ipynb) a notebook for gaining familiarty with the HEARTS environment.  Allows user to play an interactive game of hearts

 - [simple.ipynb](./simple.ipynb) trains and tests linear value function approximation using monte carlo rollouts

 - [mlp.ipynb](./mlp.ipynb) trains and tests non linear value function approximation with a Neural Network using monte carlo rollouts with a configurable feature set

 - [reinforce.ipynb](./reinforce.ipynb) trains and test a multi layer perception network using the policy gradient algorithm REINFORCE.


## Repository Structure

`|
|
|\
| gymhearts
|   |
|   | 
|   |\
|   | Agent
|   | 
|   |\
|   | Hearts
|
|
|\
| model_zoo
|  | 
|  |
|  |\
|  | feature_study_models
|  |
|  |\
|  | linear_v_nonlinear_models
|  |
|  |\
|  | policy_grad_models
|
|
|\
| writeup
|  |
|  |
|  |\
|  | data   
`
### Gymhearts
Contains the OpenAI gym environment code as well as the logic for various agents implemented 

#### Agent
Agent folder contains a variety of agents for playing the game of Hearts.  This includes a human player, which enables an interactive game through command prompt, monte_carlo agent, which uses simple linear function approximation and trains with linear rollouts, and monte_carlo_nn.py, which contain an agent using nonlinear function approximation with neural networks.  The code for policy gradient methods and REINFORCE agents also reside in this folder with the random agent which is used for baseline comparison in experiments.  Finally, utilities for agents, such as filtering valid moves and manipulating raw observations are provided in this folder.

#### Hearts
Hearts contains the code for the game environment (see https://github.com/zmcx16/OpenAI-Gym-Hearts), with minor modifications to environment rendering and valid moves.

### model_zoo
Directory containing the saved models from experiments.  These can be loaded and evaluated using the notebooks provided

#### feature_study_models
This directory contains models from the study on combinations of raw features and agent performance.  The numbers following file names correspond to the different features used, with these numbers corresponding to: 
    `[in_hand, in_play, played_cards, won_cards, scores]`
using one based indexing.

#### linear_v_nonlinear_models
These models correspond to the models trained in using only in_hand set of raw features for both a linear (simple dot product) and nonlinear (neural network).  This folder contains the weights for the linear model, as well as the saved model for the neural network, trained for 10_000 epochs

#### policy_grad_models
This directory store the models trained using the REINFORCE algorithm, following the naming convention in feature_study_models.  

### Writeup
This folder contains a report on the construction and findings of our experiments

#### Data
Contains charts, graphs, and the raw data collected for our expirement


## License:
This project is licensed under the terms of the MIT license.