# robohearts

**Akash Kwatra and Lucas Kabela**

_Using reinforcement learning for the game of hearts. <3_

<a href="http://www.youtube.com/watch?feature=player_embedded&v=n8zhiRtXqHM
" target="_blank"><img src="http://img.youtube.com/vi/n8zhiRtXqHM/0.jpg" 
alt="Video of project description" width="240" height="180" border="10" /></a>

[Results and Writeup](./writeup/CS394R_Final_Project_First_Draft.pdf)

---
## TODO LAST DAY:
 - ~~Finish Data Collection~~
 - ~~Rename alpha parameter models based on learning rate, upload models and weights for the final writeup~~
 - ~~Clean simple and reinforce notebooks~~
 - ~~Clean code we have added (and specify which we added / modified)~~
 - ~~Add data to writeup~~
 - ~~Record video and add link~~
 - Final WriteUp
 - ~~Update ReadMe~~
 - Review checklist (website)
 - Turn it in!

### Stretch Goals:
 - ~~Refactor agent utils, reinforce utils into nn utils, hand utils~~
 - Add Cool logo to repo
 - ~Comment all the code~

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

 - [run_hearts.ipynb](./run_hearts.ipynb) Executable for the grader / reader to run, this will showcase the performance of our best models. Run this from the root directory of the repo -- __jupyter notebook run_hearts.ipynb__. Make sure requirements are installed. May be easiest on a conda installation.

 - [simple.ipynb](./simple.ipynb) trains and tests linear value function approximation using monte carlo rollouts

 - [mlp.ipynb](./mlp.ipynb) trains and tests non linear value function approximation with a Neural Network using monte carlo rollouts with a configurable feature set

 - [reinforce.ipynb](./reinforce.ipynb) trains and test a multi layer perception network using the policy gradient algorithm REINFORCE.


## Repository Structure

    
    .
    ├── gymhearts               # Agents and Environment 
    |   ├── Agent
    |   ├── Hearts
    |
    ├── model_zoo               # Saved trained models from experiments
    |   ├── feature_study_models
    |   ├── linear_v_nonlinear_models
    |   ├── policy_grad_models
    |
    ├── writeup                 # Paper and data from expirements                   
    ├── LICENSE
    └── README.md


### Gymhearts
Contains the OpenAI gym environment code as well as the logic for various agents implemented 

#### Agent
Agent folder contains a variety of agents for playing the game of Hearts, includes human players, linear value approximation agent, nonlinear value apporximation agent,
policy gradient agents, and a random agent which serves as a baseline for comparisson and training.

#### Hearts
Hearts contains the code for the game environment (see https://github.com/zmcx16/OpenAI-Gym-Hearts), with minor modifications to environment rendering and valid moves.


### Model Zoo
Directory containing the saved models from experiments.  These can be loaded and evaluated using the notebooks provided

#### feature_study_models
This directory contains models from the study on combinations of raw features and agent performance.  The numbers following file names correspond to the different features used, with these numbers corresponding to: 
    `[in_hand, in_play, played_cards, won_cards, scores]`
using one based indexing.

#### linear_v_nonlinear_models
These models correspond to the models trained in using only in_hand set of raw features for both a linear (simple dot product) and nonlinear (neural network).  This folder contains the weights for the linear model, as well as the saved model for the neural network, trained for 10_000 epochs

#### policy_grad_models
This directory store the models trained using the REINFORCE algorithm, following the naming convention in feature_study_models.  

#### reinforce_alpha_study
This directory stores the models from our learning rate study of the REINFORCE with baseline method.

#### mc_simple
Stores the best model from our simple mc function approximator

### Writeup
This folder contains a report on the construction and findings of our experiments

### Data
[Raw data from experiments](https://docs.google.com/spreadsheets/d/1O8LAQ1LNYp1OG_tdRdm83FdInzPIsjeYT0f5_iFs5ZQ/edit?usp=sharing)



## License:
This project is licensed under the terms of the MIT license.