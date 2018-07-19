# DRQN-tensorflow
Deep Recurrent Q Learning using Tensorflow, openai/gym and openai/retro

This repository contains code for training a DQN or a DRQN on [openai/gym](https://github.com/openai/gym) Atari and [openai/retro](https://github.com/openai/retro) environments. 

Note that training on Retro environments is completely experimental as of now and these environments have to
be wrapped to reduce the action space to a more sensible subspace of all
actions for each game. The wrapper currently implemented only makes sense for
the SEGA Sonic environments.
 ### Installation
 You can install all dependencies by issuing following command:
 ```
 pip install -r requirements.txt
 ```
 This will install Tensorflow without GPU support. However, I highly recommend using Tensorflow with GPU support, otherwise training will take a very long time. For more information on this topic please see https://www.tensorflow.org/install/. In order to run the retro environments, you have to gather the roms of the games you want to play and import them: https://github.com/openai/retro#roms
### Running
You can start training by:
```
python main.py --gym=gym --steps=10000000 --train=True --network_type=dqn --env_name=Breakout-v0
```
This will train a DQN on Atari Breakout for 10 mio observations. For more on command line parameters please see
```
python main.py -h
```
Visualizing the training process can be done using tensorboard by:
```
tensorboard --logdir=out
```

### Result after training for 10mio steps (approx. 11 hours on GTX 1080 Ti)
![Alt Text](https://github.com/marctuscher/dqn/blob/master/assets/breakout_10mio.gif)
### References
1. [DQN-tensorflow](https://github.com/devsisters/DQN-tensorflow)
2. [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/pdf/1312.5602.pdf)
3. [Playing FPS Games with Deep Reinforcement Learning](https://arxiv.org/pdf/1609.05521.pdf)
4. [Deep Recurrent Q-Learning for Partially Observable MDPs](https://arxiv.org/pdf/1507.06527.pdf)
