Implementation of a Memory-Q-Network using keras and a slightly modified keras-rl. WIP


## Usage: 
```
python MQN.py [weights_name].h5 [train || test] [options = (debug, )]
```

[weights_name] will be the name of a file containing the weights for the network (in .h5 format). The weights will be loaded prior to any training or testing.

If this file does not exist and you specify `train`, then a folder with the name [weights_name] will be created in the directory `./data` (which will also be created if it does not exist). Then once your model has finished training, the program will save the model weights, hyperparameters, and training metrics in this folder.

If you specify `test` then the model will be tested on an environment without any training.

[options] are optional arguments. Currently only a `debug` option is implemented.

## Dependencies:
* Keras v2.0.8
* Numpy
* Matplotlib
* Seaborn
* h5py
* (might be more)

OpenAI Gym is packaged as part of the repo - this was for portability reasons. It's also easier to make new environments this way.

## TODO:
* Change naming conventions for cmd. line args ([weights_name].h5 should just be [model_name])
* Automatic model initialization from hyperparameters
* A `setup.py` to easily install dependencies
* Test mode callbacks / data recording
* Containerization for easier portability (maybe?)
