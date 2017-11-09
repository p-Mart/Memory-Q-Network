Implementation of a Memory-Q-Network using keras and a slightly modified keras-rl. WIP

Be sure to run the following command to install the maze environments.
```
sudo python openai_maze_envs/setup.py install
```


Run using: 
```
python MQN.py [weights_name].h5
```

[weights_name] will be the name of a file containing the weights for the network. If it doesn't exist already, it will be created automatically. Must be in .h5 format.
