# ViZDoom Dockerfiles

Provided dockerfiles (all images use Ubuntu 16.04):

* **python3** - Python3, ViZDoom with dependecies and X11 support. By default runs Python [basic example](examples/python/basic.py).
* **tensorflow** - Python3, ViZDoom with dependecies, Tensorflow and X11 support. By default runs [tensorflow learning example](examples/python/learning_tensorflow.py).
* **torch** - Torch, ViZDoom with dependencies and X11 support. By default run Lua [basic example](examples/lua/basic.py). TODO

## Building

```bash
# Uses Dockerfile in python3 directory to create image tagged vizdoom
docker build python3 -t vizdoom 


# Uses Dockerfile in tensorflow directory to create image tagged vizdoom_tf
docker build tensorflow -t vizdoom_tf 

```

## Running (requires building beforehand)

```
# '--net=host -e DISPLAY=${DISPLAY} --privileged' required for GUI
# Run basic example in container named 'basic' 
docker run  -ti --net=host -e DISPLAY=${DISPLAY} --privileged --rm --name basic vizdoom

# Run tensorflow learning example in container named 'vizdoom_tf'
nvidia-docker run  -ti --privileged --net=host -e DISPLAY=${DISPLAY} --rm --name vizdoom_tf vizdoom_tf

# Run bash in interactive mode
docker run  -ti --privileged --net=host -e DISPLAY=${DISPLAY} --rm --name basic vizdoom bash

```

## Other dockerfiles:

* Ubuntu 17.04, Anaconda3-4.4.0 by @guiambros
  * Dockerfile: https://gist.github.com/guiambros/1ee62876fb309207b02eb69c8c7b39fa 
  * Shell script: https://gist.github.com/guiambros/8847ae6d23299a395c3d8a566996a16f
