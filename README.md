Simple Python scripts for reinforced learning with SB3.
-------------------------------------------------------

Train model (by default a nVidia GPU (cuda) is required):

    python3 lunartrain_ppo.py

Run the model:

    python3 lunarrun_ppo.py


Objective: land on both legs without crashing and minimize fuel consumption.
![gif](lunar.gif?raw=1)

Interface
---------

At each timestep give one of the following discrete actions to control the lunar lander:
 - 0: Do nothing,
 - 1: Fire left orientation engine,
 - 2: Fire the main engine,
 - 3: Fire right orientation engine.

The observation state vector for each timestep is an 8-dimensional vector: the coordinates of the lander in x & y, its linear velocities in x & y, its angle, its angular velocity, and two booleans that represent whether each leg is in contact with the ground or not.

Details: https://www.gymlibrary.dev/environments/box2d/lunar_lander/

Installation on Linux
---------------------

This is a repo to solve the lunar landing task in the Hugging Face Deep RL course
(https://huggingface.co/deep-rl-course/unit0/introduction). I prefer not to run this in a Google Colab notebook, but to run it locally with the following prerequisites.

Install additional packages:

    sudo apt-get update
    sudo apt install swig
    sudo apt install ffmpeg xvfb # if you want to record the simulation

Python packages:

    pip3 install importlib-metadata==4.12.0 # To overcome an issue with importlib-metadata https://stackoverflow.com/questions/73929564/entrypoints-object-has-no-attribute-get-digital-ocean
    pip3 install gym[box2d]
    pip3 install stable-baselines3[extra]
    pip3 install pyglet==1.5.1
    pip3 install ale-py==0.7.4 # To overcome an issue with gym (https://github.com/DLR-RM/stable-baselines3/issues/875)
    pip3 install pickle5
    pip3 install huggingface_sb3 # if you want to interact with the hugging face community
    

