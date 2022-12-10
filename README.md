Simple Python scripts for reinforced learning with SB3.
-------------------------------------------------------

Train model (by default a nVidia GPU (cuda) is required):

    python3 lunartrain_ppo.py

Run the model:

    python3 lunarrun_ppo.py

Demo
----

https://user-images.githubusercontent.com/648730/206876518-f7b13619-c913-403c-b5c2-547e7950800c.mp4

Installation
------------

Install additional packages:

    sudo apt-get update
    sudo apt install swig
    sudo apt install ffmpeg xvfb

Python packages:

    pip3 install importlib-metadata==4.12.0 # To overcome an issue with importlib-metadata https://stackoverflow.com/questions/73929564/entrypoints-object-has-no-attribute-get-digital-ocean
    pip3 install gym[box2d]
    pip3 install stable-baselines3[extra]
    pip3 install huggingface_sb3
    pip3 install pyglet==1.5.1
    pip3 install ale-py==0.7.4 # To overcome an issue with gym (https://github.com/DLR-RM/stable-baselines3/issues/875)
    pip3 install pickle5
    

