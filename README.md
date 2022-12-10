Simple Python scripts for reinforced learning with SB3.

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
    pip3 install -U colabgymrender

