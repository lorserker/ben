# Using Tensorflow 2.16+

First of all you must create an environment for using this version of Tensorflow, as it is not backward compatible, so when installing Tensorflow 2.16+ you can only train TF 2.XC models.

After installing conda, execute the commands in the [conda_setup_tf2.sh](conda_setup_tf2.sh) script to create an environment and install the dependencies.

So after installing this you can start the BEN-server, but be aware, that currently only one configuration is using the TF2-models.

So running:

- python gameserver.py --conf config\default_tf2.conf

will load the configuration, and you are now using latest version of Tensorflow.


Training models using Tensorflow 2.X is simpler, and if you want to try it, then switch to the directory

- script/training

Where you can find folders for each of the different models used. In each of the folders there are a subdirectory with the name keras, where Tensorflow 2.X related training is placed.

Common principle is that the input is converted to binary, and then model training is started. You can continie training a model by increasing the number of epochs, and then start the script again.


If you upgrade to Tensorflow 2.16+ you will still be able to run the old models, but training will require the new models.

Keras should be at least version 3.4.1

Before training you must install

- pip install GPUtil