To build stand alone executables you should start by creating a virtual environment to reduce the number of modules included in the executables.

After creating the environment you should install tensorflow as many of the modules then will be loaded.

Then install PyInstaller with pip: ```pip install pyinstaller```

Now you can execute build.cmd to build all executables - remember that when executing this you need to have activated your virtual environment.

Currently there are only builds for Windows.
