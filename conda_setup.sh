conda create -n ben python=3.12

conda activate ben

# Core scientific stack via conda (versions chosen to satisfy requirements.txt).
conda install "numpy>=2.0,<2.1" "scipy>=1.14" matplotlib jupyter scikit-learn pandas
# gdbm gives shelve a real dbm backend; without it shelve falls back to dbm.dumb,
# which chmods its files and fails on filesystems that forbid it (e.g. WSL /mnt).
conda install gdbm

# All other Python dependencies live in requirements.txt - the single source of
# truth. Run this from the repository root and keep the two in sync by editing
# requirements.txt only. (Pins of note: keras 3.5.X - the version the models
# were trained on - and tensorflow 2.18.1.)
pip install -r requirements.txt

# Only needed to build the standalone executables, not to run BEN.
# setuptools<81 because PyInstaller's altgraph still imports pkg_resources, which
# setuptools removed in v81 - newer setuptools breaks the PyInstaller build.
pip install pyinstaller "setuptools<81"

# You might also need to install jq if you want to run the automated matches in scripts/match/bidding.

https://jqlang.github.io/jq/download/


# Using chocolatey to install is recommended on windows.
