To build stand alone executables you should start by creating a virtual environment to reduce the number of modules included in the executables.

After creating the environment you should install tensorflow as many of the modules then will be loaded.

Then install PyInstaller with pip: ```pip install pyinstaller```

Now you can execute build.cmd to build all executables - remember that when executing this you need to have activated your virtual environment.

Currently there are only builds for Windows.

## Publishing large data files

Some files are too large for the repository - training data in particular. They
are published as assets on a fixed GitHub release tag (`training-data`) rather
than being attached to each version release: the data changes rarely and
independently of the code, so a stable tag keeps the documentation links valid
across versions and avoids repeating a large upload every time.

Training boards are compressed first. `BBA\Boards` holds tens of GB, but only
the `-OK_boards.pbn` set for the current BBA version is worth publishing - that
is the complete input needed to retrain the bidding models. The rest is either
an intermediate, a reject file, or a superseded version.

```
.\Compress-Boards.ps1 -DryRun          list the board files that would be compressed
.\Compress-Boards.ps1                  zip them into install\data-staging
```

The data is text and compresses about 8x, so roughly 6 GB becomes well under
1 GB. Zips already newer than their source are skipped, so re-running is cheap;
use -Force to rebuild them anyway. `install\data-staging` is gitignored.

The files are listed in `training-data.txt`, one per line as
`<source path> | <asset name>`. A source path may be absolute or relative to the
repository root, so files that live outside the checkout can be included.

```
.\Publish-TrainingData.ps1 -DryRun      check the manifest and show what would upload
.\Publish-TrainingData.ps1              create or update the draft release
.\Publish-TrainingData.ps1 -Publish     make it live
.\Publish-TrainingData.ps1 -Force       replace assets already on the release
```

A release asset may be at most 2 GB; the script refuses anything larger rather
than failing part-way through an upload.

This is separate from `Publish-Release.ps1`, which uploads the four version
zips built by `BuildAll.cmd`.
