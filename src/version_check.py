"""
Checks at startup that the TensorFlow and Keras in this Python are the ones BEN's models need.

The play and opening-lead models are saved with Keras 3.15. An older Keras cannot load them and
fails deep inside model loading with "could not be deserialized properly", which says nothing
about the cause -- so every entry point that loads models calls check_tf_keras() first and tells
the user what to install.

A newer Keras than the one tested only gives a warning: it usually works, but Keras 3.7 changed
how a model with two inputs is called and silently ruined the 2024 lead models.

Keep the versions in step with requirements.txt and pyproject.toml.
"""
import os
import re
import sys

MIN_TENSORFLOW = (2, 21)
MIN_KERAS = (3, 15)
TESTED_TENSORFLOW = '2.21.0'
TESTED_KERAS = '3.15.1'


def _parse(version):
    m = re.match(r'(\d+)\.(\d+)(?:\.(\d+))?', str(version))
    return tuple(int(x or 0) for x in m.groups()) if m else None


def _text(v):
    return '.'.join(str(x) for x in v)


def check_tf_keras(tf_version, keras_version, stream=None, exit_on_error=True):
    """True when the versions are fine. Too old: explains and exits (unless
    BEN_SKIP_VERSION_CHECK=1). Newer than tested: warns and carries on."""
    stream = stream or sys.stderr
    tf_v, keras_v = _parse(tf_version), _parse(keras_version)

    if tf_v is None or keras_v is None or tf_v[:2] < MIN_TENSORFLOW or keras_v[:2] < MIN_KERAS:
        message = (
            f"\nThis BEN needs TensorFlow {_text(MIN_TENSORFLOW)} or newer and Keras {_text(MIN_KERAS)} or newer,\n"
            f"but this Python has TensorFlow {tf_version} and Keras {keras_version}.\n"
            f"The play and opening-lead models are saved with Keras {_text(MIN_KERAS)} and cannot be loaded\n"
            f"by an older Keras (it fails with \"could not be deserialized properly\").\n"
            f"Update the packages from the BEN folder:\n"
            f"    pip install -r requirements.txt\n"
            f"Tested with TensorFlow {TESTED_TENSORFLOW} and Keras {TESTED_KERAS}.\n")
        if os.environ.get('BEN_SKIP_VERSION_CHECK') == '1':
            stream.write(message + "Continuing anyway because BEN_SKIP_VERSION_CHECK=1.\n\n")
            return False
        stream.write(message + "To start anyway (e.g. with a configuration that uses only older models), "
                               "set BEN_SKIP_VERSION_CHECK=1.\n\n")
        if exit_on_error:
            sys.exit(1)
        return False

    if tf_v[:2] > _parse(TESTED_TENSORFLOW)[:2] or keras_v[:2] > _parse(TESTED_KERAS)[:2]:
        stream.write(
            f"Warning: BEN is tested with TensorFlow {TESTED_TENSORFLOW} and Keras {TESTED_KERAS}; this Python has "
            f"TensorFlow {tf_version} and Keras {keras_version}. A newer Keras has changed model results before "
            f"without any error -- if play or leads look wrong, install the tested versions "
            f"(pip install -r requirements.txt).\n")
    return True
