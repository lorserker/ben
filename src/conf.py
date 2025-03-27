import os
import sys
import util
from configparser import ConfigParser


def load(fnm: str) -> ConfigParser:
    conf = ConfigParser()
    # If not bundled, use the current script's directory
    base_path = os.path.dirname(os.path.abspath(__file__))
    base_path = base_path.replace("\\_internal", "")

    # Build the full path to the config file
    full_path = os.path.join(base_path, fnm)
    sys.stderr.write("Loading config file %s\n" % full_path)
    with open(full_path) as f:
        conf.read_file(f)
    return conf
