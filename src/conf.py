from configparser import ConfigParser


def load(fnm: str) -> ConfigParser:
    conf = ConfigParser()
    with open(fnm) as f:
        conf.read_file(f)
    print(f"Reading configuration from {fnm}")
    return conf
