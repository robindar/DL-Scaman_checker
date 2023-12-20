import os

dirname = os.path.dirname
COMMON_DIR = dirname(os.path.abspath(__file__))
ROOT_DIR = dirname(COMMON_DIR)

with open(os.path.join(ROOT_DIR, 'VERSION')) as version_file:
    version = version_file.read().strip()

__version__ = version
