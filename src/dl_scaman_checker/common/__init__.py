import os

dirname = os.path.dirname
COMMON_DIR = dirname(os.path.abspath(__file__))
ROOT_DIR = dirname(COMMON_DIR)

with open(os.path.join(ROOT_DIR, 'VERSION')) as version_file:
    version = version_file.read().strip()

__version__ = version

def check_install():
    done = "\033[92m{}\033[00m" .format("DONE")
    print(f"[{done}] Install ok. Version is v{__version__}")
