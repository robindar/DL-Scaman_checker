import os
from packaging.version import Version

dirname = os.path.dirname
COMMON_DIR = dirname(os.path.abspath(__file__))
ROOT_DIR = dirname(COMMON_DIR)

with open(os.path.join(ROOT_DIR, 'VERSION')) as version_file:
    version = version_file.read().strip()

__version__ = version

def check_install(requires=None):
    done = "\033[92m{}\033[00m" .format("DONE")
    print(f"[{done}] Install ok. Version is v{__version__}")

    if requires is None:
        return

    if Version(__version__) < Version(requires):
        msg = f"Requirement >= '{requires}' not satisfied. Please update"
        fail = "\033[91m{}\033[00m" .format("FAIL")
        print(f"[{fail}] {msg}")
