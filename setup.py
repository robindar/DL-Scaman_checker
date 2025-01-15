from setuptools import setup
from src.dl_scaman_checker.common import __version__

setup(
    name="dl-scaman-checker",
    version=__version__,
    url="https://gitlab.com/robindar/dl-scaman_checker",
    author="David A. R. Robin",
    author_email="david.a.r.robin@gmail.com",
    package_dir={"": "src"},
    packages = [ 'dl_scaman_checker', 'dl_scaman_checker.common' ],
    python_requires=">=3.7, <4",
    install_requires=[
        "numpy",
        "packaging",
        ],
    extras_require={
        "dev": ["torch"],
    },
    package_data={
        "": [ "VERSION" ],
    },
)
