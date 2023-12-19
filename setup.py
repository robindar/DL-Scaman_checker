from setuptools import setup

setup(
    name="dls_check",
    version="0.1.0",
    url="https://gitlab.com/robindar/dl-scaman_checker",
    author="David A. R. Robin",
    author_email="david.a.r.robin@gmail.com",
    package_dir={"": "src"},
    packages = [ 'dls_check' ],
    python_requires=">=3.7, <4",
    install_requires=[],
    extras_require={
        "dev": ["torch"],
    },
    package_data={
    },
)
