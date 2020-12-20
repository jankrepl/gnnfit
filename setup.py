from setuptools import find_packages, setup

import gnnfit

DESCRIPTION = ""
LONG_DESCRIPTION = DESCRIPTION

INSTALL_REQUIRES = [
    "scikit-learn",
    "torch",
    "tqdm",
]

setup(
    name="gnnfit",
    version=gnnfit.__version__,
    author="Jan Krepl",
    author_email="kjan.official@gmail.com",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    url="https://github.com/jankrepl/gnnfit",
    packages=find_packages(exclude=["tests"]),
    license="MIT",
    install_requires=INSTALL_REQUIRES,
    extras_require={
        "dev": [
            "black",
            "flake8",
            "pydocstyle",
            "pytest",
            "pytest-coverage",
            "tox",
        ],
    },
)
