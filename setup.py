from setuptools import setup

setup(
    name="torch3dseg",
    py_modules=["torch3dseg"],
    install_requires=["torch>=1.4.0+cu92", "tqdm"],
)