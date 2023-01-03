from setuptools import setup

setup(
    name="pytorch-3dseg",
    py_modules=["pytorch-3dseg"],
    install_requires=["torch", "tqdm"],
)