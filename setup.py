from setuptools import setup

setup(
    name="3dseg",
    py_modules=["3dseg"],
    install_requires=["torch", "tqdm"],
)