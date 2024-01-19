from setuptools import setup

setup(
    name="torch3dseg",
    py_modules=["torch3dseg"],
    install_requires=[
        "torch",
        "torchvision",
        "numpy", 
        "tqdm",
        "h5py",
        "pyyaml",
        "tensorboard",
        "scikit-image",
        "torchsummary",
        "nrrd"
        ],
)