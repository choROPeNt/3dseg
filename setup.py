from setuptools import setup

setup(
    name="torch3dseg",
    py_modules=["torch3dseg"],
    install_requires=[
        ## computation packages
        "torch",
        "torchvision",
        "numpy", 
        "scikit-image",
        ## analytics packages
        "tqdm",
        "tensorboard",
        "torchsummary",
        ## data pacakges
        "h5py", 
        "pynrrd",
        "pyyaml", ],
)