# 3dseg for CT-Data

This reposirory is based on [pytorch-3dunet](https://github.com/wolny/pytorch-3dunet.git) implementation. 


## TODO's






- [ ] create bigger dataset of volume

## :cd: Installation



### 🔥 PyTorch

This reposirotry is builded with 

#### 


#### macOS
Please not also that 3D-segemtantion is not available on a macOS machine, as pytorch with metal support only supports 4D arrays!



Clone this repository using the terminal:

```bash
git clone https://github.com/choROPeNt/3dseg.git
```

and navigate to it in your terminal. 
```bash
cd 3dseg
```

Then run:

```bash
python -m pip install -e .
```

This should install the `3dseg` python package via PIP in the current active virtual enviroment. How to set up a virtual enviroment please refer to [virtual enviroment section](#virtual-enviroment)





## 3DUnet model

The model is a 3D-UNet 





## losses

### Dice Loss
The DiceLoss $\mathcal{L}_{Dice}$ is defined as following
$$\mathcal{L}_{\text{Dice}} = \frac{2 \sum_i^N p_i g_i}{\sum_i^Np_i^2+\sum_i^N g_i^2}$$



### binary-corss entropy
- [Pytorch - BCEloss](https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html)

$$ \mathcal{L}_{\text{BCE}}(x,y) = \{ l_1,\dots,l_N\}^\top \quad l_N=-w_n \left[ y_n \; \log x_n + (1-y_N)\;\log (1-x_n)\right]$$

$\mathcal{L}_{\text{BCE}}$ over one batch is determined by

$$\mathcal{L}_{\text{BCE}}(x,y) = \left\{
\begin{array}{ll}
\frac{1}{N} \sum_i^N l_i & \text{if reduction = 'mean'} \\
\sum_i^N l_i & \text{if reduction = 'sum'} \\
\end{array}
\right.$$

### BCE-DiceLoss

linear combination of BCE and Dice loss

$$ $$

## training

```
python ./scripts/train.py --config <CONFIG>
```



[b,c,d,w,h] -> [b,3*c,d,w,h]

## support
### virtual enviroment

to create a virtual enviroment named `.venv` you may run
```bash
python -m venv .venv
````

to activate the virtual enviroment please run

- for linux and macos
```bash
source .venv/bin/activate
```

## Support and Advanced Installations
### HDF5 with MPI (macOS)
If you are using a macOS machine and want to access your results faster, it's reccommended to install HDF5 with MPI support.  

make sure you have installed HDF5 with MPI support. Otherwise you can install it using the Homebrew formula

```zsh
brew install hdf5-mpi
```

After a succefull installation you have to set the `PATH` variables for the compiler

```zsh
export CC=mpicc
export HDF5_MPI="ON"
export HDF5_DIR="/path/to/parallel/hdf5"  # If this isn't found by default
```
If you are unsure where the location of your HDF5-MPI binary is, you can find the `HDF5_DIR` with running

```
h5pcc -showconfig
```
In the case you have an 


Install the python package `h5py` with

```zsh
 % python -m pip install --no-binary=h5py --no-cache-dir h5py
```

the options `--no-binary=h5py` and `--no-cache-dir` that `pip` is forced to build the package from source via `wheel`