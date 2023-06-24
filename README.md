# 3dseg for CT-Data

This reposirory is based on [pytorch-3dunet](https://github.com/wolny/pytorch-3dunet.git) implementation. 


## TODO's

#TODO: [tensorboar3d](https://www.kitware.com/tensorboardplugin3d-visualizing-3d-deep-learning-models-in-tensorboard/)



- [x] [tensorboar3d](https://www.kitware.com/tensorboardplugin3d-visualizing-3d-deep-learning-models-in-tensorboard/)
- [ ] create bigger dataset of volume

## Installation

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


## losses

### dice Loss
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





## support
### virtual enviroment

to create a virtual enviroment you may run
```bash
````
