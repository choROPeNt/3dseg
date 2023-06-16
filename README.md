# 3dseg for CT-Data

Thsi reposirory is based on 


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
### DicelLoss

$$\mathcal{D} = \frac{2 \sum_i^N p_i g_i}{\sum_i^Np_i^2+\sum_i^N g_i^2}$$

### BinaryCrossEntropy



### BCEDiceLoss
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
