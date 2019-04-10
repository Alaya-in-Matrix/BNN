# README

Implementation of several Bayeisan neural networks 

Implemented:

- MC dropout
- SGD-HMC
- SVI(from pyro) 
- BBB

## Reproduce reguression results of MC-dropout on UCI datasets

- `keras-RMSE` and `keras-NLL`: Yarin Gal's original [keras implementation](https://github.com/yaringal/DropoutUncertaintyExps)
- Dropout rate fixed to 0.05, tau selected from `test_tau_100_xepochs_1_hidden_layers.txt`
- See `paper_reproduce/MC_dropout.ipynb` for details, 
- Standard deviation instead of standard error is calculated

Algorithm                  | keras-RMSE  | torch-RMSE  | keras-NLL    | torch-NLL
---------------------------|-------------|-------------|--------------|---------------
bostonHousing              | 2.795±0.753 | 2.788±0.860 | 2.378±0.173  | 2.391±0.223
concrete                   | 5.227±0.539 | 5.317±0.572 | 3.047±0.086  | 3.066±0.088
energy                     | 1.032±0.128 | 1.064±0.139 | 1.575±0.056  | 1.596±0.068
kin8nm                     | 0.095±0.003 | 0.095±0.003 | -0.970±0.026 | -0.971±0.028
naval-propulsion-plant     | 0.004±0.000 | 0.004±0.000 | -4.100±0.028 | -4.100±0.019
power-plant                | 4.203±0.153 | 4.187±0.151 | 2.839±0.031  | 2.838±0.031
protein-tertiary-structure | 4.564±0.018 | 4.570±0.017 | 2.938±0.004  | 2.938±0.004
wine-quality-red           | 0.611±0.039 | 0.612±0.040 | 0.916±0.062  | 0.919±0.065
yacht                      | 1.320±0.275 | 1.403±0.255 | 1.555±0.071  | 1.548±0.089
