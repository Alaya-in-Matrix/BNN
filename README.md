# README

Implementation of several Bayeisan neural networks 

Implemented:

- MC dropout
- SGD-HMC
- SVI(from pyro) 
- BBB

## Reproduce reguression results of MC-dropout on UCI datasets

- `keras-RMSE` and `keras-NLL`: Yarin Gal's original [keras implementation or MC-dropout](https://github.com/yaringal/DropoutUncertaintyExps)
- Dropout rate fixed to 0.05, tau selected from `test_tau_100_xepochs_1_hidden_layers.txt`
- See `paper_reproduce/MC_dropout.ipynb` for details, 
- Standard deviation instead of standard error is calculated
- BBB: Bayes-by-Backprop
    - See `paper_reproduce/BBB.py` for details
    - Batch size set to 32
    - 10x epochs (instead of 100x epochs for dropout models)

Algorithm                  | keras-RMSE  | Dropout-RMSE | BBB-RMSE
---------------------------|-------------|--------------|-------------
bostonHousing              | 2.795±0.753 | 2.788±0.860  | 3.313±0.963 
concrete                   | 5.227±0.539 | 5.317±0.572  | 5.789±0.436 
energy                     | 1.032±0.128 | 1.064±0.139  | 0.999±0.494 
kin8nm                     | 0.095±0.003 | 0.095±0.003  | 0.086±0.004 
naval-propulsion-plant     | 0.004±0.000 | 0.004±0.000  | 0.003±0.001 
power-plant                | 4.203±0.153 | 4.187±0.151  | 4.218±0.150 
protein-tertiary-structure | 4.564±0.018 | 4.570±0.017  | 4.684±0.051 
wine-quality-red           | 0.611±0.039 | 0.612±0.040  | 0.641±0.038 
yacht                      | 1.320±0.275 | 1.403±0.255  | 0.968±0.224 


Algorithm                  | keras-NLL    | Dropout-NLL  | BBB-NLL
---------------------------|--------------|--------------|------------
bostonHousing              | 2.378±0.173  | 2.391±0.223  | 2.638±0.409
concrete                   | 3.047±0.086  | 3.066±0.088  | 3.260±0.156
energy                     | 1.575±0.056  | 1.596±0.068  | 1.555±0.501
kin8nm                     | -0.970±0.026 | -0.971±0.028 | -0.979±0.076
naval-propulsion-plant     | -4.100±0.028 | -4.100±0.019 | -4.297±0.107
power-plant                | 2.839±0.031  | 2.838±0.031  | 2.935±0.069
protein-tertiary-structure | 2.938±0.004  | 2.938±0.004  | 2.965±0.012
wine-quality-red           | 0.916±0.062  | 0.919±0.065  | 0.968±0.064
yacht                      | 1.555±0.071  | 1.548±0.089  | 1.449±0.086
