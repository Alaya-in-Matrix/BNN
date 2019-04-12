# README

Implementation of several Bayeisan neural networks 

Implemented:

- MC dropout
- SGD-HMC
- SVI(from pyro) 
- BBB

## Reguression results on UCI datasets

- Standard deviation instead of standard error is calculated
- Dropout models:
    - `keras-RMSE` and `keras-NLL`: Yarin Gal's original [keras implementation of MC-dropout](https://github.com/yaringal/DropoutUncertaintyExps)
    - Dropout rate fixed to 0.05, tau selected from `test_tau_100_xepochs_1_hidden_layers.txt`
    - 100x epochs (4000 epochs for all datasets except for the protein dataset, 2000 epochs for protein dataset)
    - See `paper_reproduce/MC_dropout.ipynb` for details, 
    - The [Reported results](https://github.com/yaringal/DropoutUncertaintyExps/blob/master/readme.md) are much better, as the hyper-parameters are very carefully tuned (different dropout rates for different datasets and train/test splitting)
- BBB: Bayes-by-Backprop
    - Batch size set to 32
    - 10x epochs instead of 100x epochs
    - See `paper_reproduce/BBB.py` for details
- SVI: stochastic variational inference in [pyro](https://github.com/pyro-ppl/pyro)
    - Batch size set to 32
    - 10x epochs instead of 100x epochs
    - See `paper_reproduce/SVI.py` for details

Algorithm                  | keras-RMSE  | Dropout-RMSE | BBB-RMSE    | SVI-RMSE
---------------------------|-------------|--------------|-------------|------------
bostonHousing              | 2.795±0.753 | 2.788±0.860  | 3.313±0.963 | 3.206±0.857
concrete                   | 5.227±0.539 | 5.317±0.572  | 5.789±0.436 | 5.811±0.536
energy                     | 1.032±0.128 | 1.064±0.139  | 0.999±0.494 | 1.118±0.290
kin8nm                     | 0.095±0.003 | 0.095±0.003  | 0.086±0.004 | 0.086±0.003
naval-propulsion-plant     | 0.004±0.000 | 0.004±0.000  | 0.003±0.001 | 0.002±0.000
power-plant                | 4.203±0.153 | 4.187±0.151  | 4.218±0.150 | 4.159±0.169
protein-tertiary-structure | 4.564±0.018 | 4.570±0.017  | 4.684±0.051 | 4.606±0.025
wine-quality-red           | 0.611±0.039 | 0.612±0.040  | 0.641±0.038 | 0.637±0.035
yacht                      | 1.320±0.275 | 1.403±0.255  | 0.968±0.224 | 1.324±0.319


Algorithm                  | keras-NLL    | Dropout-NLL  | BBB-NLL      | SVI-NLL
---------------------------|--------------|--------------|--------------|--------------
bostonHousing              | 2.378±0.173  | 2.391±0.223  | 2.638±0.409  | 2.626 ±0.386
concrete                   | 3.047±0.086  | 3.066±0.088  | 3.260±0.156  | 3.285 ±0.184
energy                     | 1.575±0.056  | 1.596±0.068  | 1.555±0.501  | 1.584 ±0.217
kin8nm                     | -0.970±0.026 | -0.971±0.028 | -0.979±0.076 | -0.965±0.070
naval-propulsion-plant     | -4.100±0.028 | -4.100±0.019 | -4.297±0.107 | -4.377±0.016
power-plant                | 2.839±0.031  | 2.838±0.031  | 2.935±0.069  | 2.915 ±0.075
protein-tertiary-structure | 2.938±0.004  | 2.938±0.004  | 2.965±0.012  | 2.945 ±0.005
wine-quality-red           | 0.916±0.062  | 0.919±0.065  | 0.968±0.064  | 0.970 ±0.061
yacht                      | 1.555±0.071  | 1.548±0.089  | 1.449±0.086  | 1.622 ±0.061
