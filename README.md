# README

Bayesian optimization through Bayeisan neural network and Thompson sampling

Possible models:

- SVI with reparameterization trick
- MCMC with SGD
- Dropout
- Neural process
- [Variational continual learning](https://github.com/nvcuong/variational-continual-learning/blob/master/ddm/alg/vcl.py)
    - See citations of the original paper
    - Jordan: Streaming Variational Bayes

## Papers to Read

- [Yes, but Did It Work?: Evaluating Variational Inference](https://arxiv.org/pdf/1802.02538.pdf)

## Possible Ideas

- Variational continual learning + PSIS diagnostic
- HMC + Importance sampling
- Kaiming and Xavier prior

## MC-dropout on concreate dataset

- Three hidden layers, with tanh activation function, see `eval_uci.py`.
- Dropout rate, learning rate, precision and length scale are obtained by running Bayesian optimization using the training data of the #19 split. 

```
conf['dropout_rate'] = 0.01
conf['lr']           = 0.1
conf['tau']          = 0.15
conf['lscale']       = 0.25
conf['batch_size']   = 128
conf['num_epochs']   = 1600
conf['num_layers']   = 3
conf['num_hidden']   = 50
```

`RMSE(reported)` and `NLL(reported)` can be seen in `UCI_Datasets/concrete/results`

Train_Test Split ID | RMSE     | NLL      | RMSE(reported)       | NLL(reported)
--------------------|----------|----------|----------------------|----------------------
0                   | **4.57** | **3.02** | 6.244042339856867    | 3.0498930186595428
1                   | **3.68** | **2.69** | 4.732309639246589    | 2.930489207038067
2                   | **3.30** | **2.60** | 3.855345929373324    | 2.7917025469190224
3                   | **4.60** |   2.94   | 5.689284798290305    | **2.9249630654602736**
4                   | **5.23** | **3.07** | 5.822389613706666    | 3.0924028369497027
5                   | **3.92** | **2.71** | 4.217566358545883    | 2.830087106592978
6                   | **4.51** | **2.92** | 6.7181316388370576   | 3.028777661296283
7                   | **4.83** | **2.90** | 4.8773172142394206   | 2.9874813175062247
8                   | **3.72** | **2.74** | 5.178048810289077    | 2.8692172804169984
9                   | **4.30** | **2.84** | 6.092125554083686    | 2.9762331118227525
10                  | **3.52** | **2.55** | 6.115516188361745    | 2.710934497558663
11                  | **4.10** | **2.73** | 5.560704831638099    | 2.8171335896701164
12                  | **4.72** |   3.06   | 5.887968842184838    | **2.882523035425393**
13                  | **3.42** | **2.60** | 4.019714145350288    | 2.7977036086593725
14                  | **4.98** | **3.03** | 6.333958523507762    | 3.0500209788479316
15                  | **4.67** | **2.84** | 5.515987421566104    | 2.9324900927453728
16                  | **3.17** | **2.70** | 4.822574984370559    | 2.9564102950535593
17                  | **4.42** |   2.92   | 4.580797281504998    | **2.8958148206484453**
18                  | **4.85** | **2.93** | 6.426582028890418    | 3.127511896792499
19                  | **4.90** | **2.84** | 6.2763318439970375   | 3.0813276071444693
