# LCNet

This is the repo for the following paper: 

Lu, Q., Nguyen, T. T., Zhang, Q., Hasson, U., Griffiths, T. L., Zacks, J. M., Gershman, S. J., & Norman, K. A. (2023). 
**Toward a More Biologically Plausible Neural Network Model of Latent Cause Inference.** 
In arXiv [q-bio.NC]. arXiv. http://arxiv.org/abs/2312.08519 

This repo contains the code for Simulation 1 and Simulation 2. The code for simulation 3 is in a separate repo [here](https://github.com/qihongl/meta-model). 

### to replicate the results: 

For Simulation 1, run the following code under `src`
```sh
python sim-poly.py
```

For Simulation 2, run the following code under `src`
```sh
python sim-csw.py
```

### directory structure
```sh
.
├── LICENSE
├── README.md
└── src
    ├── demo                  
    │   └── ...... 
    ├── model                         # model components 
    │   ├── A2C.py
    │   ├── CGRU.py
    │   ├── CRPLSTM.py
    │   ├── CRPNN.py
    │   ├── ContextRep.py
    │   ├── GRUA2C.py
    │   ├── PECRP.py
    │   ├── PEKDECRP.py
    │   ├── PEKNN.py
    │   ├── PEKNNCRP.py
    │   ├── PETracker.py
    │   ├── ShortCut.py
    │   ├── SimpleContext.py
    │   ├── SimplePETracker.py
    │   ├── TabularShortCut.py
    │   ├── TabularShortCutIS.py
    │   ├── __init__.py
    │   ├── anderson91.py
    │   └── utils.py
    ├── task                         # definition for the tasks
    │   ├── _Polynomial.py           # task used in Simulation 1 
    │   ├── _CSW.py                  # task used in Simulation 2 
    │   ├── _ContextualBandit.py
    │   ├── _MixedCSW.py
    │   ├── _SimpleTwoArmBandit.py
    │   ├── _Waves.py
    │   └── __init__.py
    ├── sim-poly.py                  # code for Simulation 1 
    ├── sim-csw.py                   # code for Simulation 2 - model training 
    ├── sim-mixedcsw.py             
    ├── stats.py                     
    ├── utils.py                     
    ├── vis-group-poly.py            # code for Simulation 2 - result visualization
    └── vis.py

```

