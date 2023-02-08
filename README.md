# Faithful Heteroscedasticity

This repository is the official implementation of
[''Faithful Heteroscedastic Regression with Neural Networks.''](https://arxiv.org/abs/2212.09184v1)

## Requirements
We use Docker to set up our environment.
We selected [the official TensorFlow docker image for version 2.8.0 w/ GPU support](https://hub.docker.com/layers/tensorflow/tensorflow/2.8.0-gpu/images/sha256-1e03623e335aac1610b1a3cfa6a96cf10156acb095287f9d6031df3980148663?context=explore)
as our starting point.
Running our `Dockerfile` will grab this image and install the additional requirements listed in `requirements.txt`.

## Downloading the Datasets
To download the UCI and VAE datasets, run:
```
python3 datasets.py
```
To ease reproducibility for those without computational biology backgrounds, we provide the three CRISPR-Cas13 efficacy datasets from our manuscript as pickle files:
```
data/crispr/flow-cytometry-HEK293.pkl
data/crispr/survival-screen-A375.pkl
data/crispr/survival-screen-HEK293.pkl
```

## Reproducing Results
Executing the following commands will run our experiments and perform our analyses.
Running the complete set of commands from any of the subsequent subsections will create the `experiments` and  `results` directories.
The former will contain model weights.
The latter will contain the plots and tables from our manuscript.

### Convergence Experiments
```
python3 experiments_convergence.py
python3 analysis.py --experiment convergence --model_class "Normal"
python3 analysis.py --experiment convergence --model_class "Deep Ensemble"
python3 analysis.py --experiment convergence --model_class "Monte Carlo Dropout"
python3 analysis.py --experiment convergence --model_class "Student"
```

### UCI Regression Experiments
```
python3 experiments_uci.py --dataset boston
python3 experiments_uci.py --dataset carbon
python3 experiments_uci.py --dataset concrete
python3 experiments_uci.py --dataset energy
python3 experiments_uci.py --dataset naval
python3 experiments_uci.py --dataset "power plant"
python3 experiments_uci.py --dataset protein
python3 experiments_uci.py --dataset superconductivity
python3 experiments_uci.py --dataset wine-red
python3 experiments_uci.py --dataset wine-white
python3 experiments_uci.py --dataset yacht
python3 analysis.py --experiment uci --model_class "Normal"
python3 analysis.py --experiment uci --model_class "Deep Ensemble"
python3 analysis.py --experiment uci --model_class "Monte Carlo Dropout"
python3 analysis.py --experiment uci --model_class "Student"
```

### VAE Experiments
```
python3 experiments_vae.py --dataset mnist
python3 experiments_vae.py --dataset fashion_mnist
python3 analysis.py --experiment vae
```

### CRISPR-Cas13 Experiments
```
python3 experiments_crispr.py --dataset flow-cytometry-HEK293
python3 experiments_crispr.py --dataset survival-screen-A375
python3 experiments_crispr.py --dataset survival-screen-HEK293
python3 analysis.py --experiment crispr
```
