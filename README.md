# Faithful Heteroscedasticity

This repository is the official implementation of
''Faithful Heteroscedastic Regression with Neural Networks.'''

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
To ease reproducibility for those without computational biology backgrounds, we provide pre-processed CRISPR-Cas13 efficacy datasets as pickle files.
The pickle file for the publicly available dataset with flow-cytometry-measured Cas13 efficacy scores in HEK293 cells is located at:
```
data/crispr/flow-cytometry.pkl
```
We promise to release the two survival screen datasets by camera-ready.
At that time, we will make our repository publicly available with an MIT license.

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
The following only reproduces results for the one publicly available dataset, which we provide as part of our submission.
We will update these instructions to include the two survival screen datasets when we make this repository publicly available.
```
python3 experiments_crispr.py --dataset flow-cytometry
python3 analysis.py --experiment crispr
```
