# Structure-Based Drug Design via Atom-Motif Consistency

This repository is the official implementation of Structure-Based Drug Design via Atom-Motif Consistency.

# Install environment

####  install via Pip and Conda

```
conda create --name amc_diff python=3.7.16
conda activate amc_diff
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install torch-geometric
conda install rdkit openbabel tensorboard pyyaml easydict python-lmdb -c conda-forge
pip install tqdm matplotlib networkx biopython e3nn giotto-tda
pip install meeko==0.1.dev3 scipy pdb2pqr vina==1.2.2 
python -m pip install git+https://github.com/Valdes-Tresanco-MS/AutoDockTools_py3
```

# Datasets

1. Download the processed dataset form this [link](https://drive.google.com/file/d/1kjp3uLft4t6M62HgSAakiT7BnkQaSRvf/view?usp=sharing).

2. Download CrossDocked2020 v1.1 from [here](https://bits.csb.pitt.edu/files/crossdock2020/), If you want to process the dataset from scratch. Run data_process to process the data.

   ```
   python datasets/data_process.py
   ```

# Training

```
python train_amc_diff.py
```

Download trained model checkpoint



# Sampling

#### Sampling for proteins in the testset

```
python evaluation/sample_amc_diff.py
```

#### Sampling for pdb file

```
python evaluation/sample_for_pdb.py
```

# Evaluation

#### Evaluation from sampling results

```
python evaluation/evaluate_amc_diff.py
python evaluation/evaluate_for_pdb.py
```


