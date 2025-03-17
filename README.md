# Molecule Generation For Target Protein Binding with Hierarchical Consistency Diffusion Model

This repository is the official implementation of Molecule Generation For Target Protein Binding with Hierarchical Consistency Diffusion Model. The importance of molecular generation in drug discovery cannot be overstated, as it is pivotal for identifying novel therapeutic agents. Molecular structures can be deconstructed into multiple levels of resolution, each offering unique advantages and interaction mechanisms within the binding pocket. To fully harness the potential of molecular generation, **we propose AMDiff, a classifier-free hierarchical diffusion model specifically designed for accurate and interpretable *de novo* ligand design through atom-motif consistency.**


# Install environment

####  Install via Pip and Conda

```
conda create --name AMDiff python=3.7.16
conda activate AMDiff
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install torch-geometric
conda install rdkit openbabel tensorboard pyyaml easydict python-lmdb -c conda-forge
pip install tqdm matplotlib networkx biopython e3nn giotto-tda
pip install meeko==0.1.dev3 scipy pdb2pqr vina==1.2.2 
python -m pip install git+https://github.com/Valdes-Tresanco-MS/AutoDockTools_py3
```


# Datasets

1. The "data" folder contains model training and evaluation datasets, along with case studies focusing on protein kinases ALK (Anaplastic Lymphoma Kinase) and CDK4 (Cyclin-dependent kinase 4). Download the processed dataset form this [link](https://drive.google.com/file/d/1UCK4bGJdmzqp3EKI_pPsM0jYu4oHgV9s/view?usp=drive_link).

2. If you want to process the dataset from scratch, you need to download CrossDocked2020 v1.1 from [here](https://bits.csb.pitt.edu/files/crossdock2020/). Run data_process to process the data.

   ```
   python datasets/data_process.py
   ```

# Training

```
python train_amc_diff.py
```

The trained model checkpoints are available in the 'data' folder and can also be downloaded from here [link](https://drive.google.com/drive/folders/12Xqnm-7YDqfglkCYzdjRHdAcGlt3YYh_?usp=sharing).

The experiments were conducted on a computing cluster with NVIDIA RTX 4090 GPUs, each with 24 GB of memory. The code was executed using Python with PyTorch. The total computation time for training was approximately 16 hours on NVIDIA RTX 4090 GPU. We trained for 600000 steps with batch size 4. We used the Adam optimizer with a start learning of 5.e-4. We also schedule to decay the learning rate exponentially with a factor of 0.6 and a minimum learning rate of 1e-6. The learning rate is decayed if there is no improvement for the validation loss in 10 consecutive evaluations. The model use 5973s for generating 100 valid molecules on average separately.


# Sampling

To generate predictions for proteins in the test set, run "sample_amc_diff.py":

```
python evaluation/sample_amc_diff.py
```

To generate predictions for pdb file, run "sample_for_pdb.py":

```
python evaluation/sample_for_pdb.py
```

# Evaluation

To evaluation the sampling results, set "evaluate_amc_diff.py" and evaluate_for_pdb.py as file to run.

```
python evaluation/evaluate_amc_diff.py

python evaluation/evaluate_for_pdb.py
```
# Generated Molecules of AMDiff

The generated molecular structures of AMdiff are stored in "/data/AMDiff_generated_samples.zip"


