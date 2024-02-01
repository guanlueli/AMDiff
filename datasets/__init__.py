import torch
from torch.utils.data import Subset
from .data_process import PocketLigandPairDataset


data_path_base = './data'

def get_dataset(config, version, *args, **kwargs):
    name = 'pl'
    # root = config.path
    root = data_path_base + '/crossdocked_v1.1_rmsd1.0_pocket10'
    config.split = data_path_base + '/crossdocked_pocket10_pose_split_new.pt'
    if name == 'pl':
        dataset = PocketLigandPairDataset(root, *args, **kwargs, version=version)
    else:
        raise NotImplementedError('Unknown dataset: %s' % name)

    if 'split' in config:
        split = torch.load(config.split)
        subsets = {k: Subset(dataset, indices=v) for k, v in split.items()}
        return dataset, subsets
    else:
        return dataset
