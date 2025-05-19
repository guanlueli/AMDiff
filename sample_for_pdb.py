import argparse
import os
import shutil
import torch
from torch_geometric.transforms import Compose

import utils.misc as misc
import utils.transforms as trans
from datasets.data_load import ProteinLigandData, torchify_dict
from models.amc_diff import hier_diff, log_sample_categorical, index_to_log_onehot
from sample_amc_diff import sample_diffusion_ligand
from utils.data import PDBProtein, parse_sdf_file
from datasets.mol_tree import Vocab, MolTree
from torch_scatter import scatter_sum, scatter_mean
import pickle
from rdkit import Chem

def pdb_to_pocket_data(pdb_path):
    pocket_dict = PDBProtein(pdb_path).to_dict_atom()
    data = ProteinLigandData.from_protein_ligand_dicts(
        protein_dict=torchify_dict(pocket_dict),
        ligand_dict={
            'element': torch.empty([0, ], dtype=torch.long),
            'pos': torch.empty([0, 3], dtype=torch.float),
            'atom_feature': torch.empty([0, 8], dtype=torch.float),
            'bond_index': torch.empty([2, 0], dtype=torch.long),
            'bond_type': torch.empty([0, ], dtype=torch.long),
        }
    )

    return  data


def pdb_to_pocket_data_ligand(pdb_path, ligand_path,d=4):
    pocket_dict = PDBProtein(pdb_path).to_dict_atom()
    ligand_dict = parse_sdf_file(ligand_path)
    data = ProteinLigandData.from_protein_ligand_dicts(
        protein_dict=torchify_dict(pocket_dict),
        ligand_dict=torchify_dict(ligand_dict),
    )
    data.protein_filename = pdb_path
    data.ligand_filename = ligand_path
    ligand_pos = data.ligand_pos
    protein_pos = data.protein_pos
    distances = torch.cdist(ligand_pos.unsqueeze(0), protein_pos.unsqueeze(0)).squeeze(0)
    is_in_pocket = torch.any(distances < d, dim=0)
    data.is_in_pocket = is_in_pocket
    data.center = torch.mean(ligand_pos, dim=0).unsqueeze(dim=0)
    print(is_in_pocket)
    print('is_in_pocket', is_in_pocket.sum())
    print('ligand_pos', ligand_pos.shape)
    print('protein_pos', protein_pos.shape)

    return data


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default = '/configs/sample.yml')
    parser.add_argument('--pdb_path', type=str, default = '/data/ALK/3lcs/3lcs_protein_nowater.pdb')
    parser.add_argument('--mol_path', type=str, default = '/data/ALK/3lcs/3lcs_ligand.mol2')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--result_path', type=str, default= '/results/')  #
    parser.add_argument('--set_center_pos', type=bool, default=False)  #
    parser.add_argument('--set_atom_num', type=bool, default=False)  #
    args = parser.parse_args()

    logger = misc.get_logger('evaluate')

    # Load config
    config = misc.load_config(args.config)
    logger.info(config)
    # Load checkpoint
    ckpt = {}
    ckpt['model'] = torch.load(config.model.checkpoint, map_location=args.device)

    checkpoint_dir = os.path.dirname(config.model.checkpoint)
    config_path = os.path.join(checkpoint_dir, "config.pt")
    ckpt['config'] = torch.load(config_path, map_location=args.device)

    logger.info(f"Training Config: {ckpt['config']}")

    # Transforms
    protein_featurizer = trans.FeaturizeProteinAtom()
    ligand_atom_mode = ckpt['config'].data.transform.ligand_atom_mode
    ligand_featurizer = trans.FeaturizeLigandAtom(ligand_atom_mode)
    transform = Compose([
        protein_featurizer,
    ])

    # Load model
    model = hier_diff(
        ckpt['config'].model,
        protein_atom_feature_dim=protein_featurizer.feature_dim,
        ligand_atom_feature_dim=ligand_featurizer.feature_dim
    ).to(args.device)
    model.load_state_dict(ckpt['model'], strict=False if 'train_config' in config.model else True)
    logger.info(f'Successfully load the model! {config.model.checkpoint}')

    # Load vocab
    with open( '/data/vocab_df_crossdock.pkl', 'rb') as f:
        # Load the data from the file
        vocab_df = pickle.load(f)
    smile_cluster_list = vocab_df['smile_cluster'].tolist()
    vocab = Vocab(smile_cluster_list)

    # Load pocket
    data = pdb_to_pocket_data_ligand(args.pdb_path, args.mol_path, d=4)
    data.size_vocab = vocab.size()
    data = transform(data)

    args.result_path = os.path.join(args.result_path, os.path.basename(os.path.dirname(os.path.dirname(args.pdb_path))))
    os.makedirs(args.result_path, exist_ok=True)

    seed_list = [0, 32, 64,128,256,2003,2015]
    for seed in seed_list:
        misc.seed_all(seed)
        print('args.result_path', args.result_path)
        print('save_path', os.path.join(args.result_path, '_' + str(seed) + 'sample.pt'))

        pred_pos, pred_v, pred_pos_traj, pred_v_traj, pred_v0_traj, pred_vt_traj, time_list, all_init_ligand_pos , \
        pred_pos_motif, pred_v_motif, pred_pos_traj_motif, pred_v_traj_motif, pred_v0_traj_motif, pred_vt_traj_motif,  all_init_motif_pos \
        = sample_diffusion_ligand(
            model, data, config.sample.num_samples,
            batch_size=args.batch_size, device=args.device,
            num_steps=config.sample.num_steps,
            pos_only=config.sample.pos_only,
            center_pos_mode=config.sample.center_pos_mode,
            sample_num_atoms=config.sample.sample_num_atoms
        )
        result = {
            'data': data,
            'pred_ligand_pos': pred_pos,
            'pred_ligand_v': pred_v,
            'init_ligand_pos': all_init_ligand_pos,
            'pred_ligand_pos_traj': pred_pos_traj,
            'pred_ligand_v_traj': pred_v_traj,
            'pred_motif_pos': pred_pos_motif,
            'pred_motif_v': pred_v_motif,
            'init_motif_pos': all_init_motif_pos,
            'pred_motif_pos_traj': pred_pos_traj_motif,
            'pred_motif_v_traj': pred_v_traj_motif

        }
        logger.info('Sample done!')

        torch.save(result, os.path.join(args.result_path, os.path.splitext(os.path.basename(args.mol_path))[0] + '_' + str(seed) + 'sample.pt'))

    shutil.copyfile(args.config, os.path.join(args.result_path, os.path.splitext(os.path.basename(args.mol_path))[0] + 'sample.yml'))