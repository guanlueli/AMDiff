import argparse
import os
import shutil
import time
import numpy as np
import torch
from torch_geometric.data import Batch
from torch_geometric.transforms import Compose
from torch_scatter import scatter_sum, scatter_mean
from tqdm.auto import tqdm
import utils.misc as misc
import utils.transforms as trans
from datasets import get_dataset
from datasets.data_load import FOLLOW_BATCH
from models.amc_diff import hier_diff, log_sample_categorical, index_to_log_onehot
from utils.evaluation import atom_num
from datasets.mol_tree import Vocab, MolTree
import pickle

print(os.path.dirname(os.path.abspath(__file__)))

def unbatch_v_traj(ligand_v_traj, n_data, ligand_cum_atoms):
    all_step_v = [[] for _ in range(n_data)]
    for v in ligand_v_traj:  # step_i
        v_array = v.cpu().numpy()
        for k in range(n_data):
            all_step_v[k].append(v_array[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]])
    all_step_v = [np.stack(step_v) for step_v in all_step_v]  # num_samples * [num_steps, num_atoms_i]
    return all_step_v


def sample_diffusion_ligand(model, data, num_samples, batch_size=16, device='cuda:0',
                            num_steps=None, pos_only=False, center_pos_mode='protein',
                            sample_num_atoms='prior', set_center_pos = False, set_atom_num = False, bbox_size = None):

    all_pred_pos, all_pred_v, all_init_ligand_pos = [], [], []
    all_pred_pos_traj, all_pred_v_traj = [], []
    all_pred_v0_traj, all_pred_vt_traj = [], []

    all_pred_pos_motif, all_pred_v_motif, all_init_motif_pos = [], [], []
    all_pred_pos_traj_motif, all_pred_v_traj_motif = [], []
    all_pred_v0_traj_motif, all_pred_vt_traj_motif = [], []

    time_list = []
    num_batch = int(np.ceil(num_samples / batch_size))
    current_i = 15
    data.ligand_ph = torch.tensor(data.ligand_ph)

    for i in tqdm(range(num_batch)):
        n_data = batch_size if i < num_batch - 1 else num_samples - batch_size * (num_batch - 1)
        batch = Batch.from_data_list([data.clone() for _ in range(n_data)], follow_batch=FOLLOW_BATCH).to(device)
        t1 = time.time()
        with torch.no_grad():
            batch_protein = batch.protein_element_batch
            if sample_num_atoms == 'prior':
                pocket_size = atom_num.get_space_size(data.protein_pos.detach().cpu().numpy())
                ligand_num_atoms = [atom_num.sample_atom_num(pocket_size).astype(int) for _ in range(n_data)]
                batch_ligand = torch.repeat_interleave(torch.arange(n_data), torch.tensor(ligand_num_atoms)).to(device)
                ligand_num_motif = [ int(num/5) for num in ligand_num_atoms]
                batch_motif = torch.repeat_interleave(torch.arange(n_data), torch.tensor(ligand_num_motif)).to(device)

            elif sample_num_atoms == 'range':
                ligand_num_atoms = list(range(current_i + 1, current_i + n_data + 1))
                batch_ligand = torch.repeat_interleave(torch.arange(n_data), torch.tensor(ligand_num_atoms)).to(device)

                ligand_num_motif = [int(num / 5) for num in ligand_num_atoms]
                batch_motif = torch.repeat_interleave(torch.arange(n_data), torch.tensor(ligand_num_motif)).to(device)

            elif sample_num_atoms == 'ref':
                batch_ligand = batch.ligand_element_batch
                ligand_num_atoms = scatter_sum(torch.ones_like(batch_ligand), batch_ligand, dim=0).tolist()
            else:
                raise ValueError

            # init ligand pos
            
            pos_in_pocket = batch.protein_pos * batch.is_in_pocket.view(-1, 1)  #
            center_pos = batch.center
            batch_center_pos = center_pos[batch_ligand]
            init_ligand_pos = batch_center_pos + torch.randn_like(batch_center_pos)

            # init motif pos
            batch_center_pos = center_pos[batch_motif]
            init_motif_pos = batch_center_pos + torch.randn_like(batch_center_pos)


            # init ligand v
            if pos_only:
                init_ligand_v = batch.ligand_atom_feature_full
            else:
                uniform_logits = torch.zeros(len(batch_ligand), model.num_classes).to(device)
                init_ligand_v = log_sample_categorical(uniform_logits)

                uniform_logits = torch.zeros(len(batch_motif), model.num_classes_motif).to(device)
                init_motif_v = log_sample_categorical(uniform_logits)

            r = model.sample_diffusion(
                protein_pos=batch.protein_pos,
                protein_v=batch.protein_atom_feature.float(),
                protein_ph = batch.protein_ph,
                batch_protein=batch_protein,
                batch_is_in_pocket=batch.is_in_pocket,

                init_ligand_pos=init_ligand_pos,
                init_ligand_v=init_ligand_v,
                ligand_ph=batch.ligand_ph,
                batch_ligand=batch_ligand,

                init_motif_pos = init_motif_pos,
                init_motif_wid = init_motif_v,
                batch_motif = batch_motif,

                num_steps=num_steps,
                pos_only=pos_only,
                center_pos_mode=center_pos_mode
            )
            ligand_pos, ligand_v, ligand_pos_traj, ligand_v_traj = r['pos'], r['v'], r['pos_traj'], r['v_traj']
            ligand_v0_traj, ligand_vt_traj = r['v0_traj'], r['vt_traj']

            # unbatch pos
            ligand_cum_atoms = np.cumsum([0] + ligand_num_atoms)
            ligand_pos_array = ligand_pos.cpu().numpy().astype(np.float64)
            all_pred_pos += [ligand_pos_array[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]] for k in range(n_data)]  # num_samples * [num_atoms_i, 3]

            init_ligand_array = init_ligand_pos.cpu().numpy().astype(np.float64)
            all_init_ligand_pos += [init_ligand_array[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]] for k in
                             range(n_data)]

            all_step_pos = [[] for _ in range(n_data)]
            for p in ligand_pos_traj:  # step_i
                p_array = p.cpu().numpy().astype(np.float64)
                for k in range(n_data):
                    all_step_pos[k].append(p_array[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]])
            all_step_pos = [np.stack(step_pos) for step_pos in
                            all_step_pos]  # num_samples * [num_steps, num_atoms_i, 3]
            all_pred_pos_traj += [p for p in all_step_pos]

            # unbatch v
            ligand_v_array = ligand_v.cpu().numpy()
            all_pred_v += [ligand_v_array[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]] for k in range(n_data)]

            all_step_v = unbatch_v_traj(ligand_v_traj, n_data, ligand_cum_atoms)
            all_pred_v_traj += [v for v in all_step_v]

            if not pos_only:
                all_step_v0 = unbatch_v_traj(ligand_v0_traj, n_data, ligand_cum_atoms)
                all_pred_v0_traj += [v for v in all_step_v0]
                all_step_vt = unbatch_v_traj(ligand_vt_traj, n_data, ligand_cum_atoms)
                all_pred_vt_traj += [v for v in all_step_vt]

            motif_pos, motif_v, motif_pos_traj, motif_v_traj = r['pos_motif'], r['v_motif'], r['pos_traj_motif'], r['v_traj_motif']
            motif_v0_traj, motif_vt_traj = r['v0_traj_motif'], r['vt_traj_motif']
            # unbatch pos motif
            motif_cum_atoms = np.cumsum([0] + ligand_num_motif)
            motif_pos_array = motif_pos.cpu().numpy().astype(np.float64)
            all_pred_pos_motif += [motif_pos_array[motif_cum_atoms[k]:motif_cum_atoms[k + 1]] for k in range(n_data)]  # num_samples * [num_atoms_i, 3]

            init_motif_array = init_motif_pos.cpu().numpy().astype(np.float64)
            all_init_motif_pos += [init_motif_array[motif_cum_atoms[k]:motif_cum_atoms[k + 1]] for k in range(n_data)]

            all_step_pos_motif = [[] for _ in range(n_data)]
            for p in motif_pos_traj:  # step_i
                p_array = p.cpu().numpy().astype(np.float64)
                for k in range(n_data):
                    all_step_pos_motif[k].append(p_array[motif_cum_atoms[k]:motif_cum_atoms[k + 1]])
            all_step_pos_motif = [np.stack(step_pos) for step_pos in all_step_pos]  # num_samples * [num_steps, num_atoms_i, 3]
            all_pred_pos_traj_motif += [p for p in all_step_pos_motif]

            # unbatch v motif
            motif_v_array = motif_v.cpu().numpy()
            all_pred_v_motif += [motif_v_array[motif_cum_atoms[k]:motif_cum_atoms[k + 1]] for k in range(n_data)]

            all_step_v_motif = unbatch_v_traj(motif_v_traj, n_data, motif_cum_atoms)
            all_pred_v_traj_motif += [v for v in all_step_v_motif]

            if not pos_only:
                all_step_v0_motif = unbatch_v_traj(motif_v0_traj, n_data, motif_cum_atoms)
                all_pred_v0_traj_motif += [v for v in all_step_v0_motif]
                all_step_vt_motif = unbatch_v_traj(motif_vt_traj, n_data, motif_cum_atoms)
                all_pred_vt_traj_motif += [v for v in all_step_vt_motif]

        t2 = time.time()
        time_list.append(t2 - t1)
        current_i += n_data


    return all_pred_pos, all_pred_v, all_pred_pos_traj, all_pred_v_traj, all_pred_v0_traj, all_pred_vt_traj, time_list, all_init_ligand_pos, \
           all_pred_pos_motif, all_pred_v_motif, all_pred_pos_traj_motif, all_pred_v_traj_motif, all_pred_v0_traj_motif, all_pred_vt_traj_motif,  all_init_motif_pos



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,default='/code/configs/sample.yml')
    parser.add_argument('-i', '--data_id', type=int, default=49)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--result_path', type=str, default= '/results/')
    args = parser.parse_args()

    logger = misc.get_logger('sampling')

    # Load config
    config = misc.load_config(args.config)
    logger.info(config)
    misc.seed_all(config.sample.seed)

    # Load checkpoint
    ckpt = {}
    ckpt['model'] = torch.load(config.model.checkpoint, map_location=args.device)

    checkpoint_dir = os.path.dirname(config.model.checkpoint)
    config_path = os.path.join(checkpoint_dir, "config.pt")
    ckpt['config'] = torch.load(config_path, map_location=args.device)

    logger.info(f"Training Config: {ckpt['config']}")

    args.result_path = os.path.join(os.path.dirname(args.result_path),os.path.basename(os.path.dirname(os.path.dirname(config.model.checkpoint))))
    
    print('result_path', args.result_path)

    # Transforms
    protein_featurizer = trans.FeaturizeProteinAtom()
    ligand_atom_mode = ckpt['config'].data.transform.ligand_atom_mode
    ligand_featurizer = trans.FeaturizeLigandAtom(ligand_atom_mode)
    transform = Compose([
        protein_featurizer,
    ])

    # Load dataset
    dataset, subsets = get_dataset(
        config=ckpt['config'].data,
        transform=transform,
        version='final_ph1_pocket_tree_torch_float_v1'
    )
    train_set, test_set = subsets['train'], subsets['test']
    logger.info(f'Successfully load the dataset (size: {len(test_set)})!')

    # Load model
    model = hier_diff(
        ckpt['config'].model,
        protein_atom_feature_dim=protein_featurizer.feature_dim,
        ligand_atom_feature_dim=ligand_featurizer.feature_dim
    ).to(args.device)
    model.load_state_dict(ckpt['model'])
    logger.info(f'Successfully load the model! {config.model.checkpoint}')

    with open('/data/vocab_df_crossdock.pkl', 'rb') as f:
        # Load the data from the file
        vocab_df = pickle.load(f)
    smile_cluster_list = vocab_df['smile_cluster'].tolist()
    vocab = Vocab(smile_cluster_list)

    for i in tqdm(range(0,10)):
        args.data_id = i
        data = test_set[args.data_id]
        data.center = torch.mean(data.ligand_pos, dim=0).unsqueeze(dim=0)
        pred_pos, pred_v, pred_pos_traj, pred_v_traj, pred_v0_traj, pred_vt_traj, time_list, all_init_ligand_pos, \
        pred_pos_motif, pred_v_motif, pred_pos_traj_motif, pred_v_traj_motif, pred_v0_traj_motif, pred_vt_traj_motif, all_init_motif_pos  = sample_diffusion_ligand(
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

        result_path = args.result_path
        os.makedirs(result_path, exist_ok=True)
        shutil.copyfile(args.config, os.path.join(result_path, 'sample.yml'))
        torch.save(result, os.path.join(result_path, f'sample_{args.data_id}.pt'))
        print(os.path.join(result_path, f'result_{args.data_id}.pt'))
