import argparse
import os
import numpy as np
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import rdmolfiles
import torch
from tqdm.auto import tqdm
from glob import glob
from collections import Counter
from utils import misc, reconstruct, transforms
from utils.evaluation.docking_vina import VinaDockingTask, pdbqt_to_sdf
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import concurrent.futures
from rdkit.Chem.QED import qed
from utils.evaluation.sascorer import compute_sa_score
from utils.evaluation.MCF import Filter_Mols
from rdkit.Chem import AllChem, Descriptors, Crippen

data_path_base = './data'

def calculate_qed_dockscore(mol, example_idx, sample_idx, ligand_filename, protein_filename,  save_path):

    filename = f'{args.save_path}/{example_idx}_{sample_idx}.mol'
    rdmolfiles.MolToMolFile(mol, filename)

    protein_root = './data/crossdocked_pocket10/'
    exhaustiveness = 32
    r_ligand_filename =  ligand_filename
    protein_filename =   protein_filename
    try:
        vina_task = VinaDockingTask.from_generated_mol(mol, ligand_filename, r_ligand_filename = r_ligand_filename,  protein_path_in = protein_filename, protein_root=protein_root)
        docking_results = vina_task.run(mode='dock', exhaustiveness=exhaustiveness)
        dock_score = docking_results[0]['affinity']

        lig_pdbqt_out = docking_results[0]['lig_pdbqt_out']
        dock_file_name = save_path + f'/{example_idx}_{sample_idx}_dock.sdf'

        pdbqt_to_sdf(pdbqt_file=lig_pdbqt_out, output=dock_file_name, form = 'sdf')
        print('example_idx',example_idx, 'sample_idx', sample_idx, 'done')
    except Exception as e:
        print(e)
        print('example_idx',example_idx, 'sample_idx', sample_idx, 'error')
        dock_score = 0

    try:
        qed_score = round(qed(mol), 3)
    except Exception as e:
        qed_score = np.nan
    try:
        sa_score = round(compute_sa_score(mol), 3)
    except Exception as e:
        sa_score = np.nan
    try:
        logp_score = round(Crippen.MolLogP(mol), 3)
    except Exception as e:
        logp_score = np.nan

    return dock_score, sa_score, qed_score, logp_score

def get_information(results, save_dock_path):

    result_all = []
    results = Filter_Mols(results)
    result = results.iloc[0]
    data_result = calculate_qed_dockscore(result['mol'], 0,0, result['ligand_filename'], result['protein_filename'], save_dock_path)
    #
    with concurrent.futures.ProcessPoolExecutor(max_workers=16) as executor:
        tasks = []
        for index, result in results.iterrows():
            future = executor.submit(calculate_qed_dockscore, result['mol'], result['example_idx'],result['sample_idx'], result['ligand_filename'], result['protein_filename'], save_dock_path)
            tasks.append((future, result))

        for completed_task in concurrent.futures.as_completed([t[0] for t in tasks]):
            for t in tasks:
                if t[0] == completed_task:
                    future, result = t
                    break
            else:
                continue

            dock_score, sa_score, qed_score, logp_score  = completed_task.result()
            try:
                result_all.append({
                    'mol': mol,
                    'smiles': result['smiles'],
                    'example_idx': result['example_idx'],
                    'sample_idx': result['sample_idx'],
                    "dock score": dock_score,
                    "sa score": sa_score,
                    "qed score": qed_score,
                    "logp score": logp_score,
                })
            except Exception as e:
                print(e)

    result_all = pd.DataFrame(result_all)

    return result_all

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_path', type=str, default= './data/outputs_pdb/')
    parser.add_argument('--eval_step', type=int, default=-1)
    parser.add_argument('--eval_num_examples', type=int, default=None)
    parser.add_argument('--save', type=eval, default=True)
    parser.add_argument('--protein_root', type=str, default= './data/ALK')
    parser.add_argument('--atom_enc_mode', type=str, default='add_aromatic')
    args = parser.parse_args()

    args.save_path = args.sample_path + '/reconstruction'
    os.makedirs(args.save_path, exist_ok=True)
    args.save_path_dock = args.sample_path + '/dock'
    os.makedirs(args.save_path_dock, exist_ok=True)

    # Load generated data
    results_fn_list = glob(args.sample_path + '/*sample.pt')
    if args.eval_num_examples is not None:
        results_fn_list = results_fn_list[:args.eval_num_examples]
    num_examples = len(results_fn_list)

    results = []
    all_pair_dist, all_bond_dist = [], []
    all_atom_types = Counter()
    success_pair_dist, success_atom_types = [], Counter()
    for example_idx, r_name in enumerate(tqdm(results_fn_list, desc='Eval')):
        r = torch.load(r_name)
        all_pred_ligand_pos = r['pred_ligand_pos_traj']
        all_pred_ligand_v = r['pred_ligand_v_traj']

        for sample_idx, (pred_pos, pred_v) in enumerate(zip(all_pred_ligand_pos, all_pred_ligand_v)):

            pred_pos, pred_v = pred_pos[args.eval_step], pred_v[args.eval_step]
            pred_atom_type = transforms.get_atomic_number_from_index(pred_v, mode=args.atom_enc_mode)
            try:
                pred_aromatic = transforms.is_aromatic_from_index(pred_v, mode=args.atom_enc_mode)
                mol = reconstruct.reconstruct_from_generated(pred_pos, pred_atom_type, pred_aromatic, drop = True)
                smiles = Chem.MolToSmiles(mol)
            except reconstruct.MolReconsError:
                print('Reconstruct failed %s' % f'{example_idx}_{sample_idx}')
                continue

            results.append({
                'mol': mol,
                'smiles': smiles,
                'ligand_filename': r['data'].ligand_filename,
                'protein_filename': r['data'].protein_filename,
                'pred_pos': pred_pos,
                'pred_v': pred_v,
                'example_idx': example_idx,
                'sample_idx': sample_idx,

            })

    results = pd.DataFrame(results)
    result_all = get_information(results, args.save_path_dock)
    result_all = result_all.sort_values(by="dock score", ascending=True)
    result_all.to_csv(f'{args.sample_path}/data_conclusion.csv', index=True)

    print(f'save to {args.sample_path}/data_conclusion.csv', 'wb')
