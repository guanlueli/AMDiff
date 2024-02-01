import pandas as pd
import copy
import pickle
from tqdm.auto import tqdm
import numpy as np
import torch
import random
from utils.chemutils import get_clique_mol, tree_decomp, get_mol, get_smiles, set_atommap, get_clique_mol_simple
from utils.mol_tree import MolTree_process
from rdkit.Chem.rdPartialCharges import ComputeGasteigerCharges
import os
import matplotlib.pyplot as plt
from rdkit import Chem
import seaborn as sns
from collections import Counter

def get_slots(smiles):
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    return [(atom.GetSymbol(), atom.GetFormalCharge(), atom.GetTotalNumHs()) for atom in mol.GetAtoms()]


class Vocab(object):

    def __init__(self, smiles_list):
        self.vocab = smiles_list
        self.vmap = {x: i for i, x in enumerate(self.vocab)}
        # self.slots = [get_slots(smiles) for smiles in self.vocab]

    def get_index(self, smiles):
        try:
            index = self.vmap[smiles]
            return index
        except Exception as e:
            return 0

    def get_smiles(self, idx):
        return self.vocab[idx]

    def get_slots(self, idx):
        return copy.deepcopy(self.slots[idx])

    def size(self):
        return len(self.vocab)


class MolTreeNode(object):

    def __init__(self, mol, cmol, clique, cluster_center, atom_indices):
        self.smiles = Chem.MolToSmiles(cmol, canonical=True)
        self.mol = cmol
        self.clique = [x for x in clique]  # copy
        self.cluster_center = cluster_center
        self.atom_indices = atom_indices

        self.neighbors = []
        self.rotatable = False
        if len(self.clique) == 2:
            if mol.GetAtomWithIdx(self.clique[0]).GetDegree() >= 2 and mol.GetAtomWithIdx(
                    self.clique[1]).GetDegree() >= 2:
                self.rotatable = True
        # should restrict to single bond, but double bond is ok

    def add_neighbor(self, nei_node):
        self.neighbors.append(nei_node)

    def recover(self, original_mol):
        clique = []
        clique.extend(self.clique)
        if not self.is_leaf:
            for cidx in self.clique:
                original_mol.GetAtomWithIdx(cidx).SetAtomMapNum(self.nid)

        for nei_node in self.neighbors:
            clique.extend(nei_node.clique)
            if nei_node.is_leaf:  # Leaf node, no need to mark
                continue
            for cidx in nei_node.clique:
                # allow singleton node override the atom mapping
                if cidx not in self.clique or len(nei_node.clique) == 1:
                    atom = original_mol.GetAtomWithIdx(cidx)
                    atom.SetAtomMapNum(nei_node.nid)

        clique = list(set(clique))
        label_mol = get_clique_mol_simple(original_mol, clique)
        self.label = Chem.MolToSmiles(Chem.MolFromSmiles(get_smiles(label_mol)))
        self.label_mol = get_mol(self.label)

        for cidx in clique:
            original_mol.GetAtomWithIdx(cidx).SetAtomMapNum(0)

        return self.label

    def assemble(self):
        # neighbors = [nei for nei in self.neighbors if nei.mol.GetNumAtoms() > 1]
        neighbors = sorted(self.neighbors, key=lambda x: x.mol.GetNumAtoms(), reverse=True)
        # singletons = [nei for nei in self.neighbors if nei.mol.GetNumAtoms() == 1]
        # neighbors = singletons + neighbors

        cands = enum_assemble(self, neighbors)
        if len(cands) > 0:
            self.cands, self.cand_mols, _ = zip(*cands)
            self.cands = list(self.cands)
            self.cand_mols = list(self.cand_mols)
        else:
            self.cands = []
            self.cand_mols = []



class MolTree(object):
    def __init__(self, mol, vocab=None, ligand_path=None):
        try:
            self.smiles = Chem.MolToSmiles(mol)
        except Exception as e:
            supplier = Chem.ForwardSDMolSupplier(ligand_path, removeHs=True)
            for mol in supplier:
                smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
            self.smiles = smiles
            # print(e)

        self.mol = mol
        self.num_rotatable_bond = 0
        self.vocab = vocab
        '''
        # use reference_vocab and threshold to control the size of vocab
        reference_vocab = np.load('./utils/reference.npy', allow_pickle=True).item()
        reference = defaultdict(int)
        for k, v in reference_vocab.items():
            reference[k] = v'''

        # use vanilla tree decomposition for simplicity
        cliques, edges, cluster_centers, atom_indices= tree_decomp(self.mol, reference_vocab=None)
        self.nodes = []
        root = 0

        for i, c in enumerate(cliques):
            cmol = get_clique_mol_simple(self.mol, c)
            node = MolTreeNode(self.mol, cmol, c, cluster_centers[i], atom_indices[i])
            self.nodes.append(node)
            if min(c) == 0:
                root = i

        for node in self.nodes:
            if node.rotatable:
                self.num_rotatable_bond += 1

        for x, y in edges:
            self.nodes[x].add_neighbor(self.nodes[y])
            self.nodes[y].add_neighbor(self.nodes[x])

        if root > 0:
            self.nodes[0], self.nodes[root] = self.nodes[root], self.nodes[0]

        for i, node in enumerate(self.nodes):
            node.nid = i + 1
            node.wid = vocab.get_index(node.smiles)
            '''
            if len(node.neighbors) > 1:  # Leaf node mol is not marked
                set_atommap(node.mol, node.nid)
            node.is_leaf = (len(node.neighbors) == 1)'''
        # assign node IDs to atoms
        atom_cluster_map = {}
        atom_vocab_map = {}
        for i, node in enumerate(self.nodes):
            for atom_idx in node.atom_indices:
                atom_cluster_map[atom_idx] = node.nid
                atom_vocab_map[atom_idx] = node.wid

        # create sorted cluster ID array mapped by atom ID
        n_atoms = self.mol.GetNumAtoms()
        atom_cluster_array = np.zeros(n_atoms, dtype=np.int64)
        for atom_idx, cluster_id in atom_cluster_map.items():
            atom_cluster_array[atom_idx] = cluster_id
            atom_cluster_array[np.argsort(np.arange(n_atoms))]
        self.atom_cluster_array = atom_cluster_array

        atom_vocab_array = np.zeros(n_atoms, dtype=np.int64)
        for atom_idx, cluster_id in atom_cluster_map.items():
            atom_vocab_array[atom_idx] = cluster_id
            atom_vocab_array[np.argsort(np.arange(n_atoms))]
        self.atom_vocab_array = atom_vocab_array
        # every cluster has cluster_id, how to get the cluster_id for every atom in mol?

        self.motif_pos = np.array(cluster_centers, dtype=np.float32)
        self.node_wid = np.array([node.wid for node in self.nodes], dtype=np.int64)


    def size(self):
        return len(self.nodes)

    def recover(self):
        for node in self.nodes:
            node.recover(self.mol)

    def assemble(self):
        for node in self.nodes:
            node.assemble()


def safe_index(l, e):
    """
    Return index of element e in list l. If e is not present, return the last index
    """
    try:
        return l.index(e)
    except:
        return len(l) - 1

def add_motif_feature():

    with open('./data/vocab_df_crossdock_all.pkl', 'rb') as f:
        vocab_df = pickle.load(f)
    # smile_cluster_list = vocab_df['smile_cluster'].tolist()
    # vocab = Vocab(smile_cluster_list)

    all_vocab_feature = []

    for _, row in vocab_df.iterrows():
        mol = row['mol']
        smile_cluster = row['smile_cluster']
        print(row['smile_cluster'])
        ComputeGasteigerCharges(mol)  # they are Nan for 93 molecules in all of PDBbind. We put a 0 in that case.
        try:
            sssr = Chem.GetSSSR(mol)
            num_rings = len(sssr)
            ring_sizes = []
            for ring in sssr:
                ring_size = len(ring)
                ring_sizes.append(ring_size)

            is_in_ring3 = allowable_features['possible_is_in_ring3_list'].index(3 in ring_sizes)
            is_in_ring4 = allowable_features['possible_is_in_ring4_list'].index(4 in ring_sizes)
            is_in_ring5 = allowable_features['possible_is_in_ring5_list'].index(5 in ring_sizes)
            is_in_ring6 = allowable_features['possible_is_in_ring6_list'].index(6 in ring_sizes)
        except Exception as e:
            num_rings = 0
            is_in_ring3 =0
            is_in_ring4 =0
            is_in_ring5 =0
            is_in_ring6 =0

        weight = Chem.Descriptors.MolWt(mol)

        all_vocab_feature.append({
            'mol' : mol,
            'smile_cluster': smile_cluster,
            'num_rings': num_rings,
            'is_in_ring3': is_in_ring3,
            'is_in_ring4': is_in_ring4,
            'is_in_ring5': is_in_ring5,
            'is_in_ring6': is_in_ring6,
            'weight': weight,
        })

    all_vocab_feature = pd.DataFrame(all_vocab_feature)

    # Min-max normalization
    min_weight = all_vocab_feature['weight'].min()
    max_weight = all_vocab_feature['weight'].max()
    all_vocab_feature['weight'] = (all_vocab_feature['weight'] - min_weight) / (max_weight - min_weight)

    print(all_vocab_feature)

    with open('/data/vocab_df_crossdock_all_feature.pkl', 'wb') as f:
        pickle.dump(all_vocab_feature, f)

def get_crossdock_vocab():
    seed = 2022
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    vocab_smile = {}
    vocab_mol = {}
    cnt = 0
    rot = 0
    # index_path = '/raid/ligl/data1/data_ge/crossdocked_v1.1_rmsd1.0_pocket10/index.pkl'
    index_path = '/data/guanlueli/1Data/1Data/data_ge/crossdocked_v1.1_rmsd1.0_pocket10/index.pkl'
    with open(index_path, 'rb') as f:
        index = pickle.load(f)
    for i, (pocket_fn, ligand_fn, rmsd_str) in enumerate(tqdm(index)):
        if pocket_fn is None: continue
        try:
            path = '/data/guanlueli/1Data/1Data/data_ge/crossdocked_v1.1_rmsd1.0_pocket10/' + ligand_fn
            mol = Chem.MolFromMolFile(path, sanitize=False)
            moltree = MolTree_process(mol)
            cnt += 1
            if moltree.num_rotatable_bond > 0:
                rot += 1
        except Exception as e:
            print(e)
            continue

        for c in moltree.nodes:
            smile_cluster = c.smiles
            if smile_cluster not in vocab_smile:
                vocab_smile[smile_cluster] = 1
                vocab_mol[smile_cluster] = c.mol

            else:
                vocab_smile[smile_cluster] += 1

    # vocab_smile = dict(sorted(vocab_smile.items(), key=lambda kv: (kv[1], kv[0]), reverse=True))
    data = [{'smile_cluster': smile_cluster, 'frequence': vocab_smile[smile_cluster], 'mol': vocab_mol[smile_cluster]} for smile_cluster in vocab_smile]
    sorted_data = sorted(data, key=lambda x: x['frequence'], reverse=True)
    vocab_df = pd.DataFrame(sorted_data)

    with open('/data/guanlueli/1Data/1Data/data_ge/vocab_df_crossdock_all.pkl', 'wb') as f:
        pickle.dump(vocab_df, f)

    # for k, v in vocab.items():
    #     filename.write(k + ':' + str(v))
    #     filename.write('\n')
    # filename.close()

    # number of molecules and vocab
    print('Size of the motif vocab:', len(vocab_smile))
    print('Total number of molecules', cnt)
    print('Percent of molecules with rotatable bonds:', rot / cnt)

def count_sp_carbon(mol):

    carbon_atoms = [idx for idx, atom in enumerate(mol.GetAtoms()) if atom.GetAtomicNum() == 6]  # Get
    total_atoms = len(mol.GetAtoms())

    sp1_count = 0
    sp2_count = 0
    sp3_count = 0

    for idx in carbon_atoms:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetHybridization() == Chem.HybridizationType.SP:
            sp1_count += 1
        elif atom.GetHybridization() == Chem.HybridizationType.SP2:
            sp2_count += 1
        elif atom.GetHybridization() == Chem.HybridizationType.SP3:
            sp3_count += 1

    if total_atoms == 0:
        sp1_per, sp2_per, sp3_per = 0,0,0
    else:
        sp1_per = sp1_count / total_atoms
        sp2_per = sp2_count / total_atoms
        sp3_per = sp3_count / total_atoms

    return sp1_per, sp2_per, sp3_per

def calculate_total_atoms2(mol):
    total_atoms = 0

    for atom in mol.GetAtoms():
        total_atoms += atom.GetTotalNumHs() + 1  # Adding the number of hydrogens attached to the atom plus 1 for the atom itself

    return total_atoms

def get_sp_carbon(mols):

    all_sp1_per, all_sp2_per, all_sp3_per = [],[],[]
    for mol in mols:
        sp1_per, sp2_per, sp3_per = count_sp_carbon(mol)
        all_sp1_per.append(sp1_per)
        all_sp2_per.append(sp2_per)
        all_sp3_per.append(sp3_per)

    return all_sp1_per, all_sp2_per, all_sp3_per

def count_atoms(mol):
    atom_counts = Counter()
    for atom in mol.GetAtoms():
        atomic_symbol = atom.GetSymbol()
        atom_counts[atomic_symbol] += 1
    return atom_counts



if __name__ == "__main__":

    add_motif_feature()




