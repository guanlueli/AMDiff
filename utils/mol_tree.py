import sys
import pandas as pd
import rdkit
import rdkit.Chem as Chem
import copy
import pickle
from tqdm.auto import tqdm
import numpy as np
import torch
import random
from utils.chemutils1 import get_clique_mol, tree_decomp, get_mol, get_smiles, set_atommap, get_clique_mol_simple
from collections import defaultdict
import os
import matplotlib.pyplot as plt
from rdkit import Chem
# from rdkit.Chem import Draw
from collections import Counter
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from glob import glob

def get_slots(smiles):
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    return [(atom.GetSymbol(), atom.GetFormalCharge(), atom.GetTotalNumHs()) for atom in mol.GetAtoms()]


class Vocab(object):

    def __init__(self, smiles_list):
        self.vocab = smiles_list
        self.vmap = {x: i for i, x in enumerate(self.vocab)}
        # self.slots = [get_slots(smiles) for smiles in self.vocab]

    def get_index(self, smiles):
        return self.vmap[smiles]

    def get_smiles(self, idx):
        return self.vocab[idx]

    def get_slots(self, idx):
        return copy.deepcopy(self.slots[idx])

    def size(self):
        return len(self.vocab)


class MolTreeNode(object):

    def __init__(self, mol, cmol, clique):
        self.smiles = Chem.MolToSmiles(cmol, canonical=True)
        self.mol = cmol
        self.clique = [x for x in clique]  # copy

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


class MolTree_process(object):
    def __init__(self, mol):
        self.smiles = Chem.MolToSmiles(mol)
        self.mol = mol
        self.num_rotatable_bond = 0
        '''
        # use reference_vocab and threshold to control the size of vocab
        reference_vocab = np.load('./utils/reference.npy', allow_pickle=True).item()
        reference = defaultdict(int)
        for k, v in reference_vocab.items():
            reference[k] = v'''

        # use vanilla tree decomposition for simplicity
        cliques, edges = tree_decomp(self.mol, reference_vocab=None)
        self.nodes = []
        root = 0
        for i, c in enumerate(cliques):
            cmol = get_clique_mol_simple(self.mol, c)
            node = MolTreeNode(self.mol, cmol, c)
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
            '''
            if len(node.neighbors) > 1:  # Leaf node mol is not marked
                set_atommap(node.mol, node.nid)
            node.is_leaf = (len(node.neighbors) == 1)'''

    def size(self):
        return len(self.nodes)

    def recover(self):
        for node in self.nodes:
            node.recover(self.mol)

    def assemble(self):
        for node in self.nodes:
            node.assemble()



if __name__ == "__main__":

    MolTree_process()


