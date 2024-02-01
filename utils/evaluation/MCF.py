import pandas as pd
import os
import pickle
import sys
sys.path.append('/home/ligl/project/DVAE/generation1')
sys.path.append('/home/guanlueli/project/DVAE/generation1')

from rdkit.RDLogger import logger
logger = logger()
from rdkit.Chem import FilterCatalog, rdfiltercatalog, rdMolDescriptors
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
from utils.mol_tree import MolTree_process
from collections import Counter
from rdkit import Chem
from tqdm import tqdm


class Filter(object):
    def __init__(self, vocab_path = None):

        if vocab_path is not None:
            with open(vocab_path, 'rb') as f:
                self.vocab_df = pickle.load(f)

    def Pains_structure(self, results):

        params = FilterCatalogParams()
        params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
        catalog = FilterCatalog(params)

        # search for PAINS
        matches = []
        out_results_ind = []

        for index, row in results.iterrows():
            if row['mol'] is not None:
                entry = catalog.GetFirstMatch(row['mol'])
            else:
                print("Error: Invalid molecule object.")
                continue
            if entry is not None:
                # store PAINS information
                matches.append(
                    {
                        "chembl_id": index,
                        "rdkit_molecule": row['mol'],
                        "pains": entry.GetDescription().capitalize(),
                    }
                )
            # collect indices of molecules without PAINS
            else:
                out_results_ind.append(index)

        out_results_df = results.loc[out_results_ind]

        return out_results_df

    def get_unique(self, results):

        out_results_ind = []
        unique_molecules = set()
        for index, row in results.iterrows():
            try:
                if row['mol'] is not None:
                    unique_molecules.add(Chem.MolToSmiles(row['mol']))
                    out_results_ind.append(index)
            except Exception as e:
                continue

        out_results_df = results.loc[out_results_ind]

        return out_results_df

    def novocab(self, results):

        out_results_ind, not_results_ind = [], []
        smile_list = self.vocab_df['smile_cluster'].tolist()
        for index, row in results.iterrows():
            try:
                moltree = MolTree_process(row['mol'])
                assert moltree.num_rotatable_bond > 0
                missing_smiles = any(c.smiles not in smile_list for c in moltree.nodes)
                if not missing_smiles:
                    out_results_ind.append(index)
            except Exception as e:
                not_results_ind.append(index)

        out_results_df = results.loc[out_results_ind]

        return out_results_df

    def fused_ring(self, results):

        out_results_ind, not_results_ind = [], []
        for index, row in results.iterrows():

            add_to_clean = False
            ssr = [set(x) for x in Chem.GetSymmSSSR(row['mol'])]
            max_fused_ring = 1
            for i in range(len(ssr) - 1):
                n_fused_ring = 1
                if len(ssr[i]) <= 2:
                    continue
                for j in range(i + 1, len(ssr)):
                    if len(ssr[j]) <= 2:
                        continue
                    inter = ssr[i] & ssr[j]
                    if len(inter) >= 2:
                        n_fused_ring = n_fused_ring + 1
                        merge = ssr[i] | ssr[j]
                        ssr[i] = merge
                        ssr[j] = set()
                if n_fused_ring > max_fused_ring:
                    max_fused_ring = n_fused_ring

            for i in range(len(ssr) - 1):

                if len(ssr[i]) <= 2:
                    continue
                n_fused_ring = 2
                for j in range(i + 1, len(ssr)):
                    if len(ssr[j]) <= 2:
                        continue
                    inter = ssr[i] & ssr[j]
                    if len(inter) >= 2:
                        n_fused_ring = n_fused_ring + 2
                        merge = ssr[i] | ssr[j]
                        ssr[i] = merge
                        ssr[j] = set()
                if n_fused_ring > max_fused_ring:
                    max_fused_ring = n_fused_ring

            if max_fused_ring >= 4:
                add_to_clean = True

            if add_to_clean:
                not_results_ind.append(index)
            else:
                out_results_ind.append(index)

        out_results_df = results.loc[out_results_ind]

        return out_results_df

    def rings_size(self, results):

        out_results_ind, not_results_ind = [], []

        for index, row in results.iterrows():

            try:
                ring_info = row['mol'].GetRingInfo()
            except Exception as e:
                print(e)
                continue
            all_ring_size = Counter([len(r) for r in ring_info.AtomRings()])
            add_to_clean = False

            for ring_size, ring_count in all_ring_size.items():
                if ring_size > 10:
                    add_to_clean = True
                    break
                # elif ring_size < 5 and ring_count > 1:
                #     add_to_clean = True
                #     break

            if add_to_clean:
                not_results_ind.append(index)
            else:
                out_results_ind.append(index)

        out_results_df = results.loc[out_results_ind]

        return out_results_df

    def filter_substructure(self, results, substructure_smiles):

        for structure_smile in substructure_smiles:
            structure_mol = Chem.MolFromSmiles(structure_smile)
        substructures = pd.DataFrame({'smile': substructure_smiles, 'mol': structure_mol})
        out_results_ind, matches = [], []
        for index, row in results.iterrows():
            match = False
            molecule = row['mol']
            for _, substructure in substructures.iterrows():
                if molecule.HasSubstructMatch(substructure.mol):
                    matches.append(index)
                    match = True
                    break
            if not match:
                out_results_ind.append(index)

        out_results_df = results.loc[out_results_ind]

        return out_results_df


def Filter_Mols(results, filter_active_structure=False, filter_fused_ring = False):

    filter = Filter()
    results = filter.Pains_structure(results)
    results = filter.rings_size(results)

    if filter_fused_ring == True:
        results = filter.fused_ring(results)

    if filter_active_structure == True:
        substructure_smiles = ['O=COC=O', 'OC(O)O', 'OCN', 'C=N', 'NCN', 'OCO', 'NCO',
                               'C1=CCC=CC1', 'C1=CC=CCC1', 'C1=CCCC=C1', 'C1CC=CC=C1',
                               'OS(O)(C)O', 'OS(C)(C)O', 'CS(C)(C)O', 'OS(C)(C)O',
                               'OC1OCCCC1', 'C12OCCCC1CCCO2', 'OC12OCCCC1CCC2', 'CNCN(C)C', 'OC1(O)CCCCC1',
                               'OC(O)(C)O', 'CC=C=CC', 'O=CC=O', 'O=S(C)C=O', 'C1=CC=CC=CC1']

        for smile in substructure_smiles:
            results = filter.filter_substructure(results, substructure_smiles=[smile])

    return results