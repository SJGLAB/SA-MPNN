import os
import math
import torch
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Any
from collections import defaultdict
from tqdm import tqdm
from math import isnan

# Import custom geometry and structure parsing utilities
from utils import get_pdb, alt_parse_PDB

ALPHABET = 'ACDEFGHIKLMNPQRSTVWYX'


@dataclass
class Mutation:
    """
    Represents a single point mutation and its corresponding experimental effect.
    """
    position: int
    wildtype: str
    mutation: str
    ddG: Optional[torch.Tensor] = None
    pdb: Optional[str] = ''


class ddgBench(torch.utils.data.Dataset):
    """
    A PyTorch Dataset for loading structural features and target stability metrics (ddG/dTm)
    for protein mutational scans.
    """
    def __init__(
        self, 
        pdb_dir: str, 
        csv_fname: str, 
        dataset_name: str, 
        stage: str = 'full', 
        mut_seq: bool = False, 
        train_size: float = 1.0
    ):
        self.pdb_dir = pdb_dir
        self.dataset_name = dataset_name
        self.mut_seq = mut_seq
        self.fake_bs = 32 if stage == 'train' else 10000

        # Load mutation database
        dataframe = pd.read_csv(csv_fname)
        dataframe.PDB = dataframe.PDB + dataframe.chain
        self.df = dataframe

        # Determine all unique valid wild-type keys
        self.wt_names = [
            str(name) for name in dataframe.PDB.unique() 
            if str(name) != 'nan'
        ]
        print(f"Loaded {len(self.wt_names)} unique protein structures.")

        # Initialize tracking dictionaries
        self.wt_seqs = {}
        self.mut_rows = {}
        self.json_dataset = defaultdict(lambda: defaultdict(lambda: -1))

        # Group mutation records by PDB ID
        for name in self.wt_names:
            self.mut_rows[name] = dataframe[dataframe.PDB == name].reset_index(drop=True)
            if 'ssym' in self.pdb_dir:
                self.wt_seqs[name] = self.mut_rows[name].SEQ[0]

        record_lengths = [len(self.mut_rows[name]) for name in self.wt_names]
        self.protein_index_list = [np.arange(length) for length in record_lengths]

        # Precompute offsets and indexes if mut_seq is enabled
        if self.mut_seq:
            self._initialize_mutant_sequence_tracking()

    def _initialize_mutant_sequence_tracking(self):
        """
        Calculates tracking indices when include mutant sequence features is enabled.
        """
        self.index_list = []
        self.start_index = []
        self.proteins = [self._get_wt_item(i) for i in tqdm(range(len(self.wt_names)), desc="Preloading structures")]
        
        mut_numbers = [
            len(self.mut_rows[name[:4]]) 
            for name in self.wt_names
        ]
        mut_numbers = [math.ceil(num / self.fake_bs) for num in mut_numbers]
        
        for prot_idx, num_chunks in enumerate(mut_numbers):
            self.index_list += [prot_idx] * num_chunks
            self.start_index += [offset * self.fake_bs for offset in range(num_chunks)]

        self.dataset_len = sum(mut_numbers)
        self.mut_numbers = mut_numbers

    def __len__(self) -> int:
        if self.mut_seq:
            return self.dataset_len
        return len(self.wt_names)

    def _get_wt_item(self, index: int) -> Optional[Tuple[List[Dict[str, Any]], List[Mutation]]]:
        """
        Parses structure and compiles feature tensors for a single wild-type protein.
        """
        wt_name = self.wt_names[index]
        chain_id = wt_name[-1]

        # Strip file extension and chain ID to get raw PDB name
        pdb_base_name = wt_name.split(".pdb")[0]
        mut_data = self.mut_rows[pdb_base_name]
        pdb_base_name = pdb_base_name[:-1]

        # Parse structural coordinates and cache them
        if isinstance(self.json_dataset[pdb_base_name][chain_id], int):
            pdb_path = os.path.join(self.pdb_dir, pdb_base_name + ".pdb")
            pdb_structure = alt_parse_PDB(pdb_path, [chain_id])
            self.json_dataset[pdb_base_name][chain_id] = pdb_structure
            
        pdb = self.json_dataset[pdb_base_name][chain_id]
        if not pdb:
            return None
            
        resn_list = pdb[0]["resn_list"]
        protein = get_pdb(pdb[0], pdb_base_name, pdb_base_name, check_assert=False)
        protein['mut_seq'] = []

        mutations = []

        for _, row in mut_data.iterrows():
            mut_info = row.MUT
            wt_aa, mut_aa = mut_info[0], mut_info[-1]
            
            # Identify mutation position within PDB sequence
            try:
                pos_str = mut_info[1:-1]
                pdb_idx = resn_list.index(pos_str)
            except ValueError:
                # Skip mutations that cannot be indexed (e.g. insertion codes)
                continue

            try:
                assert pdb[0]['seq'][pdb_idx] == wt_aa
            except AssertionError:
                # Alignment fallback: check for sequence gaps
                if 'S669' in self.pdb_dir:
                    gaps = [char for char in pdb[0]['seq'] if char == '-']
                else:
                    gaps = [char for char in pdb[0]['seq'][:pdb_idx + 10] if char == '-']

                if gaps:
                    pdb_idx += len(gaps)
                else:
                    pdb_idx += 1

                if pdb_idx >= len(pdb[0]['seq']) or pdb[0]['seq'][pdb_idx] != wt_aa:
                    continue

            # Generate mutant sequence
            mut_seq_chars = list(pdb[0]['seq'])
            mut_seq_chars[pdb_idx] = mut_aa
            protein['mut_seq'].append(''.join(mut_seq_chars))

            # Compile ddG labels
            if 'DTM' in row:
                ddG = torch.tensor([row.DTM * -1.], dtype=torch.float32)
            else:
                ddG = (
                    None if row.DDG is None or isnan(row.DDG) 
                    else torch.tensor([row.DDG * -1.], dtype=torch.float32)
                )

            # Compile amino acid encoding vectors
            wt_onehot = torch.zeros(21)
            wt_onehot[ALPHABET.index(wt_aa)] = 1.0
            
            mt_onehot = torch.zeros(21)
            mt_onehot[ALPHABET.index(mut_aa)] = 1.0
            
            append_tensor = torch.cat([wt_onehot, mt_onehot]).float()

            protein['ddG'].append(ddG)
            protein['append_tensors'].append(append_tensor)

            mutations.append(Mutation(
                position=pdb_idx,
                wildtype=wt_aa,
                mutation=mut_aa,
                ddG=ddG,
                pdb=wt_name
            ))

        if len(protein['ddG']) == 0:
            return None

        # Expand batch sizes if mut_seq is enabled
        if self.mut_seq:
            self._expand_wildtype_features(protein)

        protein['ddG'] = torch.stack(protein['ddG'])
        protein['append_tensors'] = torch.stack(protein['append_tensors'])
        protein['dataset'] = self.dataset_name

        return pdb, mutations

    def _expand_wildtype_features(self, protein: Dict[str, Any]):
        """
        Helper method to expand sequence and geometry tensors across batch dimension.
        """
        batch_len = len(protein['S'])
        protein['S'] = torch.cat(protein['S'], dim=0).clone()
        protein['X'] = protein['X'].expand(batch_len, -1, -1, -1).clone()
        protein['mask'] = protein['mask'].expand(batch_len, -1).clone()
        protein['chain_M'] = protein['chain_M'].expand(batch_len, -1).clone()
        protein['chain_M_chain_M_pos'] = protein['chain_M_chain_M_pos'].expand(batch_len, -1).clone()
        protein['residue_idx'] = protein['residue_idx'].expand(batch_len, -1).clone()
        protein['chain_encoding_all'] = protein['chain_encoding_all'].expand(batch_len, -1).clone()
        protein['randn_1'] = protein['randn_1'].expand(batch_len, -1).clone()

    def __getitem__(self, index: int) -> Tuple[List[Dict[str, Any]], List[Mutation]]:
        while True:
            protein = self._get_wt_item(index)
            if protein is not None:
                return protein
            index = (index + 1) % len(self)
