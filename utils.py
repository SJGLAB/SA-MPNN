import os
import glob
import json
import numpy as np
import torch
from tqdm import tqdm

# Import structural utilities from protein_mpnn_utils
from protein_mpnn_utils import (
    tied_featurize as mpnn_tied_featurize,
    alt_parse_PDB as mpnn_alt_parse_PDB,
    parse_PDB as mpnn_parse_PDB
)

# Aliased functions
tied_featurize = mpnn_tied_featurize
alt_parse_PDB = mpnn_alt_parse_PDB


def r2_score_manual(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)

    return 1 - ss_res / ss_tot


def parse_single_PDB(path_to_pdb, input_chain_list=None, ca_only=False):
    """
    Parses a single PDB file and returns its name and its dictionary representation.
    """
    parsed_list = mpnn_parse_PDB(path_to_pdb, input_chain_list=input_chain_list, ca_only=ca_only)
    if parsed_list:
        return parsed_list[0]['name'], parsed_list[0]
    return "", {}


def parse_pdb_dir(pdb_files):
    """
    Generator yielding parsed PDB structures for a directory of files.
    """
    for file in pdb_files:
        yield parse_single_PDB(file)


def get_pdb(pdb_data, seq_wt, name_wt, check_assert=True):
    """
    Prepares a protein coordinate and sequence dictionary for downstream stability models.
    Uses protein_mpnn_utils featurizer.
    """
    if check_assert:
        if len(pdb_data["seq"]) != len(seq_wt) or pdb_data["seq"] != seq_wt:
            raise AssertionError("Sequence length or content mismatch between PDB and wild-type sequence.")
    else:
        seq_wt = pdb_data["seq"]

    # Extract target chain identifier
    target_chain = None
    for key in pdb_data.keys():
        if key.startswith("seq_chain_"):
            target_chain = key.split("_")[-1]
            break
            
    if target_chain is None:
        raise ValueError("Could not find chain information in PDB dictionary.")

    chain_coords = pdb_data[f'coords_chain_{target_chain}']
    coords_extracted = {k.split('_')[0]: torch.FloatTensor(v) for k, v in chain_coords.items()}

    # Construct clean profile
    profile = {
        'seq': seq_wt,
        'coords': coords_extracted,
        'name': name_wt,
        'chain_ids': target_chain,
        'mut_ids': [],
        'ddG': [],
        'append_tensors': [],
        'mut_seq': []
    }

    # Clone configuration for standard featurization
    batch_item = {
        f'seq_chain_{target_chain}': seq_wt,
        f'coords_chain_{target_chain}': pdb_data[f'coords_chain_{target_chain}'],
        'seq': seq_wt,
        'name': name_wt,
        'num_of_chains': pdb_data.get('num_of_chains', 1)
    }

    # Perform structural featurization
    feat_output = tied_featurize(
        [batch_item], 
        device='cpu', 
        chain_dict=None, 
        fixed_position_dict=None, 
        omit_AA_dict=None, 
        tied_positions_dict=None, 
        pssm_dict=None, 
        bias_by_res_dict=None, 
        ca_only=False
    )
    
    # Unpack featurized tensors
    X_tensor, S_tensor, mask_tensor, _, M_chain_tensor, enc_tensor, *_, M_pos_tensor, _, _, _, _, _, _, _ = feat_output

    profile['X'] = X_tensor
    profile['S'] = S_tensor
    profile['mask'] = mask_tensor
    profile['chain_M'] = M_chain_tensor
    profile['chain_M_chain_M_pos'] = M_chain_tensor * M_pos_tensor
    profile['residue_idx'] = feat_output[12]
    profile['chain_encoding_all'] = enc_tensor
    profile['randn_1'] = torch.randn(M_chain_tensor.shape, device=X_tensor.device)

    return profile


def parse_pdb(dir_pdb, output_json_path):
    """
    Parses PDB files in a directory and stores them as a JSON database.
    """
    if os.path.exists(output_json_path):
        return

    pattern = os.path.join(dir_pdb, '*.pdb')
    pdb_paths = glob.glob(pattern)
    parsed_database = {}

    for path in tqdm(pdb_paths, desc="Parsing PDB directory"):
        name, item = parse_single_PDB(path)
        if name:
            parsed_database[name] = item

    with open(output_json_path, 'w') as out_f:
        json.dump(parsed_database, out_f)
