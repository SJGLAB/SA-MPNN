import pandas as pd
from tqdm import tqdm
import time
import os
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from Bio.PDB import PDBParser

import sys
sys.path.append('./')
from datasets import Mutation
from protein_mpnn_utils import tied_featurize, alt_parse_PDB
from transfer_model_self import TransferModelPL
import torch
print(f"CUDA 可用状态: {torch.cuda.is_available()}")
ALPHABET = 'ACDEFGHIKLMNPQRSTVWYX'


def get_ssm_mutations(pdb):
    # make mutation list for SSM run
    mutation_list = []
    for seq_pos in range(len(pdb['seq'])):
        wtAA = pdb['seq'][seq_pos]
        # check for missing residues
        if wtAA != '-':
            # add each mutation option
            for mutAA in ALPHABET[:-1]:
                mutation_list.append(wtAA + str(seq_pos) + mutAA)
        else:
            mutation_list.append(None)

    return mutation_list


def get_chains(pdb):
  parser = PDBParser(QUIET=True)
  structure = parser.get_structure('', pdb)
  chains = [c.id for c in structure.get_chains()]
  return chains


def main(cfg, args):

    # define config for model loading
    config = {
        'training': {
            'num_workers': 0,
            'learn_rate': 0.0003,
            'epochs': 50,
            'lr_schedule': True,
        },
        'model': {
            'hidden_dims': [64, 32],
            'subtract_mut': True,
            'num_final_layers': 2,
            'freeze_weights': True,
            'load_pretrained': True,
            'lightattn': True,
            'dropout': 0.3
        },
        'esm': {
            'enabled': True,
            'local_dir': "/home/bingxing2/home/scx6a62/zxy/esm2_t30_150M_UR50D.pt",
            'freeze': True,
            'layer': -1,
            'hidden_size': 640
        },
    }
    cfg = OmegaConf.merge(config, cfg)

    # load the chosen model and dataset
    models = {
        #"ThermoMPNN":get_trained_model(model_name=args.model_path,config=cfg, override_custom=True)
        "SA-MPNN":get_trained_model(model_name='./best_model/best_sa_mpnn.ckpt',
        config = cfg, override_custom = True)
    }

    input_pdb = args.pdb
    pdb_id = os.path.basename(input_pdb).rstrip('.pdb')

    datasets = {
        pdb_id: args.pdb
    }

    raw_pred_df = pd.DataFrame(columns=['Model', 'Dataset', 'ddG_pred', 'position', 'wildtype', 'mutation',])
    row = 0
    for name, model in models.items():
        model = model.eval()
        model = model.cuda()
        for dataset_name, dataset in datasets.items():
            if len(args.chain) < 1:  # if unspecified, take first chain
                chain = get_chains(input_pdb)[0]
            else:
                chain = args.chain
            mut_pdb = alt_parse_PDB(input_pdb, chain)
            mutation_list = get_ssm_mutations(mut_pdb[0])
            final_mutation_list = []

            # build into list of Mutation objects
            for n, m in enumerate(mutation_list):
                if m is None:
                    final_mutation_list.append(None)
                    continue
                m = m.strip()  # clear whitespace
                wtAA, position, mutAA = str(m[0]), int(str(m[1:-1])), str(m[-1])

                assert wtAA in ALPHABET, f"Wild type residue {wtAA} invalid, please try again with one of the following options: {ALPHABET}"
                assert mutAA in ALPHABET, f"Wild type residue {mutAA} invalid, please try again with one of the following options: {ALPHABET}"
                mutation_obj = Mutation(position=position, wildtype=wtAA, mutation=mutAA,
                                        ddG=None, pdb=mut_pdb[0]['name'])
                final_mutation_list.append(mutation_obj)

            #pred,_= model(mut_pdb, final_mutation_list)
            pred = model(mut_pdb, final_mutation_list)
            for mut, out in zip(final_mutation_list, pred):
                if mut is not None:
                    col_list = ['ddG_pred', 'position', 'wildtype', 'mutation', 'pdb', 'chain']
                    val_list = [out["ddG"].cpu().item(), mut.position, mut.wildtype,
                                mut.mutation, mut.pdb.strip('.pdb'), chain]
                    for col, val in zip(col_list, val_list):
                        raw_pred_df.loc[row, col] = val

                    raw_pred_df.loc[row, 'Model'] = name
                    raw_pred_df.loc[row, 'Dataset'] = dataset_name
                    row += 1

    print(raw_pred_df)
    if args.out_dir == './':
        args.out_dir = os.getcwd()
    assert os.path.isdir(args.out_dir), print(f"{args.out_dir} is not a valid directory.")
    csv_file = os.path.join(args.out_dir, "SA-MPNN_%s.csv" % pdb_id)
    raw_pred_df.to_csv(csv_file)
    print(f'Saved SA-MPNN output to {csv_file}')

def get_trained_model(model_name, config, checkpt_dir='models/', override_custom=False):
    if override_custom:
        print("if")
        return TransferModelPL.load_from_checkpoint(model_name, cfg=config).model
    else:
        model_loc = os.path.join(config.platform.thermompnn_dir, checkpt_dir)
        model_loc = os.path.join(model_loc, model_name)
        return TransferModelPL.load_from_checkpoint(model_loc, cfg=config).model

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--pdb', type=str, default='', help='Input PDB to use for custom inference')
    parser.add_argument('--chain', type=str, default='A', help='Chain in input PDB to use.')
    parser.add_argument('--model_path', type=str, default='', help='filepath to model to use for inference')
    parser.add_argument('--out_dir', type=str, default='./', help='Output directory in which to save predictions.')

    args = parser.parse_args()
    cfg = OmegaConf.load("./local.yaml")
    if torch.cuda.is_available():
        torch.cuda.synchronize()


    start_time = time.time()
    with torch.no_grad():
        main(cfg, args)

    if torch.cuda.is_available():
        torch.cuda.synchronize()


    end_time = time.time()
    real_time_cost = end_time - start_time

    print(f"\n🚀   纯推理实际耗时 (Wall-clock time): {real_time_cost:.3f} 秒")
