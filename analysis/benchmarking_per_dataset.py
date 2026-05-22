import pandas as pd
from tqdm import tqdm
import os
import torch,argparse
import torch.nn as nn
from torchmetrics import MeanSquaredError, R2Score, SpearmanCorrCoef, PearsonCorrCoef
from omegaconf import OmegaConf
import esm
import sys
from ddgbench import ddgBench
sys.path.append('../')
from datasets import MegaScaleDataset, FireProtDataset, ddgBenchDataset
#from transfer_model_gate import get_protein_mpnn
#from transfer_model_jian import get_protein_mpnn
#from transfer_model_atten import get_protein_mpnn
from transfer_model_self import get_protein_mpnn
from transfer_model_self import TransferModelPL
#from train_thermompnn_atten import TransferModelPL
#torch.backends.cudnn.enabled = False
from protein_mpnn_utils import tied_featurize
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#torch.serialization.add_safe_globals([argparse.Namespace])
ALPHABET = 'ACDEFGHIKLMNPQRSTVWYX'


def compute_centrality(xyz, basis_atom: str = "CA", radius: float = 10.0, core_threshold: int = 20, surface_threshold: int = 15, backup_atom: str = "C", chain: str = 'A') -> torch.Tensor:

    coords = xyz[basis_atom + f'_chain_{chain}']
    coords = torch.tensor(coords)
    # Compute distances and number of neighbors.
    pairwise_dists = torch.cdist(coords, coords)
    pairwise_dists = torch.nan_to_num(pairwise_dists, nan=2 * radius)
    num_neighbors = torch.sum(pairwise_dists < radius, dim=-1) - 1
    # Compute centralities
    # centralities = {
    #     'all': torch.ones(num_neighbors.shape, device=num_neighbors.device),
    #     'core': num_neighbors >= core_threshold,
    #     # 'boundary': num_neighbors < core_threshold & num_neighbors > surface_threshold,
    #     'surface': num_neighbors <= surface_threshold,
    # }
    return num_neighbors


class ProteinMPNNBaseline(nn.Module):
    """Class for running ProteinMPNN as a ddG proxy predictor"""

    def __init__(self, cfg, version='v_48_020.pt'):
        super().__init__()
        self.prot_mpnn = get_protein_mpnn(cfg, version=version)

    def forward(self, pdb, mutations, tied_feat=True):
        device = next(self.parameters()).device
        X, S, mask, lengths, chain_M, chain_encoding_all, chain_list_list, visible_list_list, masked_list_list, masked_chain_length_list_list, chain_M_pos, omit_AA_mask, residue_idx, dihedral_mask, tied_pos_list_of_lists_list, pssm_coef, pssm_bias, pssm_log_odds_all, bias_by_res_all, tied_beta = tied_featurize(
                [pdb[0]], device, None, None, None, None, None, None, ca_only=False)

        *_, log_probs = self.prot_mpnn(X, S, mask, chain_M, residue_idx, chain_encoding_all, None)

        out = []
        for mut in mutations:
            if mut is None:
                out.append(None)
                continue

            aa_index = ALPHABET.index(mut.mutation)
            pred = log_probs[0, mut.position, aa_index]

            out.append({
                "ddG": -torch.unsqueeze(pred, 0),
                "dTm": torch.unsqueeze(pred, 0)
            })
        return out, log_probs


def get_metrics():
    return {
        "r2": R2Score().to(device),
        "mse": MeanSquaredError(squared=True).to(device),
        "rmse": MeanSquaredError(squared=False).to(device),
        "spearman": SpearmanCorrCoef().to(device),
        "pearson":  PearsonCorrCoef().to(device),
    }


def get_trained_model(model_name, config, checkpt_dir='models/', override_custom=False):
    if override_custom:
        #return TransferModelPL.load_from_checkpoint(model_name, cfg=config,strict=False).model
        return TransferModelPL.load_from_checkpoint(model_name, cfg=config).model
    else:
        model_loc = os.path.join(config.platform.thermompnn_dir, checkpt_dir)
        model_loc = os.path.join(model_loc, model_name)
        #return TransferModelPL.load_from_checkpoint(model_loc, cfg=config,strict=False).model
        return TransferModelPL.load_from_checkpoint(model_loc, cfg=config).model


def run_prediction_default(name, model, dataset_name, dataset, results):
    """Standard inference for CSV/PDB based dataset"""

    max_batches = None

    metrics = {
        "ddG": get_metrics(),
    }
    print('Testing Model %s on dataset %s' % (name, dataset_name))

    for i, batch in enumerate(tqdm(dataset)):
        #print(type(batch), len(batch) if hasattr(batch, '__len__') else 'N/A')
        #print(batch.keys() if isinstance(batch, dict) else batch)
        pdb, mutations = batch
        pred_out = model(pdb, mutations)
        #print(type(pred_out), pred_out)
        for mut, out_dict in zip(mutations, pred_out):
            if mut.ddG is not None:
                mut.ddG = mut.ddG.to(device)
                for metric in metrics["ddG"].values():
                    metric.update(out_dict["ddG"], mut.ddG)

        if max_batches is not None and i >= max_batches:
            break
    column = {
        "Model": name,
        "Dataset": dataset_name,
    }
    for dtype in ["ddG"]:
        for met_name, metric in metrics[dtype].items():
            try:
                column[f"{dtype} {met_name}"] = metric.compute().cpu().item()
                #print(column[f"{dtype} {met_name}"])
            except ValueError:
                pass
    results.append(column)
    return results


def run_prediction_keep_preds(name, model, dataset_name, dataset, results, centrality=False):
    """Inference for CSV/PDB based dataset saving raw predictions for later analysis."""
    row = 0
    max_batches = None
    raw_pred_df = pd.DataFrame(
        columns=['WT Seq', 'Model', 'Dataset', 'ddG_true', 'ddG_pred', 'position', 'wildtype', 'mutation',
                 'neighbors', 'best_AA'])
    metrics = {
        "ddG": get_metrics(),
    }
    print('Running model %s on dataset %s' % (name, dataset_name))
    for i, batch in enumerate(tqdm(dataset)):
        #print(type(batch), len(batch) if hasattr(batch, '__len__') else 'N/A')
        #print(batch.keys() if isinstance(batch, dict) else batch)
        mut_pdb, mutations = batch
        pred = model(mut_pdb, mutations)

        if centrality:
            coord_chain = [c for c in mut_pdb[0].keys() if 'coords' in c][0]
            chain = coord_chain[-1]
            neighbors = compute_centrality(mut_pdb[0][coord_chain], basis_atom='CA', backup_atom='C', chain=chain,
                                           radius=10.)

        for mut, out in zip(mutations, pred):
            if mut.ddG is not None:
                mut.ddG = mut.ddG.to(device)
                for metric in metrics["ddG"].values():
                    metric.update(out["ddG"], mut.ddG)

                # assign raw preds and useful details to df
                col_list = ['ddG_true', 'ddG_pred', 'position', 'wildtype', 'mutation', 'pdb']
                val_list = [mut.ddG.cpu().item(), out["ddG"].cpu().item(), mut.position, mut.wildtype,
                            mut.mutation, mut.pdb.strip('.pdb')]
                for col, val in zip(col_list, val_list):
                    raw_pred_df.loc[row, col] = val

                if centrality:
                    raw_pred_df.loc[row, 'neighbors'] = neighbors[mut.position].cpu().item()

            raw_pred_df.loc[row, 'Model'] = name
            raw_pred_df.loc[row, 'Dataset'] = dataset_name
            if 'Megascale' not in dataset_name: # different pdb column formatting
                key = mut.pdb
            else:
                key = mut.pdb + '.pdb'
            if 'S669' not in dataset_name: # S669 is missing WT seq info - omit to prevent error
                raw_pred_df.loc[row, 'WT Seq'] = dataset.wt_seqs[key]
            row += 1

        if max_batches is not None and i >= max_batches:
            break
    column = {
        "Model": name,
        "Dataset": dataset_name,
    }
    for dtype in ["ddG"]:  # , "dTm"]:
        for met_name, metric in metrics[dtype].items():
            try:
                column[f"{dtype} {met_name}"] = metric.compute().cpu().item()
            except ValueError:
                pass
    results.append(column)
    raw_pred_df.to_csv(name + '_' + dataset_name + "_raw_preds.csv")
    del raw_pred_df

    return results


import os
import torch
import pandas as pd
from omegaconf import OmegaConf

def main(cfg, args):
    # 基础配置
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

    # 遍历目录下所有 ckpt 文件
    #ckpt_dir = "/home/bingxing2/home/scx6a62/zxy/checkpoints/no_frozen_seed_42"
    ckpt_files = "/home/bingxing2/home/scx6a62/zxy/code/best_model/best_sa_mpnn.ckpt"

    # 构建数据集
    misc_data_loc = '/home/bingxing2/home/scx6a62/zxy'
    datasets = {
        "Megascale-test": MegaScaleDataset(cfg, csv_file="data_all/testing/mega_test.csv"),
        "Fireprot-test": FireProtDataset(cfg, "test"),
        "Fireprot-homologue-free": FireProtDataset(cfg, "homologue-free"),
        "SSYM_dir": ddgBenchDataset(cfg, pdb_dir=os.path.join(misc_data_loc, 'data_all/PDB/SSYM'),
                                    csv_fname=os.path.join(misc_data_loc, 'data_all/testing/ssym-5fold_clean_dir.csv')),
        "SSYM_inv": ddgBenchDataset(cfg, pdb_dir=os.path.join(misc_data_loc, 'data_all/PDB/SSYM'),
                                    csv_fname=os.path.join(misc_data_loc, 'data_all/testing/ssym-5fold_clean_inv.csv')),
        "S669": ddgBenchDataset(cfg, pdb_dir=os.path.join(misc_data_loc, 'data_all/PDB/S669'),
                                csv_fname=os.path.join(misc_data_loc, 'data_all/testing/s669_clean_dir.csv')),

        "S2648": ddgBench(pdb_dir=os.path.join(misc_data_loc, 'data_all/S2648'),
                                csv_fname=os.path.join(misc_data_loc, 'data_all/S2648.csv'),dataset_name='S2648'),
        "S783": ddgBench(pdb_dir=os.path.join(misc_data_loc, 'data_all/S783'),
                                csv_fname=os.path.join(misc_data_loc, 'data_all/S783.csv'),dataset_name='S783'),
        "S461": ddgBench(pdb_dir=os.path.join(misc_data_loc, 'data_all/S461'),
                                csv_fname=os.path.join(misc_data_loc, 'data_all/S461.csv'),dataset_name='S461'),
        "S8754": ddgBench(pdb_dir=os.path.join(misc_data_loc, 'data_all/S8754'),
                                csv_fname=os.path.join(misc_data_loc, 'data_all/S8754.csv'),dataset_name='S8754'),
        "S571": ddgBench(pdb_dir=os.path.join(misc_data_loc, 'data_all/S571'),
                                csv_fname=os.path.join(misc_data_loc, 'data_all/S571.csv'),dataset_name='S571'),
        "S4346": ddgBench(pdb_dir=os.path.join(misc_data_loc, 'data_all/S4346'),
                                csv_fname=os.path.join(misc_data_loc, 'data_all/S4346.csv'),dataset_name='S4346'),
        
    }

    results = []

    # 遍历每个 ckpt 文件
    for ckpt_path in ckpt_files:
        model_name = os.path.basename(ckpt_path)
        print(f"正在处理模型: {model_name}")

        model = get_trained_model(model_name=ckpt_path, config=cfg, override_custom=True)
        model = model.eval().to(device)

        for dataset_name, dataset in datasets.items():
            if args.keep_preds:
                results = run_prediction_keep_preds(model_name, model, dataset_name, dataset, results, centrality=args.centrality)
            else:
                print(dataset_name, type(dataset))
                results = run_prediction_default(model_name, model, dataset_name, dataset, results)

    # 保存所有结果到一个 CSV
    df = pd.DataFrame(results)
    print(df)
    df.to_csv("seed42_150_no_frozen_seed42).csv", index=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--keep_preds', action='store_true', default=False, help='Save raw model predictions as csv')
    parser.add_argument('--centrality', action='store_true', default=False,
                        help='Calculate centrality value for each residue (# neighbors). '
                             'Only used if --keep_preds is enabled.')

    args = parser.parse_args()
    cfg = OmegaConf.load("./local.yaml")
    with torch.no_grad():
        main(cfg, args)
