import pandas as pd
from tqdm import tqdm
import os
import torch
import torch.nn as nn
from torchmetrics import MeanSquaredError, R2Score, SpearmanCorrCoef, PearsonCorrCoef
from omegaconf import OmegaConf
from ddgbench import ddgBench
import sys
sys.path.append('../')
from datasets import MegaScaleDataset, FireProtDataset, ddgBenchDataset
from transfer_model_self import get_protein_mpnn
from transfer_model_self import TransferModelPL
from protein_mpnn_utils import tied_featurize
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        "r2": R2Score().to('cuda'),
        "mse": MeanSquaredError(squared=True).to('cuda'),
        "rmse": MeanSquaredError(squared=False).to('cuda'),
        "spearman": SpearmanCorrCoef().to('cuda'),
        "pearson":  PearsonCorrCoef().to('cuda'),
    }


def get_trained_model(model_name, config, checkpt_dir='models/', override_custom=False):
    if override_custom:
        return TransferModelPL.load_from_checkpoint(model_name, cfg=config).model
    else:
        model_loc = os.path.join(config.platform.thermompnn_dir, checkpt_dir)
        model_loc = os.path.join(model_loc, model_name)
        return TransferModelPL.load_from_checkpoint(model_loc, cfg=config).model



def run_prediction_default(name, model, dataset_name, dataset, results):
    """Standard inference for CSV/PDB based dataset"""

    max_batches = None
    protein_metrics = []
    all_predictions = []
    print('Testing Model %s on dataset %s' % (name, dataset_name))

    for i, batch in enumerate(tqdm(dataset)):
        pdb, mutations = batch
        pred  = model(pdb, mutations)
        protein_name = mutations[0]
        metrics = get_metrics()

        for mut, out in zip(mutations, pred):
            if mut.ddG is not None:
                mut.ddG = mut.ddG.to('cuda')
                for metric in metrics.values():
                    metric.update(out["ddG"], mut.ddG)

                prediction_record = {
                    'Model': name,
                    'Dataset': dataset_name,
                    'Protein': protein_name,
                    'Position': mut.position,
                    'Wildtype': mut.wildtype,
                    'Mutation': mut.mutation,
                    'ddG_true': mut.ddG.cpu().item(),
                    'ddG_pred': out["ddG"].cpu().item()
                }
                all_predictions.append(prediction_record)

        protein_result = {
            "Model": name,
            "Dataset": dataset_name,
            "Protein": protein_name
        }
        for met_name, metric in metrics.items():
            try:
                protein_result[f"ddG {met_name}"] = metric.compute().cpu().item()
            except ValueError:
                protein_result[f"ddG {met_name}"] = float('nan')

        protein_metrics.append(protein_result)
        import pandas as pd
        df_protein = pd.DataFrame(protein_metrics)
        save_filename = f"{name}_{dataset_name}_per_protein_metrics.csv"
        df_protein.to_csv(save_filename, index=False)
        print(f"已保存每个蛋白的指标到：{save_filename}")



        if max_batches is not None and i >= max_batches:
            break

    if all_predictions:
        df_predictions = pd.DataFrame(all_predictions)
        predictions_filename = f"{name}_{dataset_name}_all_predictions.csv"
        df_predictions.to_csv(predictions_filename, index=False)
        print(f"已保存所有突变的详细预测结果到：{predictions_filename}")
        print(f"  - 包含列：{list(df_predictions.columns)}")
        print(f"  - 总突变数：{len(df_predictions)}")

    avg_metrics = {
        "Model": name,
        "Dataset": dataset_name,
        "Aggregation": "Mean Across Proteins"
    }
    metric_keys = [key for key in protein_metrics[0].keys() if key.startswith("ddG")]
    for key in metric_keys:
        values = [p[key] for p in protein_metrics if not pd.isna(p[key])]
        if values:
            avg_metrics[key] = sum(values) / len(values)
        else:
            avg_metrics[key] = float('nan')

    results.append(avg_metrics)

    median_metrics = {
        "Model": name,
        "Dataset": dataset_name,
        "Aggregation": "Median Across Proteins"
    }
    for key in metric_keys:
        values = [p[key] for p in protein_metrics if not pd.isna(p[key])]
        if values:
            sorted_vals = sorted(values)
            n = len(sorted_vals)
            if n % 2 == 1:
                median_value = sorted_vals[n // 2]
            else:
                median_value = sum(sorted_vals[n // 2 - 1:n // 2 + 1]) / 2
            median_metrics[key] = median_value
        else:
            median_metrics[key] = float('nan')

    results.append(median_metrics)

    return results


def run_prediction_keep_preds(name, model, dataset_name, dataset, results, centrality=False):
    """Inference for CSV/PDB based dataset saving raw predictions for later analysis."""
    row = 0
    max_batches = None
    raw_pred_df = pd.DataFrame(
        columns=['WT Seq', 'Model', 'Dataset', 'ddG_true', 'ddG_pred', 'position', 'wildtype', 'mutation',
                 'neighbors', 'best_AA'])
    protein_metrics = []
    all_predictions = []
    print('Running model %s on dataset %s' % (name, dataset_name))
    for i, batch in enumerate(tqdm(dataset)):
        mut_pdb, mutations = batch
        pred, _ = model(mut_pdb, mutations)

        metrics = get_metrics()

        if centrality:
            coord_chain = [c for c in mut_pdb[0].keys() if 'coords' in c][0]
            chain = coord_chain[-1]
            neighbors = compute_centrality(mut_pdb[0][coord_chain], basis_atom='CA', backup_atom='C', chain=chain,
                                           radius=10.)

        for mut, out in zip(mutations, pred):
            if mut.ddG is not None:
                mut.ddG = mut.ddG.to('cuda')
                for metric in metrics.values():
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
            if 'Megascale' not in dataset_name:  # different pdb column formatting
                key = mut.pdb
            else:
                key = mut.pdb + '.pdb'
            if 'S669' not in dataset_name:  # S669 is missing WT seq info - omit to prevent error
                raw_pred_df.loc[row, 'WT Seq'] = dataset.wt_seqs[key]
            row += 1

        protein_result = {
            "Model": name,
            "Dataset": dataset_name,
            "Protein": i
        }
        for met_name, metric in metrics.items():
            try:
                protein_result[f"ddG {met_name}"] = metric.compute().cpu().item()
            except ValueError:
                protein_result[f"ddG {met_name}"] = float('nan')

        protein_metrics.append(protein_result)

        if max_batches is not None and i >= max_batches:
            break

    avg_metrics = {
        "Model": name,
        "Dataset": dataset_name,
        "Aggregation": "Mean Across Proteins"
    }
    metric_keys = [key for key in protein_metrics[0].keys() if key.startswith("ddG")]
    for key in metric_keys:
        values = [p[key] for p in protein_metrics if not pd.isna(p[key])]
        if values:
            avg_metrics[key] = sum(values) / len(values)
        else:
            avg_metrics[key] = float('nan')

    results.append(avg_metrics)
    raw_pred_df.to_csv(name + '_' + dataset_name + "_raw_preds.csv")
    del raw_pred_df

    return results


def main(cfg, args):

    # define config for model loading
    config = {
        'training': {
            'num_workers': 8,
            'learn_rate': 0.001,
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

    models = {
        #'ProteinMPNN': ProteinMPNNBaseline(cfg, version='v_48_020.pt'),
        #"ThermoMPNN": get_trained_model(model_name='/home/bingxing2/home/scx6a62/zxy/models/thermoMPNN_default.pt',
        #                                config=cfg)
        "SA-MPNN": get_trained_model(model_name='/home/bingxing2/home/scx6a62/zxy/code/best_model/best_sa_mpnn.ckpt',
                                        config=cfg)

    }
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
                         csv_fname=os.path.join(misc_data_loc, 'data_all/S2648.csv'),
                         dataset_name='S2648'),
         "S783": ddgBench(pdb_dir=os.path.join(misc_data_loc, 'data_all/S783'),
                        csv_fname=os.path.join(misc_data_loc, 'data_all/S783.csv'),
                        dataset_name='S783'),
         "S461": ddgBench(pdb_dir=os.path.join(misc_data_loc, 'data_all/S461'),
                        csv_fname=os.path.join(misc_data_loc, 'data_all/S461.csv'),
                        dataset_name='S461'),
         "S8754": ddgBench(pdb_dir=os.path.join(misc_data_loc, 'data_all/S8754'),
                         csv_fname=os.path.join(misc_data_loc, 'data_all/S8754.csv'),
                         dataset_name='S8754'),
         "S571": ddgBench(pdb_dir=os.path.join(misc_data_loc, 'data_all/S571'),
                        csv_fname=os.path.join(misc_data_loc, 'data_all/S571.csv'),
                        dataset_name='S571'),
         "S4346": ddgBench(pdb_dir=os.path.join(misc_data_loc, 'data_all/S4346'),
                         csv_fname=os.path.join(misc_data_loc, 'data_all/S4346.csv'),
                         dataset_name='S4346'),
         #"p53": ddgBenchDataset(cfg, pdb_dir=os.path.join(misc_data_loc, 'data_all/p53/pdb'),
                                #csv_fname=os.path.join(misc_data_loc, 'data_all/p53.csv')),
    }

    results = []

    for name, model in models.items():
        model = model.eval().to(device)
        model_name = "ThermoMPNN"
        for dataset_name, dataset in datasets.items():
            if args.keep_preds:
                results = run_prediction_keep_preds(model_name,model, dataset_name, dataset, results, centrality=args.centrality)
            else:
                #results = run_prediction_per_protein_metrics(name, model, dataset_name, dataset, results)
                results = run_prediction_default(name, model, dataset_name, dataset, results)
    df = pd.DataFrame(results)
    print(df)
    df.to_csv("SA-MPNN_metrics_ava.csv")


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

