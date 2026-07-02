---

# SA-MPNN: A Sequence-Aware ThermoMPNN for Accurate Prediction of Mutational Effects on Protein Thermodynamic Stability

**SA-MPNN** (Sequence-Aware ThermoMPNN) is a deep learning architecture designed for the highly accurate prediction of protein thermodynamic stability ($\Delta\Delta G$ and $T_m$). By dynamically integrating 3D structural features with 1D sequence features, SA-MPNN captures both global conformational flexibility and local microenvironment changes upon mutation.https://colab.research.google.com/drive/1rjVDqtd3WYlJ4NWg3fGiH1yYf3P_DKOp?usp=sharing

<img width="1015" height="593" alt="image" src="https://github.com/user-attachments/assets/065fe205-99e0-4564-a8db-8b3ca6d4bcf2" />


## ✨ Key Features

* **Multimodal Architecture:** Seamlessly integrates protein 3D structural features with 1D sequence evolutionary features to improve stability prediction.
* **Rigorous Benchmarking:** Comprehensive evaluation across multiple independent benchmarking datasets, including $\Delta\Delta G$ datasets (Megascale, Fireprot-homologue-free, S669, S2648, S783, SSYM_dir) and $T_m$ datasets (S571 and S4346).
* **Empirical Wet-Lab Validation:** Validated via rigorous wet-lab experiments using the UNcle platform. Among the top 20 predicted single-point mutants, 13 were successfully expressed, and 5 exhibited enhanced thermal stability, with the optimal variant (GC20) achieving a remarkable $\Delta T_m$ of +5.97 °C.
* **Robust Metrics:** To ensure precise physical interpretation, the evaluation metrics for AUPRC are strictly calibrated with a threshold of 0.0 kcal/mol, distinctly separating stabilizing from destabilizing mutations.

## 🛠️ Installation

We highly recommend using Conda/Mamba to manage the Python environment to ensure reproducibility.

```bash
# 1. Clone the repository
git clone [https://github.com/zdf1122/SA-MPNN.git](https://github.com/zdf1122/SA-MPNN.git)
cd SA-MPNN

# 2. Create a dedicated conda environment (Python 3.10)
conda create -n sampnn_env python=3.10 -y
conda activate sampnn_env

# 3. Install PyTorch (Strictly matched to development environment: PyTorch 2.0.0 + CUDA 11.7)
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.7 -c pytorch -c nvidia

# 4. Install other dependencies
pip install -r requirements.txt
```

## 📊 Data and weights Availability

**Download the datasets**
The pre-processed Megascale dataset splits, along with the Fireprot-homologue-free, S669, and SSYM_dir benchmarking datasets, are accessible through the ThermoMPNN data repository at [https://github.com/Kuhlman-Lab/ThermoMPNN]. The independent benchmarking datasets S8754 and S783 were downloaded from the GeoStab ddG data repository ([https://github.com/Gonglab-THU/GeoStab/tree/main/data/ddG]), while S4346 and S571 were obtained from the GeoStab dTm repository ([https://github.com/Gonglab-THU/GeoStab/tree/main/data/dTm]). The S2648 dataset is available on the INPS-MD platform ([https://inpsmd.biocomp.unibo.it/inpsmd/datasets/]).

**Download the Pre-trained Language Model Weights**
SA-MPNN relies on the pre-trained ESM-2 (150M) model as the default to extract sequence semantic features. However, we also support other variant models evaluated in our study. You need to modify the path in the script to point to the location of your downloaded model weights before running the inference script.

* **ESM-2 (150M) [Default]:** [https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t30_150M_UR50D.pt](https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t30_150M_UR50D.pt)
* **ESM-2 (8M):** [https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t6_8M_UR50D.pt](https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t6_8M_UR50D.pt)
* **ESM-2 (650M):** [https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt](https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt)
* **ESM-MSA-1b:** [https://dl.fbaipublicfiles.com/fair-esm/models/esm_msa1b_t12_100M_UR50S.pt](https://dl.fbaipublicfiles.com/fair-esm/models/esm_msa1b_t12_100M_UR50S.pt)
* **E1 Models:** The retrieval-augmented E1 encoder models can be accessed and downloaded via the official Profluent GitHub repository: [https://github.com/Profluent-AI/E1](https://github.com/Profluent-AI/E1).

**Download the pre-trained model weights** 
[https://github.com/SJGLAB/SA-MPNN/releases/tag/v1.0.0]

## 💻 Usage

### Predicting $\Delta\Delta G$ for new mutations

To predict the thermodynamic impact of specific point mutations on a given PDB structure, use the inference.sh script.

```bash
python custom_inference.py \
                --pdb ./examples/2R7E.pdb \
                --chain B \
                --model_path ./best_model/best_sa_mpnn.ckpt \
                --out_dir ./results
```
### `analysis/`
Contains scripts for calculating evaluation metrics on the Megascale test set. It includes tools to compute model performance both at the individual protein level and aggregated across the entire dataset.

## 📝 Citation

If you use SA-MPNN in your research, please cite our upcoming paper in the *Journal of Chemical Information and Modeling (JCIM)*:

```bibtex
@article{sampnn2026,
  title={SA-MPNN: Integrating Sequence Semantic and Structural Features for Robust Prediction of Protein Thermodynamic Stability},
  author={[Xin Yue Zhang] and [Xiang Zheng] and [ Ji Guo Su]},
  journal={Journal of Chemical Information and Modeling},
  year={2026},
  note={Submitted/Under Review}
}

```

## 🤝 Acknowledgments

* The structural backbone is inspired by the foundational work of **ProteinMPNN** and **ThermoMPNN**.
* Language model embeddings are derived from the **ESM** architecture.

## Contact

For any questions, bug reports, or feature requests regarding SA-MPNN, please feel free to reach out:

* **Email:** [xinyuez686@gmail.com](mailto:xinyuez686@gmail.com)
