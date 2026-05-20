---

# SA-MPNN: A Sequence-Aware ThermoMPNN for Accurate Prediction of Mutational Effects on Protein Thermodynamic Stability

**SA-MPNN** (Sequence-Aware ThermoMPNN) is a deep learning architecture designed for the highly accurate prediction of protein thermodynamic stability ($\Delta\Delta G$ and $T_m$). By integrating 3D structural constraints (via ProteinMPNN layers) with 1D sequence semantic features (via protein language models like ESM), SA-MPNN captures both global conformational flexibility and local microenvironment changes upon mutation.

## ✨ Key Features

* **Multimodal Architecture:** Seamlessly fuses structural embeddings with sequence-level representations.
* **Rigorous Benchmarking:** Validated across 9 independent datasets including ΔΔG datasets (Megascale, Fireprot-homologue-free, S669, S2648, S571, S4346, and S2648 ) and ∆T_m datasets (S571 and S4346).
* **Dry-Wet Closed-Loop:** Model predictions are strongly correlated with experimental differential scanning fluorimetry (DSF) measurements on the UNcle platform.
* **Robust Metrics:** To ensure precise physical interpretation, the evaluation metrics for AUPRC are strictly calibrated with a threshold of $0.0$ kcal/mol, distinctly separating stabilizing from destabilizing mutations.

## 🚀 Hardware Requirements

The model has been fully optimized for local training and inference.

* **GPU:** A single NVIDIA RTX 4090 (24GB VRAM) is sufficient for both full-scale training and high-throughput inference.
* **OS:** Linux (Ubuntu 20.04/22.04) or Windows Subsystem for Linux (WSL2).

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

## 📊 Dataset Preparation

Pre-processed training and evaluation datasets (including the aligned Megascale and S571 PDB files) should be placed in the `data/` directory.

```text
SA-MPNN/
├── data/
│   ├── train/          # Training sets (Megascale)
│   └── test/           # Benchmark sets (S571, S4346, etc.)
├── weights/            # Pre-trained SA-MPNN checkpoints
├── scripts/            # Evaluation scripts
└── ...

```

## 💻 Usage

### 1. Inference (Predicting $\Delta\Delta G$ for new mutations)

To predict the thermodynamic impact of specific point mutations on a given PDB structure:

```bash
python predict.py \
    --pdb_path ./examples/wildtype.pdb \
    --mutations "A:V600E, A:T315I" \
    --checkpoint ./weights/sampnn_best.pt \
    --output_csv ./results/predictions.csv

```

### 2. Training the Model

To train SA-MPNN from scratch on your custom datasets or reproduce the paper's results.

> **Note on Training Dynamics:** To ensure smoother and more stable convergence, the default training configuration uses a **constant learning rate**. Dynamic learning rate schedulers have been intentionally excluded from the optimal training protocol.

```bash
python train.py \
    --train_data ./data/train/ \
    --val_data ./data/test/S571/ \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --epochs 100 \
    --save_dir ./checkpoints/

```

## 📈 Evaluation

To evaluate a trained model against standard benchmarks and output comprehensive metrics (SCC, PCC, R2, AUPRC):

```bash
python evaluate.py \
    --test_data ./data/test/S571/ \
    --checkpoint ./weights/sampnn_best.pt \
    --auprc_threshold 0.0 

```

## 📝 Citation

If you use SA-MPNN in your research, please cite our upcoming paper in the *Journal of Chemical Information and Modeling (JCIM)*:

```bibtex
@article{sampnn2026,
  title={SA-MPNN: Integrating Sequence Semantic and Structural Features for Robust Prediction of Protein Thermodynamic Stability},
  author={[Your Name] and [Co-first authors] and [Your PI's Name]},
  journal={Journal of Chemical Information and Modeling},
  year={2026},
  note={Submitted/Under Review}
}

```

## 🤝 Acknowledgments

* The structural backbone is inspired by the foundational work of **ProteinMPNN** and **ThermoMPNN**.
* Language model embeddings are derived from the **ESM** architecture.
* Special thanks to the Guoyao Group for their support in the development of this computational pipeline.