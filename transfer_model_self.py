import torch
import torch.nn as nn
from protein_mpnn_utils import ProteinMPNN, tied_featurize
from model_utils import featurize
import os
import esm
from esm import pretrained

HIDDEN_DIM = 128
EMBED_DIM = 128
VOCAB_DIM = 21
ALPHABET = 'ACDEFGHIKLMNPQRSTVWYX'

MLP = True
SUBTRACT_MUT = True


def get_protein_mpnn(cfg, version='v_48_020.pt'):
    """Loading Pre-trained ProteinMPNN model for structure embeddings"""
    hidden_dim = 128
    num_layers = 3

    model_weight_dir = os.path.join(cfg.platform.thermompnn_dir, 'vanilla_model_weights')
    checkpoint_path = os.path.join(model_weight_dir, version)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model = ProteinMPNN(
        ca_only=False, num_letters=21, node_features=hidden_dim, edge_features=hidden_dim,
        hidden_dim=hidden_dim, num_encoder_layers=num_layers, num_decoder_layers=num_layers,
        k_neighbors=checkpoint['num_edges'], augment_eps=0.0
    )
    if cfg.model.load_pretrained:
        model.load_state_dict(checkpoint['model_state_dict'])

    if cfg.model.freeze_weights:
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

    return model


class TransferModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.hidden_dims = list(cfg.model.hidden_dims)
        self.subtract_mut = cfg.model.subtract_mut
        self.num_final_layers = cfg.model.num_final_layers
        self.lightattn = cfg.model.lightattn if 'lightattn' in cfg.model else False

        ## pre-compute ESM embedding dict when training
        # self.esm_dict_path = getattr(cfg.esm, "dict_path", "/home/bingxing2/home/scx6a62/zxy/esm_embedding_dict_150m.pt")
        # self.esm_embedding_dict = torch.load(self.esm_dict_path)
        # if 'decoding_order' not in self.cfg:
        #     self.cfg.decoding_order = 'left-to-right'

        # ProteinMPNN
        self.prot_mpnn = get_protein_mpnn(cfg)


        self.struct_dim = HIDDEN_DIM * self.num_final_layers + EMBED_DIM
        self.fusion_dim = self.struct_dim  # DIM=384

        # ===== ESM2-150M load and proj =====
        self.use_esm = False
        if hasattr(cfg, 'esm') and cfg.esm is not None:
            self.use_esm = getattr(cfg.esm, 'enabled', False)

        if self.use_esm:
            self.esm_model, self.esm_alphabet = esm.pretrained.load_model_and_alphabet_local(
                "/home/bingxing2/home/scx6a62/zxy/esm2_t30_150M_UR50D.pt"
            )
            self.esm_batch_converter = self.esm_alphabet.get_batch_converter()
            self.esm_model.eval()
            if getattr(cfg.esm, 'freeze', True):
                for p in self.esm_model.parameters():
                    p.requires_grad = False


            self.esm_dim = getattr(self.esm_model, 'embed_dim', None)
            if (self.esm_dim is None) or (getattr(cfg.esm, 'dim_override', None) is not None):
                self.esm_dim = 640
            self.esm_proj = nn.Linear(self.esm_dim, self.fusion_dim)
        else:
            self.esm_proj = None

        # ===== Token-wise Concat Transformer =====
        encoder_layers = getattr(cfg.model, 'token_encoder_layers', 2)
        self.tokenwise_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.fusion_dim,
                nhead=getattr(cfg.model, 'token_encoder_heads', 8),
                dim_feedforward=getattr(cfg.model, 'token_encoder_ffn', 4 * self.fusion_dim),
                dropout=getattr(cfg.model, 'token_encoder_dropout', 0.1),
                batch_first=True
            ),
            num_layers=encoder_layers
        )
        self.struct_ln = nn.LayerNorm(self.fusion_dim)
        self.esm_ln = nn.LayerNorm(self.fusion_dim)

        # ===== LightAttention and MLP =====
        first_in_dim = self.fusion_dim

        if self.lightattn:
            self.light_attention = LightAttention(embeddings_dim=first_in_dim)

        hid_sizes = [first_in_dim] + self.hidden_dims + [VOCAB_DIM]
        self.both_out = nn.Sequential()
        for sz1, sz2 in zip(hid_sizes, hid_sizes[1:]):
            self.both_out.append(nn.ReLU())
            self.both_out.append(nn.Linear(sz1, sz2))

        self.ddg_out = nn.Linear(1, 1)

    def forward(self, pdb, mutations, tied_feat=True):
        device = next(self.parameters()).device

        X, S, mask, lengths, chain_M, chain_encoding_all, chain_list_list, visible_list_list, masked_list_list, masked_chain_length_list_list, chain_M_pos, omit_AA_mask, residue_idx, dihedral_mask, tied_pos_list_of_lists_list, pssm_coef, pssm_bias, pssm_log_odds_all, bias_by_res_all, tied_beta = tied_featurize(
            [pdb[0]], device, None, None, None, None, None, None, ca_only=False
        )

        # === struct tokens ===
        all_mpnn_hid, mpnn_embed, _ = self.prot_mpnn(
            X, S, mask, chain_M, residue_idx, chain_encoding_all, None
        )
        if self.num_final_layers > 0:
            mpnn_hid = torch.cat(all_mpnn_hid[:self.num_final_layers], dim=-1)  # (1, L, HIDDEN_DIM*num_final_layers)
            struct_tokens = torch.cat([mpnn_hid, mpnn_embed], dim=-1)           # (1, L, fusion_dim)
        else:
            struct_tokens = mpnn_embed  # (1, L, EMBED_DIM)

        # === ESM2-150M seq tokens ===
        seq_idx = S[0].detach().cpu().numpy()
        L = int(lengths[0])
        seq_str = ''.join(ALPHABET[i] for i in seq_idx[:L])

        if self.use_esm:
            batch_labels, batch_strs, batch_tokens = self.esm_batch_converter([('protein', seq_str)])
            batch_tokens = batch_tokens.to(device)
            with torch.no_grad():
                layer_id = getattr(self.cfg.esm, 'layer', -1)
                if layer_id == -1:
                    layer_id = self.esm_model.num_layers
                esm_out = self.esm_model(batch_tokens, repr_layers=[layer_id], need_head_weights=False)
            esm_repr = esm_out['representations'][layer_id][:, 1:L + 1, :]  #  remove CLS/EOF
            esm_repr = self.esm_proj(esm_repr)  # (1, L, fusion_dim)
        else:
            esm_repr = torch.zeros_like(struct_tokens)


        ## pre-compute ESM embedding dict when training
        # if self.use_esm:
        #     if seq_str not in self.esm_embedding_dict:
        #         print("seq_str:", seq_str)
        #         print("dict keys sample:", list(self.esm_embedding_dict.keys())[:5])
        #         print("dict value type:", type(self.esm_embedding_dict[list(self.esm_embedding_dict.keys())[0]]))
        #         raise ValueError(f"seq {seq_str} not in dict！")
        #     esm_repr = torch.load(self.esm_embedding_dict[seq_str]).to(device)  # (1, L, fusion_dim)
        #     esm_repr = self.esm_proj(esm_repr)  # proj to fusion_dim

         #===== LayerNorm=====
        struct_tokens = self.struct_ln(struct_tokens)
        esm_repr = self.esm_ln(esm_repr)
        # === Token-wise Concat in seq length_dim  → Transformer ===
        concat_tokens = torch.cat([struct_tokens, esm_repr], dim=1)  # (1, 2L, fusion_dim)
        fused_tokens = self.tokenwise_encoder(concat_tokens)         # (1, 2L, fusion_dim)

        # ====== the effect of fusion ======
        if getattr(self.cfg, "debug_fusion", False):
            with torch.no_grad():
                struct = fused_tokens[:, :L, :]  # (1, L, F)
                seq = fused_tokens[:, L:, :]  # (1, L, F)

                # cosine_similarity between struct token and seq token
                cos = torch.nn.functional.cosine_similarity(struct, seq, dim=-1)  # (1, L)
                print(f"[Fusion Debug] Cosine mean={cos.mean().item():.4f}, "
                      f"min={cos.min().item():.4f}, max={cos.max().item():.4f}")

                # Δstruct_token
                delta = (struct - struct_tokens).norm(dim=-1)  # (1, L)
                print(f"[Fusion Debug] Struct Δ mean={delta.mean().item():.4f}")

        outputs = []

        for mut in mutations:
            if mut is None:
                outputs.append(None)
                continue

            pos = mut.position
            wt_idx = ALPHABET.index(mut.wildtype)
            mt_idx = ALPHABET.index(mut.mutation)


            struct_aa = ALPHABET[S[0, pos].item()]
            esm_aa = seq_str[pos]
            assert struct_aa == esm_aa, f"AA mismatch at pos {pos}: MPNN {struct_aa} vs ESM {esm_aa}"

            fused_struct = fused_tokens[0, pos, :]
            #fused_seq = fused_tokens[0, L + pos, :]

            lin_input = fused_struct

            if self.lightattn:
                mask_pos = mask[:, pos:pos + 1]  # (1, 1)
                lin_input = lin_input.unsqueeze(0).unsqueeze(-1)  # (1, fusion_dim, 1)
                lin_input = self.light_attention(lin_input, mask_pos)  # (fusion_dim,)

            # 21_logits
            both_input = torch.unsqueeze(self.both_out(lin_input), -1)  # (VOCAB_DIM, 1)
            ddg_out = self.ddg_out(both_input)                          # (VOCAB_DIM, 1)

            if self.subtract_mut:
                ddg = ddg_out[mt_idx][0] - ddg_out[wt_idx][0]
            else:
                ddg = ddg_out[mt_idx][0]

            outputs.append({"ddG": torch.unsqueeze(ddg, 0)})

            return outputs

class LightAttention(nn.Module):
    """Source:
    Hannes Stark et al. 2022
    https://github.com/HannesStark/protein-localization/blob/master/models/light_attention.py
    """
    def __init__(self, embeddings_dim=1024, output_dim=11, dropout=0.25, kernel_size=9, conv_dropout: float = 0.25):
        super(LightAttention, self).__init__()
        self.feature_convolution = nn.Conv1d(embeddings_dim, embeddings_dim, kernel_size, stride=1,
                                             padding=kernel_size // 2)
        self.attention_convolution = nn.Conv1d(embeddings_dim, embeddings_dim, kernel_size, stride=1,
                                               padding=kernel_size // 2)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(conv_dropout)


    def forward(self, x: torch.Tensor, mask, **kwargs) -> torch.Tensor:

        o = self.feature_convolution(x)           # [B, D, L]
        o = self.dropout(o)
        attention = self.attention_convolution(x) # [B, D, L]
        o1 = o * self.softmax(attention)
        return torch.squeeze(o1)


import pytorch_lightning as pl


class TransferModelPL(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.model = TransferModel(cfg)

        # === Debug===
        #print("\n===== Trainable Parameters =====")
        #for name, p in self.model.named_parameters():
            #print(f"{name}: requires_grad={p.requires_grad}")
        #print("================================\n")
