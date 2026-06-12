"""
enc_dec.py — Feature Engineered Encoder-Decoder Model
=====================================================
Novel for LOB: Computes high-value LOB features (Spreads, 
Order Imbalances, Mid-Prices) dynamically on the GPU.
Uses a Transformer Encoder to process the sequence and a 
Query-based Decoder (like DETR) to extract the final t0/t1 predictions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import N_FEATURES, N_TARGETS

ENC_DEC_DEFAULTS = {
    "hidden_dim": 128,
    "n_heads": 4,
    "enc_layers": 2,
    "dec_layers": 1,
    "dropout": 0.1,
    # Training config
    "batch_size": 4096,
    "lr": 5e-4,
    "weight_decay": 1e-4,
    "epochs": 4,
    "patience": 7,
    "use_amp": True,
    "use_compile": True,
}

class LOBFeatureEngineer(nn.Module):
    """
    Computes classic high-frequency finance features ON THE FLY.
    This runs entirely on GPU and exports cleanly to ONNX, avoiding
    the need to modify the 100GB parquet datasets.
    """
    def __init__(self):
        super().__init__()
        # 32 raw features -> +8 engineered features = 40 total features
        self.out_dim = N_FEATURES + 8

    def forward(self, x):
        # x is [B, T, 32]
        # Prices: 0-11, Volumes: 12-23
        p = x[..., 0:12]
        v = x[..., 12:24]

        # Assuming standard LOB convention: 
        # p[0:6] = Ask1..Ask6, p[6:12] = Bid1..Bid6
        p_ask1 = p[..., 0]
        p_bid1 = p[..., 6]
        v_ask1 = v[..., 0]
        v_bid1 = v[..., 6]

        # 1-2. Spread and Mid-Price
        spread = p_ask1 - p_bid1
        mid_price = (p_ask1 + p_bid1) / 2.0

        # 3. Level 1 Order Imbalance (OIB)
        # (Bid Vol - Ask Vol) / Total Vol
        oib_l1 = (v_bid1 - v_ask1) / (v_bid1 + v_ask1 + 1e-8)

        # 4. Total Order Imbalance (All levels)
        v_ask_tot = v[..., 0:6].sum(dim=-1)
        v_bid_tot = v[..., 6:12].sum(dim=-1)
        oib_tot = (v_bid_tot - v_ask_tot) / (v_bid_tot + v_ask_tot + 1e-8)

        # 5. Micro-price (Volume-weighted mid price)
        micro_price = (p_bid1 * v_ask1 + p_ask1 * v_bid1) / (v_bid1 + v_ask1 + 1e-8)

        # 6. Price pressure
        # (OIB * Spread) / Mid-Price
        pressure = (oib_l1 * spread) / (mid_price + 1e-8)

        # 7-8. Depth ratio (Ask / Bid depth)
        depth_ask = v[..., 0:6].mean(dim=-1)
        depth_bid = v[..., 6:12].mean(dim=-1)

        # Stack new features
        engineered = torch.stack([
            spread, mid_price, oib_l1, oib_tot, 
            micro_price, pressure, depth_ask, depth_bid
        ], dim=-1)

        # Concat with raw features
        return torch.cat([x, engineered], dim=-1)


class EncDecModel(nn.Module):
    """
    Encoder-Decoder Architecture.
    Encoder: Processes 100-step LOB sequence.
    Decoder: A single learned query token attends to the Encoder 
             output to extract the absolute best signal for t0/t1.
    """
    def __init__(
        self,
        input_dim: int = N_FEATURES,
        hidden_dim: int = 128,
        n_heads: int = 4,
        enc_layers: int = 2,
        dec_layers: int = 1,
        output_dim: int = N_TARGETS,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # 1. Feature Engineering
        self.fe = LOBFeatureEngineer()
        enhanced_dim = self.fe.out_dim

        # 2. Input Projection & Positional Encoding
        self.input_proj = nn.Linear(enhanced_dim, hidden_dim)
        self.pos_emb = nn.Parameter(torch.randn(1, 100, hidden_dim) * 0.02)

        # 3. Transformer Encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=n_heads, 
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=enc_layers)

        # 4. Learned Query Token (The "Decoder")
        # Instead of pooling the sequence, we let a query token look at the sequence
        self.query_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        
        dec_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=dec_layers)

        # 5. Output Head
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, x):
        B, T, _ = x.shape

        # Enhance features on GPU
        x = self.fe(x)

        # Project & add position embeddings
        x = self.input_proj(x)
        x = x + self.pos_emb[:, :T, :]

        # Encode
        memory = self.encoder(x)

        # Decode (Query attends to Memory)
        query = self.query_token.expand(B, -1, -1)
        out = self.decoder(tgt=query, memory=memory)

        # Predict
        out = out.squeeze(1) # [B, Hidden]
        return self.head(out)
