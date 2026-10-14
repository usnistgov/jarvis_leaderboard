"""Inference from prepared JARVIS descriptors using the frozen regression model.

``predict_feature_frame`` accepts only ID, weight, 196 numeric descriptors, and
the categorical split marker. It returns predictions in the input row order.
"""
from __future__ import annotations

import functools
import hashlib
import json
import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


CONT = [f"f_{i:03d}" for i in range(196)]
REQUIRED_COLUMNS = ["sample_id", "weight", *CONT, "split_name"]
ASSET_HASHES = {
    "RAW_fold1.pth": "23ac6a5850d740de277e88e3726af31e588f868504099895350e0bc6d35e9168",
    "preproc.json": "9046e3b5318879cf8fc469e8c691efac99a96dc2d2f4fb84df7748447006e8b1",
    "REG_STATS_fold1.json": "bc653aeb589b77f7a6756f56a6302d9e9f642e89ecc6521ceef2942b86795195",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_assets(model_dir: Path) -> Dict[str, Path]:
    paths = {name: model_dir / name for name in ASSET_HASHES}
    for name, path in paths.items():
        if not path.is_file() or _sha256(path) != ASSET_HASHES[name]:
            raise ValueError(f"Missing or changed model asset: {name}")
    return paths


class InferencePreprocessor:
    def __init__(self, saved: dict):
        self.cont_feats = saved["cont_feats"]
        self.cat_feats = saved["cat_feats"]
        self.impute_cat = saved.get("impute_cat", "UNK")
        self.median_ = {key: float(value) for key, value in saved["median_"].items()}
        self.scale_ = {key: float(value) for key, value in saved["scale_"].items()}
        self.vocabs_ = {
            key: {token: int(index) for token, index in vocab.items()}
            for key, vocab in saved["vocabs_"].items()
        }
        self.cardinalities_ = {key: int(value) for key, value in saved["cardinalities_"].items()}
        self.model_cfg = saved["model_cfg"]
        if (self.cont_feats != CONT or self.cat_feats != ["split_name"]
                or saved.get("te_state") is not None
                or self.vocabs_ != {"split_name": {"UNK": 0, "train": 1}}
                or self.cardinalities_ != {"split_name": 3}):
            raise ValueError("Unexpected saved preprocessing contract")

    def transform_cont(self, df: pd.DataFrame) -> np.ndarray:
        xs = []
        for c in self.cont_feats:
            s = pd.to_numeric(df[c], errors='coerce').fillna(self.median_[c]).astype(np.float32).values
            z = (s - self.median_[c]) / self.scale_[c]
            z = np.clip(z, -10.0, 10.0)
            z = np.where(np.isfinite(z), z, 0.0).astype(np.float32)
            xs.append(z)
        if not xs:
            return np.zeros((len(df), 0), dtype=np.float32)
        return np.stack(xs, axis=1).astype(np.float32)

    def transform_cat(self, df: pd.DataFrame) -> List[np.ndarray]:
        outs = []
        for c in self.cat_feats:
            vocab = self.vocabs_[c]
            col = df[c].astype('object').fillna(self.impute_cat).astype(str).values
            idx = np.array([vocab.get(tok, len(vocab)) for tok in col], dtype=np.int64)
            outs.append(idx)
        return outs


def _load_preprocessor(path: Path) -> InferencePreprocessor:
    return InferencePreprocessor(json.loads(path.read_text(encoding="utf-8")))


class InferenceDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, pre: InferencePreprocessor):
        self.df = frame.reset_index(drop=True)
        self.Xc = pre.transform_cont(self.df)
        self.Xcats = pre.transform_cat(self.df)
        self.Xtopo = np.zeros((len(self.df), 0, 16), dtype=np.float32)
        weights = pd.to_numeric(self.df["weight"], errors="coerce").fillna(1.0).values.astype(np.float32)
        self.weights = torch.tensor(weights, dtype=torch.float32)
        self.ids = self.df["sample_id"].astype(str).tolist()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int):
        x_cont = torch.from_numpy(self.Xc[index])
        x_cats = [torch.tensor(cat[index], dtype=torch.long) for cat in self.Xcats]
        x_topo = torch.from_numpy(self.Xtopo[index])
        return x_cont, x_cats, x_topo, self.weights[index], self.ids[index]


def _strip_averagedmodel_state(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    # Handle torch.optim.swa_utils.AveragedModel state dict: may contain 'n_averaged' and 'module.' prefix.
    if not isinstance(state, dict):
        return state
    if ("n_averaged" in state) or any(k.startswith("module.") for k in state.keys()):
        new_state = {}
        for k, v in state.items():
            if k == "n_averaged":
                continue
            nk = k.split("module.", 1)[1] if k.startswith("module.") else k
            new_state[nk] = v
        return new_state
    return state


class FTTransformer(nn.Module):
    def __init__(self, n_cont: int, cat_cardinalities: List[int],
                 d_model=256, n_heads=8, n_layers=6, dropout=0.2, d_ff=512,
                 use_cls=True, out_dim=2, token_dropout_p=0.02,
                 cont_as_tokens=True, cont_bias=True, head_msd=1, head_hidden=0,
                 supcon_proj_dim=64,
                 topo_dim: int = 0, topo_tokens: int = 0):
        super().__init__()
        self.use_cls = use_cls
        self.n_cont = int(n_cont)
        self.cat_card = list(cat_cardinalities or [])
        self.token_dropout_p = float(token_dropout_p)
        self.cont_as_tokens = bool(cont_as_tokens)
        self.head_msd = max(1, int(head_msd))
        self.head_hidden = int(head_hidden)
        self.out_dim = int(out_dim)

        self.topo_dim = int(topo_dim)
        self.topo_tokens = int(topo_tokens)

        self.emb_cats = nn.ModuleList([nn.Embedding(int(c), d_model) for c in self.cat_card])

        if self.cont_as_tokens and (self.n_cont > 0):
            self.cont_weight = nn.Parameter(torch.randn(self.n_cont, d_model) * 0.02)
            self.cont_bias = nn.Parameter(torch.zeros(self.n_cont, d_model)) if cont_bias else None
        else:
            self.proj_cont = nn.Linear(self.n_cont if self.n_cont>0 else 1, d_model)

        if self.topo_tokens > 0 and self.topo_dim > 0:
            self.proj_topo = nn.Linear(self.topo_dim, d_model)
        else:
            self.proj_topo = None

        n_tokens = (1 if use_cls else 0) + len(self.cat_card)
        if self.n_cont > 0:
            n_tokens += (self.n_cont if (self.cont_as_tokens and self.n_cont>0) else 1)
        n_tokens += self.topo_tokens

        self.pos_emb = nn.Parameter(torch.randn(1, n_tokens, d_model) * 0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, activation='gelu', norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.supcon_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, supcon_proj_dim)
        )

        if self.head_hidden and self.head_hidden > 0:
            self.head_mlp = nn.Sequential(
                nn.Linear(d_model, self.head_hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(self.head_hidden, out_dim)
            )
        else:
            self.head_mlp = nn.Linear(d_model, out_dim)

    def set_token_dropout(self, p: float):
        self.token_dropout_p = float(max(0.0, min(0.9, p)))

    def _tokens(self, x_cont: torch.Tensor, x_cats: List[torch.Tensor], x_topo: Optional[torch.Tensor] = None):
        B = x_cont.size(0)
        D = self.pos_emb.size(-1)
        tokens = []

        if self.use_cls:
            cls = torch.zeros(B, 1, D, device=x_cont.device, dtype=x_cont.dtype)
            tokens.append(cls)

        # Categorical tokens
        for i, emb in enumerate(self.emb_cats):
            xi = x_cats[i]
            tok = emb(xi).unsqueeze(1)
            if self.training and self.token_dropout_p>0:
                mask = (torch.rand(B, 1, 1, device=tok.device) > self.token_dropout_p).float()
                tok = tok * mask
            tokens.append(tok)

        # Continuous tokens
        if self.n_cont > 0:
            if self.cont_as_tokens:
                cont_tok = x_cont.unsqueeze(-1) * self.cont_weight.unsqueeze(0)
                if self.cont_bias is not None:
                    cont_tok = cont_tok + self.cont_bias.unsqueeze(0)
                if self.training and self.token_dropout_p>0:
                    mask = (torch.rand(B, cont_tok.size(1), 1, device=cont_tok.device) > self.token_dropout_p).float()
                    cont_tok = cont_tok * mask
                tokens.append(cont_tok)
            else:
                cont_tok = self.proj_cont(x_cont if self.n_cont>0 else torch.zeros(B,1,device=x_cont.device,dtype=x_cont.dtype)).unsqueeze(1)
                tokens.append(cont_tok)

        # Topology tokens
        if self.topo_tokens > 0 and self.proj_topo is not None:
            if x_topo is None:
                x_topo = torch.zeros(B, self.topo_tokens, self.topo_dim, device=x_cont.device, dtype=x_cont.dtype)
            else:
                # x_topo: [B, T_topo, topo_dim] (T_topo may be <= configured)
                if x_topo.dim() == 2:
                    x_topo = x_topo.unsqueeze(1)
                # pad/truncate to expected number of tokens
                if x_topo.size(1) < self.topo_tokens:
                    pad = torch.zeros(B, self.topo_tokens - x_topo.size(1), x_topo.size(2), device=x_topo.device, dtype=x_topo.dtype)
                    x_topo = torch.cat([x_topo, pad], dim=1)
                elif x_topo.size(1) > self.topo_tokens:
                    x_topo = x_topo[:, :self.topo_tokens, :]
            topo_tok = self.proj_topo(x_topo)
            tokens.append(topo_tok)

        X = torch.cat(tokens, dim=1)
        return X

    def encode(self, x_cont: torch.Tensor, x_cats: List[torch.Tensor], x_topo: Optional[torch.Tensor] = None):
        X = self._tokens(x_cont, x_cats, x_topo)
        X = X + self.pos_emb[:, :X.size(1), :]
        X = self.encoder(X)
        h = X[:, 0, :] if self.use_cls else X.mean(1)
        return h

    def forward_features(self, x_cont: torch.Tensor, x_cats: List[torch.Tensor], x_topo: Optional[torch.Tensor] = None):
        h = self.encode(x_cont, x_cats, x_topo)
        h = self.norm(h)
        h = self.dropout(h)
        return h

    def head(self, feats: torch.Tensor):
        if self.head_msd > 1 and self.training:
            outs = []
            for _ in range(self.head_msd):
                z = F.dropout(feats, p=self.dropout.p, training=True)
                outs.append(self.head_mlp(z))
            return torch.stack(outs, 0).mean(0)
        else:
            return self.head_mlp(feats)

    def forward(self, x_cont: torch.Tensor, x_cats: List[torch.Tensor], x_topo: Optional[torch.Tensor] = None, return_features=False):
        feats = self.forward_features(x_cont, x_cats, x_topo)
        if return_features:
            return F.normalize(feats, dim=1)
        return self.head(feats)


def _configure_encoder(model: FTTransformer, profile: str, device: str) -> None:
    if profile == "historical_cuda":
        if device != "cuda":
            raise ValueError("historical_cuda profile requires CUDA")
        torch.backends.mha.set_fastpath_enabled(True)
        return
    if profile != "portable_tanh":
        raise ValueError("Unknown inference profile")
    layers = list(model.encoder.layers)
    if len(layers) != 12:
        raise ValueError("Expected 12 encoder layers")
    for layer in layers:
        if (layer.linear1.in_features != 320 or layer.linear1.out_features != 768
                or not layer.norm_first or layer.activation_relu_or_gelu != 2):
            raise ValueError("Unexpected encoder configuration")
    torch.backends.mha.set_fastpath_enabled(False)
    for layer in layers:
        layer.activation = functools.partial(F.gelu, approximate="tanh")
        layer.activation_relu_or_gelu = 0


def _validate_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame")
    if len(frame.columns) != len(set(frame.columns)) or set(frame.columns) != set(REQUIRED_COLUMNS):
        raise ValueError("Expected only the declared descriptor columns")
    if frame.empty or frame["sample_id"].isna().any():
        raise ValueError("Missing sample IDs")
    selected = frame.loc[:, REQUIRED_COLUMNS].copy()
    selected["sample_id"] = selected["sample_id"].astype(str)
    if selected["sample_id"].str.strip().eq("").any() or not selected["sample_id"].is_unique:
        raise ValueError("Expected unique, nonblank sample IDs")
    if not selected["split_name"].eq("test").all():
        raise ValueError("Expected split_name=test")
    for column in CONT:
        if np.isinf(pd.to_numeric(selected[column], errors="raise").to_numpy(float)).any():
            raise ValueError("Infinite numeric feature")
    return selected


@torch.no_grad()
def _infer_regression_outputs(model: FTTransformer, loader: DataLoader, device: str) -> np.ndarray:
    model.eval()
    model.set_token_dropout(0.0)
    outputs = []
    for batch in loader:
        x_cont, x_cats, x_topo = batch[0], batch[1], batch[2]
        x_cont = x_cont.to(device)
        x_cats = [tensor.to(device) for tensor in x_cats]
        x_topo = x_topo.to(device)
        outputs.append(model(x_cont, x_cats, x_topo).view(-1).detach().cpu())
    return torch.cat(outputs, 0).numpy()


def predict_feature_frame(
    frame: pd.DataFrame,
    model_dir: Path,
    *,
    device: str = "cpu",
    profile: str = "portable_tanh",
) -> np.ndarray:
    """Return target-unit predictions in the input row order; no labels are used."""
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    selected = _validate_frame(frame)
    paths = _check_assets(Path(model_dir))
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False

    pre = _load_preprocessor(paths["preproc.json"])
    cfg = pre.model_cfg
    if cfg.get("task_type") != "regression" or cfg.get("num_classes") != 1:
        raise ValueError("Unexpected saved model task")
    stats = json.loads(paths["REG_STATS_fold1.json"].read_text(encoding="utf-8"))
    mean = float(stats["target_mean_train"])
    std = float(stats["target_std_train"])
    if stats.get("norm") != "zscore" or not math.isfinite(mean) or not math.isfinite(std) or std <= 0:
        raise ValueError("Invalid saved target normalization")

    keys = ("d_model", "n_heads", "n_layers", "d_ff", "dropout", "cont_as_tokens",
            "cont_bias", "head_msd", "head_hidden", "supcon_proj_dim")
    model = FTTransformer(
        196, [3], out_dim=1, token_dropout_p=0.0, topo_dim=0, topo_tokens=0,
        **{key: cfg[key] for key in keys},
    ).to(device)
    state = torch.load(paths["RAW_fold1.pth"], map_location=device, weights_only=True)
    model.load_state_dict(_strip_averagedmodel_state(state), strict=True)
    model.eval()
    _configure_encoder(model, profile, device)

    dataset = InferenceDataset(selected, pre)
    if not np.all(dataset.Xcats[0] == 2):
        raise ValueError("Unexpected categorical encoding")
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0,
                        pin_memory=(device == "cuda"))
    raw = _infer_regression_outputs(model, loader, device)
    predictions = (raw * np.float32(std) + np.float32(mean)).astype(float)
    if predictions.shape != (len(selected),) or not np.isfinite(predictions).all():
        raise ValueError("Invalid predictions")
    return predictions
