import logging
from pathlib import Path
from typing import Dict

import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphSAGELayer(nn.Module):
    def __init__(self, in_feats: int, out_feats: int, activation=F.relu):
        super().__init__()
        self.linear = nn.Linear(in_feats * 2, out_feats)
        self.activation = activation

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index
        agg = torch.zeros_like(x)
        agg.index_add_(0, dst, x[src])
        h = torch.cat([x, agg], dim=1)
        h = self.linear(h)
        if self.activation is not None:
            h = self.activation(h)
        return h


class GraphSAGE(nn.Module):
    def __init__(self, in_feats: int, hidden_feats: int, num_classes: int):
        super().__init__()
        self.sage1 = GraphSAGELayer(in_feats, hidden_feats)
        self.sage2 = GraphSAGELayer(hidden_feats, num_classes, activation=None)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.sage1(x, edge_index)
        h = self.sage2(h, edge_index)
        return h


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower()
    col_map = {}
    for col in df.columns:
        if "from" in col:
            col_map[col] = "from"
        elif "to" in col:
            col_map[col] = "to"
        elif "value" in col:
            col_map[col] = "value"
        elif "error" in col:
            col_map[col] = "is_error"
    df = df.rename(columns=col_map)
    required = ["from", "to", "value"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(missing)}")
    df["from"] = df["from"].astype(str).str.lower().str.strip()
    df["to"] = df["to"].astype(str).str.lower().str.strip()
    df["value"] = pd.to_numeric(df["value"], errors="coerce").fillna(0)
    if "is_error" not in df.columns:
        df["is_error"] = 0
    else:
        df["is_error"] = pd.to_numeric(df["is_error"], errors="coerce").fillna(0)
    df = df[df["to"].notna() & (df["to"] != "nan")]
    return df


def _build_features(df: pd.DataFrame):
    G = nx.DiGraph()
    for _, row in df.iterrows():
        fr = row["from"]
        to = row["to"]
        val = float(row["value"])
        is_err = float(row.get("is_error", 0))
        if G.has_edge(fr, to):
            G[fr][to]["value_sum"] += val
            G[fr][to]["weight"] += 1
            G[fr][to]["fail_count"] += (1 if is_err else 0)
        else:
            G.add_edge(fr, to, value_sum=val, weight=1, fail_count=(1 if is_err else 0))

    node_features = {}
    for node in G.nodes():
        out_edges = G.out_edges(node, data=True)
        in_edges = G.in_edges(node, data=True)
        total_sent = sum(edata.get("value_sum", 0) for _, _, edata in out_edges)
        total_received = sum(edata.get("value_sum", 0) for _, _, edata in in_edges)
        out_deg = G.out_degree(node)
        in_deg = G.in_degree(node)
        node_features[node] = [total_sent, total_received, out_deg, in_deg]

    if not node_features:
        raise ValueError("No graph nodes built from data")

    feat_df = pd.DataFrame.from_dict(
        node_features, orient="index", columns=["total_sent", "total_received", "out_deg", "in_deg"]
    ).fillna(0)

    node_list = list(feat_df.index)
    node2idx = {n: i for i, n in enumerate(node_list)}
    edges_u, edges_v = [], []
    for u, v in G.edges():
        if u in node2idx and v in node2idx:
            edges_u.append(node2idx[u])
            edges_v.append(node2idx[v])
    edge_index = torch.tensor([edges_u, edges_v], dtype=torch.long)
    X = torch.tensor(feat_df.values, dtype=torch.float)
    return X, edge_index, node_list


def run_gnn_inference(csv_content: str, weights_path: Path) -> Dict[str, float]:
    if not weights_path.exists():
        raise FileNotFoundError(f"GraphSAGE weights not found at {weights_path}")

    df = pd.read_csv(pd.io.common.StringIO(csv_content))
    df = _normalize_columns(df)
    X, edge_index, node_list = _build_features(df)

    in_feats = X.shape[1]
    hidden_feats = 64
    num_classes = 2

    model = GraphSAGE(in_feats, hidden_feats, num_classes)
    state = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    with torch.no_grad():
        logits = model(X, edge_index)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

    return {node: float(prob) for node, prob in zip(node_list, probs)}
