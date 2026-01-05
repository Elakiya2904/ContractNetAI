import argparse
from pathlib import Path
import logging

import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


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


def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
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


def build_graph_features(df: pd.DataFrame):
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

    # labels: use is_error aggregated by 'to'
    labels_series = df.groupby("to")["is_error"].max()
    common_index = feat_df.index.intersection(labels_series.index)
    X = feat_df.loc[common_index]
    y = labels_series.loc[common_index]

    mask = ~np.isnan(y)
    X = X[mask]
    y = y[mask]

    node_list = list(X.index)
    node2idx = {n: i for i, n in enumerate(node_list)}
    edges_u, edges_v = [], []
    for u, v in G.edges():
        if u in node2idx and v in node2idx:
            edges_u.append(node2idx[u])
            edges_v.append(node2idx[v])
    edge_index = torch.tensor([edges_u, edges_v], dtype=torch.long)
    X_tensor = torch.tensor(X.values, dtype=torch.float)
    y_tensor = torch.tensor(y.values, dtype=torch.long)
    return X_tensor, y_tensor, edge_index, node_list


def train(csv_path: Path, out_path: Path, epochs: int = 30, hidden_feats: int = 64):
    logging.info(f"Loading data from {csv_path}")
    df = pd.read_csv(csv_path)
    df = normalize_df(df)
    X, y, edge_index, node_list = build_graph_features(df)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = X.to(device)
    y = y.to(device)
    edge_index = edge_index.to(device)

    in_feats = X.shape[1]
    num_classes = 2
    model = GraphSAGE(in_feats, hidden_feats, num_classes).to(device)

    idx = torch.randperm(len(X))
    train_size = int(0.7 * len(X))
    train_idx = idx[:train_size]
    test_idx = idx[train_size:]

    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    def evaluate(split_idx: torch.Tensor, split_name: str):
        model.eval()
        with torch.no_grad():
            logits = model(X, edge_index)
            preds = logits.argmax(dim=1)
        acc = (preds[split_idx] == y[split_idx]).float().mean().item()
        report = classification_report(
            y[split_idx].detach().cpu().numpy(), preds[split_idx].detach().cpu().numpy(), digits=4
        )
        logging.info(f"{split_name} accuracy={acc:.4f}\n{report}")
        return acc, report

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(X, edge_index)
        loss = criterion(out[train_idx], y[train_idx])
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                logits = model(X, edge_index)
                preds = logits.argmax(dim=1)
                train_acc = (preds[train_idx] == y[train_idx]).float().mean()
                test_acc = (preds[test_idx] == y[test_idx]).float().mean()
            logging.info(
                f"Epoch {epoch+1:02d} loss={loss.item():.4f} train_acc={train_acc:.4f} test_acc={test_acc:.4f}"
            )

    logging.info("Final evaluation reports:")
    evaluate(train_idx, "Train")
    evaluate(test_idx, "Test")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_path)
    logging.info(f"Saved weights to {out_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=Path("ethereum_transactions.csv"), help="Path to training CSV")
    parser.add_argument("--out", type=Path, default=Path("backend/models/graphsage_model.pth"), help="Output weights path")
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()
    train(args.csv, args.out, epochs=args.epochs)
