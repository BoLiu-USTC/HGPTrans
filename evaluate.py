"""Evaluation script: run a trained checkpoint over a test split (or a folder
of extra STL files) and report standard drag-regression metrics.

Example (test split):
    python evaluate.py --data_dir /path/to/DrivAerNet++ \
        --checkpoint /path/to/logs/run1/model_best.pt

Example (extra STL folder, filenames encode the true Cd as the last number,
e.g. design_0.312.stl):
    python evaluate.py --data_dir /path/to/DrivAerNet++ --extra_data ./data/Ahmed \
        --checkpoint /path/to/logs/run1/model_best.pt
"""

import argparse
import datetime
import os
import re
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr
from torch_geometric.loader import DataLoader

from dataset import STLGraphDataset, Standardize
from models import build_model
from utils import (Logger, count_parameters, print_gpu_memory,
                   query_values_from_excel, read_split_txts, read_split_txts_nostls)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate mesh-graph drag regression models")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Dataset directory containing cleaned_stls/, labels.xlsx and splits")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model_best.pt or checkpoint.pth")
    parser.add_argument("--checkpoint_key", type=str, default="auto",
                        choices=["auto", "state_dict", "model_state_dict"],
                        help="How to read weights from the checkpoint file")
    parser.add_argument("--model", type=str, default="HGPTrans",
                        choices=["HGPTrans"], help="Model architecture")
    parser.add_argument("--input_dim", type=int, default=6)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--pooling_ratio", type=float, default=0.8)
    parser.add_argument("--extra_data", type=str, default=None,
                        help="Folder of extra STL files to predict instead of the test split")
    parser.add_argument("--num_cpus", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Keep 1: forward uses dense reshapes assuming one graph per batch")
    parser.add_argument("--simplify_size", type=int, default=417114)
    parser.add_argument("--cuda", type=int, default=0, help="CUDA device index; -1 forces CPU")
    return parser.parse_args()


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    print("r2", r2, "ss_res", ss_res, "ss_tot", ss_tot,
          "MSE", np.mean((y_true - y_pred) ** 2), "NUM", len(y_true))
    return r2


def relative_l2_score(y_true, y_pred):
    x = np.sqrt(np.mean((y_true - y_pred) ** 2))
    y = np.sqrt(np.mean(y_true ** 2))
    return x / y


def load_model_weights(model, checkpoint_path, key="auto"):
    """Load weights from model_best.pt (raw state dict) or checkpoint.pth."""
    if key == "auto":
        key = "model_state_dict" if checkpoint_path.endswith("checkpoint.pth") else "state_dict"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint[key] if key == "model_state_dict" else checkpoint
    # strip DDP "module." prefixes if present
    new_state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)


def load_test_data(data_dir, extra_data):
    """Return (X_test, y_test, X_name, pred_dir)."""
    if extra_data:
        stl_files = [os.path.join(extra_data, f) for f in sorted(os.listdir(extra_data))
                     if f.endswith(".stl")]
        if not stl_files:
            raise FileNotFoundError(f"No .stl files found in {extra_data}")

        # true Cd is encoded at the end of the filename, e.g. design_0.312.stl
        labels = []
        for path in stl_files:
            match = re.findall(r"-?\d+\.?\d*", os.path.splitext(os.path.basename(path))[0])
            if not match:
                raise ValueError(
                    f"Cannot parse a numeric Cd label from filename '{path}'. "
                    f"Name extra STL files like 'design_0.312.stl'.")
            labels.append(float(match[-1]))

        names = [os.path.splitext(os.path.basename(p))[0] for p in stl_files]
        return stl_files, labels, names, extra_data

    splits_dir = os.path.join(data_dir, "train_val_test_splits")
    if os.path.exists(os.path.join(splits_dir, "test_design_ids.txt")):
        labels_xlsx = os.path.join(data_dir, "labels.xlsx")
        stl_dir = os.path.join(data_dir, "cleaned_stls")
        X_test = read_split_txts(stl_dir, os.path.join(splits_dir, "test_design_ids.txt"))
        X_name = read_split_txts_nostls(stl_dir, os.path.join(splits_dir, "test_design_ids.txt"))
        y_test = query_values_from_excel(os.path.join(splits_dir, "test_design_ids.txt"),
                                         labels_xlsx)
    elif os.path.exists(os.path.join(splits_dir, "test.txt")):
        labels_xlsx = os.path.join(data_dir, "DrivAerNetPlusPlus_Cd_8k_Updated.xlsx")
        stl_dir = os.path.join(data_dir, "cleaned_stls")
        X_test = read_split_txts(stl_dir, os.path.join(splits_dir, "test.txt"))
        X_name = read_split_txts_nostls(stl_dir, os.path.join(splits_dir, "test.txt"))
        y_test = query_values_from_excel(os.path.join(splits_dir, "test.txt"), labels_xlsx)
    else:
        raise FileNotFoundError(
            f"No test split found under {splits_dir}: expected test_design_ids.txt "
            f"or test.txt")

    missing = [n for n, v in zip(X_name, y_test) if v is None]
    if missing:
        raise ValueError(f"{len(missing)} test designs have no label in {labels_xlsx}, "
                         f"e.g. {missing[:3]}")
    return X_test, y_test, X_name, None


def main():
    args = parse_args()

    if args.cuda >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.cuda}")
    else:
        device = torch.device("cpu")

    X_test, y_test, X_name, pred_dir = load_test_data(args.data_dir, args.extra_data)
    if pred_dir is None:
        pred_dir = os.path.join(os.path.dirname(os.path.abspath(args.checkpoint)), "predLog")
    os.makedirs(pred_dir, exist_ok=True)

    file_name = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log_file = os.path.join(pred_dir, file_name + ".log")
    sys.stdout = Logger(log_file)
    sys.stderr = Logger(log_file, stream=sys.stderr)

    print(f"Testdata: {len(X_test)}")
    print(X_test[:3])
    print(y_test[:3])

    print("=" * 70)
    print(f"Model: {args.model} | CUDA: {args.cuda} | CPUs: {args.num_cpus}")

    model = build_model(args.model, input_dim=args.input_dim, hidden_dim=args.hidden_dim,
                        out_dim=1, pooling_ratio=args.pooling_ratio, n_layers=args.layers,
                        n_head=8, slice_num=32)
    print(f"Total number of parameters: {count_parameters(model)}")

    model = model.to(device)
    load_model_weights(model, args.checkpoint, key=args.checkpoint_key)
    print(f"Loaded weights from {args.checkpoint}")

    transform = Standardize()
    test_dataset = STLGraphDataset(X_test, y_test, args.input_dim, transform,
                                   simplify_size=args.simplify_size)
    prefetch = 2 if args.num_cpus > 0 else None
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_cpus, prefetch_factor=prefetch)

    criterion = nn.MSELoss()

    print("Begin testing....")
    model.eval()
    mse_loss_total = 0.0
    num_samples = 0
    Pred, Truth = [], []

    t_begin = time.time()
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            t0 = time.time()
            data = data.to(device)
            outputs = model(data)
            mse_loss = criterion(outputs, data.y.view(-1, 1)).item()

            pred = outputs.float().view(-1).cpu().numpy()
            label = data.y.float().view(-1).cpu().numpy()

            mae_loss = np.abs(pred - label)
            relative_error = mae_loss / np.abs(label)

            for ii in range(len(pred)):
                print("{} pred:{:.5f} truth:{:.5f} MAE:{:.5f} Relative:{:.4f} time:{:.2f}".format(
                    i * args.batch_size + ii, pred[ii], label[ii], mae_loss[ii],
                    relative_error[ii], time.time() - t0))
                Pred.append(pred[ii])
                Truth.append(label[ii])

            mse_loss_total += mse_loss * len(pred)
            num_samples += len(pred)

    total_time = time.time() - t_begin
    print(f"Time used for testing: {total_time:.4f} s, "
          f"per stl costs: {total_time / num_samples:.4f}s")
    print_gpu_memory()

    Pred, Truth = np.asarray(Pred, dtype=np.float64), np.asarray(Truth, dtype=np.float64)
    mean_mse_loss = mse_loss_total / num_samples

    r2 = r2_score(Truth, Pred)
    Error = np.abs(Pred - Truth)
    relative = np.mean(Error / np.abs(Truth))
    relative_std = np.std(Error / np.abs(Truth))
    pearsonr_corr = pearsonr(Truth, Pred)[0]
    spearmanr_corr = spearmanr(Truth, Pred)[0]

    abs_l2_error = np.linalg.norm(Truth - Pred)
    norm_y_truth = np.linalg.norm(Truth)
    rel_l2_error = relative_l2_score(Truth, Pred)
    print("Norm_y_truth / abs_l2_error / rel_l2:",
          norm_y_truth, abs_l2_error, rel_l2_error)

    data_to_save = list(zip(X_name, Truth, Pred))
    np.savetxt(os.path.join(pred_dir, "preds.txt"), data_to_save, fmt="%s")

    print("MSE:{:.4e} MAE:{:.5f} ± {:.5f} max:{:.4f} relative:{:.2%} ± {:.2%} "
          "relL2:{:.4f} r2:{:.4f} pearCorr:{:.4f} spearCorr:{:.4f}".format(
              mean_mse_loss, np.mean(Error), np.std(Error), np.max(Error),
              relative, relative_std, rel_l2_error, r2, pearsonr_corr, spearmanr_corr))

    # signed error breakdown
    error_sign = np.mean(Pred - Truth)
    std_sign = np.std(Pred - Truth)
    error_re_sign = np.mean((Pred - Truth) / np.abs(Truth))
    std_re_sign = np.std((Pred - Truth) / np.abs(Truth))
    print(f"CD: mean + std: {error_sign:.5f} + {std_sign:.5f}")
    print(f"RE: mean + std: {error_re_sign:.5f} + {std_re_sign:.5f}")

    # error distribution plot
    g = sns.jointplot(x=Pred, y=Error, kind="scatter")
    g.set_axis_labels("Cd", "Error")
    g.fig.set_size_inches(10, 8)
    plt.savefig(os.path.join(pred_dir, "error.png"))
    plt.close()

    print("Time used for testing: {:.4f} s".format(time.time() - t_begin))
    print("END!")


if __name__ == "__main__":
    main()
