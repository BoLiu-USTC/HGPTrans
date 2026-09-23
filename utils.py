"""Shared helpers: logging, data loading, seeding and plotting."""

import datetime
import os
import sys

import matplotlib
matplotlib.use("Agg")  # headless-safe plotting (no GUI required)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


class Logger:
    """Tee stdout/stderr to a log file with timestamps.

    Pass ``stream=sys.stderr`` to wrap stderr; the wrapped stream is captured
    at construction time so wrapping stderr no longer hijacks stdout.
    """

    def __init__(self, filename, stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, "a", encoding="utf-8")
        self.previousMsg = None
        setattr(sys, "stdout" if stream is sys.stdout else "stderr", self)

    def write(self, message):
        if self.previousMsg is None or "\n" in self.previousMsg:
            topMsg = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + " : "
            self.terminal.write(topMsg)
            self.log.write(topMsg)

        if isinstance(message, str):
            self.previousMsg = message
        if self.previousMsg is None:
            self.previousMsg = ""

        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        pass


def read_split_txts(dir_path, filepath):
    """Read a split file (one design id per line) into STL file paths."""
    lines = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            cleaned_line = line.strip()
            if cleaned_line:
                lines.append(os.path.join(dir_path, cleaned_line + ".stl"))
    return lines


def read_split_txts_nostls(dir_path, filepath):
    """Read a split file into raw design ids (no path extension)."""
    lines = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            cleaned_line = line.strip()
            if cleaned_line:
                lines.append(cleaned_line)
    return lines


def query_values_from_excel(txt_file, xlsx_file):
    """Look up scalar labels for design ids listed in ``txt_file``.

    The Excel file must contain 'ID' and 'Drag_Value' columns. Labels are
    returned in the same order as the ids; a missing id yields None (callers
    treat that as a hard error).
    """
    with open(txt_file, "r", encoding="utf-8") as f:
        filenames = [line.strip() for line in f.readlines() if line.strip()]

    df = pd.read_excel(xlsx_file)
    for col in ("ID", "Drag_Value"):
        if col not in df.columns:
            raise KeyError(f"{xlsx_file} is missing required column '{col}' "
                           f"(found: {list(df.columns)})")

    filename_to_value = df.set_index("ID")["Drag_Value"].to_dict()

    return [filename_to_value.get(name) for name in filenames]


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def plot_loss(trainLoss, valLoss, path):
    epochs = np.arange(1, len(trainLoss) + 1)
    plt.semilogy(epochs, trainLoss, label="train")
    plt.semilogy(epochs, valLoss, label="val")
    plt.legend()
    plt.savefig(os.path.join(path, "loss.png"))
    plt.close()
    np.savetxt(os.path.join(path, "loss.txt"), np.stack([trainLoss, valLoss], -1))


def print_gpu_memory():
    if torch.cuda.is_available():
        allocated_memory = torch.cuda.memory_allocated() / 1024 ** 3
        reserved_memory = torch.cuda.memory_reserved() / 1024 ** 3
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        print(f"Allocated and Reserved/Total Memory: "
              f"{allocated_memory + reserved_memory:.2f}/{total_memory:.2f} GB")
    else:
        print("CUDA is not available")
