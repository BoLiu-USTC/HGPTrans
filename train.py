"""DDP training script for mesh-graph drag regression.

Example (single GPU):
    python train.py --data_dir /path/to/DrivAerNet++ --save_dir ./logs/run1

Example (multi GPU):
    python train.py --data_dir /path/to/DrivAerNet++ --save_dir ./logs/run1 --world_size 4

The dataset directory is auto-detected:
  * DrivAerNet-style:  labels.xlsx + train_val_test_splits/{train,val}_design_ids.txt
  * DrivAerNet++ 8k:   DrivAerNetPlusPlus_Cd_8k_Updated.xlsx + train_val_test_splits/{train,val}.txt
"""

import argparse
import contextlib
import datetime
import os
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler
from torch_geometric.loader import DataLoader

from dataset import STLGraphDataset, Standardize
from models import build_model
from utils import (Logger, count_parameters, plot_loss, query_values_from_excel,
                   read_split_txts, set_seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Train mesh-graph drag regression models")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Dataset directory containing cleaned_stls/, labels.xlsx and splits")
    parser.add_argument("--save_dir", type=str, default="./logs/run",
                        help="Output directory for checkpoints, logs and plots")
    parser.add_argument("--model", type=str, default="HGPTrans",
                        choices=["HGPTrans"], help="Model architecture")
    parser.add_argument("--input_dim", type=int, default=6, help="Input feature dim (xyz + normal)")
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=4, help="Number of HGPTrans blocks")
    parser.add_argument("--pooling_ratio", type=float, default=0.8)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Keep 1: forward uses dense reshapes assuming one graph per batch")
    parser.add_argument("--accumulation_steps", type=int, default=16,
                        help="Gradient accumulation steps (effective batch = batch_size * this)")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_cpus", type=int, default=8, help="DataLoader workers per rank")
    parser.add_argument("--simplify_size", type=int, default=417114,
                        help="Decimate meshes above this vertex count (0 disables)")
    parser.add_argument("--world_size", type=int, default=1, help="Number of GPU processes")
    parser.add_argument("--port", type=str, default="12454", help="DDP master port")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from <save_dir>/checkpoint.pth")
    return parser.parse_args()


def init_distributed_mode(rank, world_size, port="12454"):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = port
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)


def save_checkpoint(model, optimizer, epoch, scheduler, filename="checkpoint.pth"):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "lr": scheduler.get_last_lr(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(model, optimizer, scheduler, filename="checkpoint.pth"):
    checkpoint = torch.load(filename, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return model, optimizer, checkpoint["epoch"] + 1, scheduler


def gather_tensor(t, device):
    """All-gather a [N,1] tensor across ranks (no-op for world_size=1).

    Works under NCCL (GPU tensors) and gloo (CPU tensors alike). Under NCCL the
    input must live on the GPU, so CPU tensors are staged there first."""
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return t
    if dist.get_backend() == "nccl" and device.type == "cpu":
        device = torch.device("cuda", torch.cuda.current_device())
    t = t.to(device)
    world_size = dist.get_world_size()
    gather_list = [torch.zeros_like(t) for _ in range(world_size)]
    dist.all_gather(gather_list, t.contiguous())
    return torch.cat(gather_list, dim=0)


def load_split(stl_dir):
    """Auto-detect dataset format and return (X_train, y_train, X_val, y_val)."""
    splits_dir = os.path.join(stl_dir, "train_val_test_splits")

    if os.path.exists(os.path.join(splits_dir, "train_design_ids.txt")):
        # DrivAerNet / DrivAerNet++ style: per-split design id lists + labels.xlsx
        xlsx = os.path.join(stl_dir, "labels.xlsx")
        stl_dir_stls = os.path.join(stl_dir, "cleaned_stls")
        X_train = read_split_txts(stl_dir_stls, os.path.join(splits_dir, "train_design_ids.txt"))
        y_train = query_values_from_excel(os.path.join(splits_dir, "train_design_ids.txt"), xlsx)
        X_val = read_split_txts(stl_dir_stls, os.path.join(splits_dir, "val_design_ids.txt"))
        y_val = query_values_from_excel(os.path.join(splits_dir, "val_design_ids.txt"), xlsx)
    elif os.path.exists(os.path.join(splits_dir, "train.txt")):
        # DrivAerNet++ 8k+ style
        xlsx = os.path.join(stl_dir, "DrivAerNetPlusPlus_Cd_8k_Updated.xlsx")
        stl_dir_stls = os.path.join(stl_dir, "cleaned_stls")
        X_train = read_split_txts(stl_dir_stls, os.path.join(splits_dir, "train.txt"))
        y_train = query_values_from_excel(os.path.join(splits_dir, "train.txt"), xlsx)
        X_val = read_split_txts(stl_dir_stls, os.path.join(splits_dir, "val.txt"))
        y_val = query_values_from_excel(os.path.join(splits_dir, "val.txt"), xlsx)
    else:
        raise FileNotFoundError(
            f"No split files found under {splits_dir}: expected train_design_ids.txt "
            f"(DrivAerNet style) or train.txt (DrivAerNet++ 8k style)")

    for name, split in [("train", y_train), ("val", y_val)]:
        missing = [x for x in split if x is None]
        if missing:
            raise ValueError(f"{len(missing)} {name} designs have no label in {xlsx}, "
                             f"e.g. {missing[:3]}")
    return X_train, y_train, X_val, y_val


def train(rank, world_size, args):
    set_seed(seed=args.seed)
    single = world_size == 1

    if not single:  # single-process runs skip DDP entirely (plain, exit-clean)
        init_distributed_mode(rank, world_size, port=args.port)

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(rank)

    if rank == 0:
        os.makedirs(args.save_dir, exist_ok=True)
        file_name = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        log_file = os.path.join(args.save_dir, file_name + ".log")
        sys.stdout = Logger(log_file)                      # tee stdout to log
        sys.stderr = Logger(log_file, stream=sys.stderr)   # tee stderr to the same log

    # data ------------------------------------------------------------------
    X_train, y_train, X_val, y_val = load_split(args.data_dir)
    if rank == 0:
        print(f"Trainset: {len(X_train)}, Valset: {len(X_val)}")

    # model -----------------------------------------------------------------
    model = build_model(args.model, input_dim=args.input_dim, hidden_dim=args.hidden_dim,
                        out_dim=1, pooling_ratio=args.pooling_ratio,
                        n_layers=args.layers, n_head=8, slice_num=32)
    model = model.to(device)
    if not single:
        model = DDP(model, device_ids=[rank] if device.type == "cuda" else [])
    params_m = sum(p.numel() for p in model.parameters()) / 1e6

    if rank == 0:
        print("=" * 70)
        print(f"Model: {args.model} | Params: {params_m:.2f}M | "
              f"Batch: {args.batch_size} (Accum={args.accumulation_steps}) | LR: {args.lr}")
        print(f"Trainable params: {count_parameters(model)}")
        print(f"CPUs: {args.num_cpus} | Epochs: {args.epochs}")
        print("=" * 70)

    # dataloaders -----------------------------------------------------------
    transform = Standardize()
    train_dataset = STLGraphDataset(X_train, y_train, args.input_dim, transform,
                                    simplify_size=args.simplify_size)
    val_dataset = STLGraphDataset(X_val, y_val, args.input_dim, transform,
                                  simplify_size=args.simplify_size)

    if single:
        train_sampler = None  # sequential order, no sharding needed
        val_sampler = None
    else:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size,
                                           rank=rank, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size,
                                         rank=rank, shuffle=False)

    prefetch = 2 if args.num_cpus > 0 else None
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              num_workers=args.num_cpus, sampler=train_sampler,
                              shuffle=(single and args.batch_size == 1),
                              pin_memory=False, prefetch_factor=prefetch)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            num_workers=args.num_cpus, sampler=val_sampler,
                            shuffle=False, pin_memory=False, prefetch_factor=prefetch)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10)

    start_epoch = 0
    best = 1e6
    if args.resume:
        model, optimizer, start_epoch, lr_scheduler = load_checkpoint(
            model, optimizer, lr_scheduler,
            os.path.join(args.save_dir, "checkpoint.pth"))
        if rank == 0:
            print(f"Resuming from epoch {start_epoch}...")

    train_loss_history, val_loss_history = [], []

    print("Begin Training....")

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        if not single:
            train_sampler.set_epoch(epoch)
        model.train()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(rank)

        train_loss_epoch = 0.0
        optimizer.zero_grad()

        for i, data in enumerate(train_loader):
            data = data.to(device)

            is_accum_step = ((i + 1) % args.accumulation_steps == 0) or \
                            ((i + 1) == len(train_loader))
            with contextlib.ExitStack() as stack:
                if not single and not is_accum_step:
                    # skip gradient sync inside the accumulation window
                    stack.enter_context(model.no_sync())
                outputs = model(data)
                loss = criterion(outputs, data.y.unsqueeze(1))
                loss = loss / args.accumulation_steps
                loss.backward()

            train_loss_epoch += loss.item() * args.accumulation_steps

            if is_accum_step:
                optimizer.step()
                optimizer.zero_grad()

        train_loss_avg = train_loss_epoch / len(train_loader)

        # --- validation ---
        model.eval()
        y_true_list = []
        y_pred_list = []
        latencies = []

        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)

                # forward-only latency; end-to-end per-car inference time
                # (mesh loading + QEM + forward) is reported in the paper
                if device.type == "cuda":
                    torch.cuda.synchronize(rank)
                start_infer = time.time()

                outputs = model(data)

                if device.type == "cuda":
                    torch.cuda.synchronize(rank)
                end_infer = time.time()
                latency_ms = (end_infer - start_infer) * 1000 / data.y.size(0)
                latencies.append(latency_ms)

                y_pred_list.append(outputs.detach().cpu())
                y_true_list.append(data.y.unsqueeze(1).detach().cpu())

        # global metrics across all ranks (no-op for world_size=1)
        y_pred = gather_tensor(torch.cat(y_pred_list, dim=0), device)
        y_true = gather_tensor(torch.cat(y_true_list, dim=0), device)

        global_mse = torch.mean((y_true - y_pred) ** 2).item()
        val_loss_avg = global_mse  # scheduler monitors validation MSE

        global_rmse = np.sqrt(global_mse)
        global_mae = torch.mean(torch.abs(y_true - y_pred)).item()
        global_max_ae = torch.max(torch.abs(y_true - y_pred)).item()

        # R2 score
        ss_res = torch.sum((y_true - y_pred) ** 2)
        ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
        global_r2 = (1 - ss_res / ss_tot).item()

        # relative errors
        global_rel_l2 = (torch.norm(y_true - y_pred, p=2) / torch.norm(y_true, p=2)).item()
        global_rel_l1 = (torch.norm(y_true - y_pred, p=1) / torch.norm(y_true, p=1)).item()

        # inference cost
        avg_latency = np.mean(latencies)
        if device.type == "cuda":
            max_memory_alloc = torch.cuda.max_memory_allocated(rank) / (1024 ** 2)  # MB
        else:
            max_memory_alloc = 0.0

        lr_scheduler.step(val_loss_avg)

        # --- logging & checkpoints (rank 0 only) ---
        if rank == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            print("[{}/{}] Train:{:.3e} Val:{:.3e} | LR:{:.2e} | Time:{:.1f}s".format(
                epoch + 1, args.epochs, train_loss_avg, val_loss_avg,
                current_lr, time.time() - t0))

            print(f"MSE={global_mse:.4e} | RMSE={global_rmse:.4e} | MAE={global_mae:.4e} | "
                  f"Max AE={global_max_ae:.4e} | R2={global_r2:.4f}")
            print(f"Rel Errors: Rel L2={global_rel_l2:.4e} | Rel L1={global_rel_l1:.4e}")
            print(f"Costs     : Params={params_m:.2f}M | Latency={avg_latency:.2f}ms/sample | "
                  f"Peak Mem={max_memory_alloc:.0f}MB")
            print("-" * 70)

            if epoch > 5 and (epoch + 1) % 25 == 0:
                save_checkpoint(model, optimizer, epoch, lr_scheduler,
                                os.path.join(args.save_dir, "checkpoint.pth"))

            if epoch > 5 and val_loss_avg < best:
                best = val_loss_avg
                torch.save(model.state_dict(), os.path.join(args.save_dir, "model_best.pt"))
                print(f"--> Best Model Saved! Val Loss: {val_loss_avg:.3e}")

            train_loss_history.append(train_loss_avg)
            val_loss_history.append(val_loss_avg)
            plot_loss(train_loss_history, val_loss_history, args.save_dir)

    if not single:
        dist.destroy_process_group()


def main():
    args = parse_args()
    print(f"Training on {args.data_dir} with model {args.model}, world_size={args.world_size}")
    if args.world_size == 1:
        train(0, 1, args)  # plain single-process run (no mp.spawn, exits cleanly)
    else:
        mp.spawn(train, args=(args.world_size, args), nprocs=args.world_size, join=True)


if __name__ == "__main__":
    main()
