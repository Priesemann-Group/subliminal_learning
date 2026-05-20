#!/usr/bin/env python3
"""Command-line runner for unified subliminal-learning experiments.

This script intentionally does one run per invocation. For sweeps, call it repeatedly
from a shell/Slurm loop with different flags. Each invocation appends rows to the
per-seed output files:

  outdir/seed_000123/runs.csv
  outdir/seed_000123/layer_metrics.csv
  outdir/seed_000123/perturbations.csv
  outdir/seed_000123/configs.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from subliminal_core import (
    ExperimentConfig,
    parse_int_list,
    parse_json_or_path,
    parse_perturb_shorthand,
    run_experiment,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Unified subliminal-learning runner: split-head teacher/student with flexible MLP/CNN specs."
    )

    # I/O and reproducibility
    p.add_argument("--outdir", type=str, default="./outputs")
    p.add_argument("--data-dir", type=str, default="./MNIST_DATA")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", type=str, default="auto", help="auto, cpu, cuda, cuda:0, ...")
    p.add_argument("--sweep-name", type=str, default="default")
    p.add_argument("--run-label", type=str, default="")
    p.add_argument(
        "--append-global",
        action="store_true",
        help="Also append to outdir/*_all_seeds.csv. Avoid this for many concurrent array jobs unless your filesystem handles concurrent appends safely.",
    )

    # Dataset
    p.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "emnist", "MNIST", "EMNIST"])
    p.add_argument("--emnist-split", type=str, default="balanced", choices=["balanced", "letters"])

    # Architectures
    p.add_argument("--teacher-type", type=str, default="mlp", choices=["mlp", "cnn"])
    p.add_argument("--student-type", type=str, default="mlp", choices=["mlp", "cnn"])
    p.add_argument(
        "--teacher-hidden-dims",
        type=str,
        default="256,256",
        help="Comma-separated MLP hidden dims. For default CNN, the last value sets the final fc feature dimension.",
    )
    p.add_argument(
        "--student-hidden-dims",
        type=str,
        default="256,256",
        help="Comma-separated MLP hidden dims. For default CNN, the last value sets the final fc feature dimension.",
    )
    p.add_argument(
        "--teacher-arch-spec",
        type=str,
        default=None,
        help="JSON string or path to JSON list describing CNN feature layers. Ignored for MLP.",
    )
    p.add_argument(
        "--student-arch-spec",
        type=str,
        default=None,
        help="JSON string or path to JSON list describing CNN feature layers. Ignored for MLP.",
    )
    p.add_argument("--m", type=int, default=10, help="Auxiliary head dimension. Default is 10.")

    # Initialisation and trainability
    p.add_argument(
        "--teacher-init",
        type=str,
        default=None,
        help="Positional e.g. A,A,A,A or named e.g. fc1:A,fc2:A,class_head:A,aux_head:A. Default: all A.",
    )
    p.add_argument(
        "--student-init",
        type=str,
        default=None,
        help="Positional or named layer init config. Sources: A, B, random. Default: all A.",
    )
    p.add_argument(
        "--teacher-trainable",
        type=str,
        default=None,
        help="Positional/named booleans. Default: all true.",
    )
    p.add_argument(
        "--student-trainable",
        type=str,
        default=None,
        help="Positional/named booleans. Default: all true.",
    )

    # Training hyperparameters
    p.add_argument("--teacher-epochs", type=int, default=5)
    p.add_argument("--student-epochs", type=int, default=5)
    p.add_argument("--data-bsize", type=int, default=1024)
    p.add_argument("--noise-bsize", type=int, default=1000)
    p.add_argument("--noise-steps", type=int, default=60)
    p.add_argument("--teacher-lr", type=float, default=1e-3)
    p.add_argument("--student-lr", type=float, default=1e-3)

    # Noise
    p.add_argument("--noise-dist", type=str, default="uniform", choices=["uniform", "normal", "perlin"])
    p.add_argument("--perlin-res", type=int, default=4)
    p.add_argument("--normalize-noise", action="store_true")
    p.add_argument("--eval-noise-batches", type=int, default=10)
    p.add_argument("--eval-noise-bsize", type=int, default=1000)

    # Perturbations. You may mix --perturb and --perturb-spec.
    p.add_argument(
        "--perturb",
        action="append",
        default=[],
        help=(
            "Repeatable shorthand: target:layers,std=...,timing=... . "
            "Example: --perturb student:aux_head,std=0.01,timing=before_student_training . "
            "Layers can be aux_head or fc1+aux_head or all."
        ),
    )
    p.add_argument(
        "--perturb-spec",
        type=str,
        default=None,
        help=(
            "JSON string or path to JSON list of perturb specs. "
            "Example: '[{\"target\":\"student\",\"layers\":[\"aux_head\"],\"std\":0.01,\"timing\":\"before_student_training\"}]'"
        ),
    )

    return p.parse_args()


def load_perturb_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    if args.perturb_spec:
        loaded = parse_json_or_path(args.perturb_spec)
        if isinstance(loaded, dict):
            specs.append(loaded)
        elif isinstance(loaded, list):
            specs.extend(loaded)
        else:
            raise ValueError("--perturb-spec must be a JSON object or list of objects.")
    for item in args.perturb:
        specs.append(parse_perturb_shorthand(item))
    return specs


def main() -> None:
    args = parse_args()
    teacher_hidden_dims = parse_int_list(args.teacher_hidden_dims, default=[256, 256])
    student_hidden_dims = parse_int_list(args.student_hidden_dims, default=[256, 256])
    teacher_arch_spec = parse_json_or_path(args.teacher_arch_spec)
    student_arch_spec = parse_json_or_path(args.student_arch_spec)
    perturb_specs = load_perturb_specs(args)

    config = ExperimentConfig(
        dataset=args.dataset.lower(),
        data_dir=args.data_dir,
        outdir=args.outdir,
        emnist_split=args.emnist_split,
        num_workers=args.num_workers,
        seed=args.seed,
        device=args.device,
        teacher_type=args.teacher_type,
        student_type=args.student_type,
        teacher_hidden_dims=teacher_hidden_dims,
        student_hidden_dims=student_hidden_dims,
        teacher_arch_spec=teacher_arch_spec,
        student_arch_spec=student_arch_spec,
        m=args.m,
        teacher_init=args.teacher_init,
        student_init=args.student_init,
        teacher_trainable=args.teacher_trainable,
        student_trainable=args.student_trainable,
        teacher_epochs=args.teacher_epochs,
        student_epochs=args.student_epochs,
        data_bsize=args.data_bsize,
        noise_bsize=args.noise_bsize,
        noise_steps=args.noise_steps,
        teacher_lr=args.teacher_lr,
        student_lr=args.student_lr,
        noise_dist=args.noise_dist,
        perlin_res=args.perlin_res,
        normalize_noise=args.normalize_noise,
        eval_noise_batches=args.eval_noise_batches,
        eval_noise_bsize=args.eval_noise_bsize,
        perturb_specs=perturb_specs,
        sweep_name=args.sweep_name,
        run_label=args.run_label,
        append_global=args.append_global,
    )

    result = run_experiment(config)
    run_row = result["run_row"]
    print(
        "[Summary] "
        f"run_id={result['run_id']} | "
        f"teacher_acc={run_row['teacher_init_acc']:.4f}->{run_row['teacher_final_acc']:.4f} | "
        f"student_acc={run_row['student_init_acc']:.4f}->{run_row['student_final_acc']:.4f} | "
        f"student_aux={run_row['student_init_aux_loss']:.6g}->{run_row['student_final_aux_loss']:.6g}"
    )


if __name__ == "__main__":
    main()
