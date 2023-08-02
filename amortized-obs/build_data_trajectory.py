import os
import tempfile
from argparse import ArgumentParser
from typing import Dict, Optional, Iterable, List

import wandb
from streaming import MDSWriter, StreamingDataset
import torch
from tqdm import tqdm
from composer.utils import get_file

from pretrain_utils import (CKPT_BASE, build_proxy_base, build_ref_base,
                            build_remote_base)

torch.multiprocessing.set_sharing_strategy('file_system')


def generate_samples(
        dataset: StreamingDataset,
        data_trajectory: List,
        truncate_num_samples: Optional[int] = None
) -> Iterable[Dict[str, bytes]]:
    """Generator over samples of a dataloader.

    Args:
       loader (DataLoader): A dataloader emitting batches like {key: [sample0_bytes, sample1_bytes, sample2_bytes, ...]}
       truncate_num_samples (Optional[int]): An optional # of samples to stop at.

    Yields:
        Sample dicts.
    """
    n_samples = 0
    for batch in data_trajectory:
        for sample_idx in batch:
            if truncate_num_samples is not None and n_samples == truncate_num_samples:
                return
            n_samples += 1
            sample = dataset[sample_idx]
            sample = {
                "tokens": sample["tokens"],
                "idx": sample["idx"],
                "uids": sample["uids"]
            }
            if sample_idx != sample["idx"]:
                raise ValueError("All sample indices should be the same")
            yield sample


def fetch_data_trajectory(proxy_ckpt: str):
    # Loading the data trajectory
    with tempfile.NamedTemporaryFile() as tmp_file:
        get_file(os.path.join("oci://mosaicml-internal-checkpoints",
                              proxy_ckpt),
                 tmp_file.name,
                 overwrite=True)
        proxy_ckpt = torch.load(tmp_file.name, map_location="cpu")

    data_trajectories = [
        alg_state[1]["data_trajectory"]
        for alg_state in proxy_ckpt["state"]["algorithms"]
        if alg_state[0] == "RestrictedHoldOut"
    ]
    assert len(
        data_trajectories
    ) == 1, f"Model ckpt must have one and only one RHO algorithm state, instead found {len(data_trajectories)}"
    data_trajectory = data_trajectories[0]
    return data_trajectory


if __name__ == "__main__":

    parser = ArgumentParser()
    # Dataset args
    parser.add_argument("--dataset", type=str, default="pile")
    parser.add_argument("--download-local", type=str, required=True)
    parser.add_argument("--num-workers", type=int, default=64)

    # Holdout args
    parser.add_argument("--holdout-num-tokens",
                        type=str,
                        choices=["2B", "5B", "20B", "26B"])

    # Reference args
    parser.add_argument("--ref-model-size", type=str, choices=["125M", "250M"])
    parser.add_argument("--ref-num-tokens",
                        type=str,
                        choices=["2B", "5B", "20B", "26B"])

    # Proxy args
    parser.add_argument("--proxy-model-size",
                        type=str,
                        choices=["125M", "250M", "1B"])
    parser.add_argument("--proxy-num-tokens",
                        type=str,
                        choices=["2B", "5B", "20B", "26B"])
    parser.add_argument(
        "--full-batch-size",
        help="Batch size for points to be labeled that will then be pruned",
        type=int,
        choices=[1024, 2048, 4096])
    parser.add_argument("--num-pplx-filter", type=int, default=0)
    parser.add_argument("--selection-algo",
                        type=str,
                        required=True,
                        choices=["rho", "hard-mine", "easy-mine"
                                ])  # Treat baseline as a selection algo

    # Upload args
    parser.add_argument("--upload-base", type=str, required=True)

    # Misc
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    use_wandb = not args.no_wandb
    if use_wandb:
        assert args.wandb_name is not None, "Wandb name necessary if using for logging"

        wandb.init(name=args.wandb_name,
                   project="pre-process-data",
                   entity="mosaic-ml")

    # Building proxy run name
    if args.selection_algo == "rho":
        assert args.ref_num_tokens is not None and args.ref_model_size is not None

    proxy_run_base = build_proxy_base(args.selection_algo,
                                      args.proxy_num_tokens,
                                      args.proxy_model_size,
                                      args.full_batch_size,
                                      args.num_pplx_filter)
    proxy_run_name = f"proxy-{args.dataset}-{proxy_run_base}"
    if args.selection_algo == "rho":
        ref_run_base = build_ref_base(args.ref_num_tokens, args.ref_model_size)
        proxy_run_name += f"-{ref_run_base}"

    proxy_ckpt = os.path.join(CKPT_BASE, args.dataset, "proxy",
                              f"{proxy_run_name}-sd-{args.seed}", "ckpts",
                              "latest-rank0.pt.symlink")
    print(f"Loading proxy ckpt from {proxy_ckpt}")

    data_trajectory = fetch_data_trajectory(proxy_ckpt)
    num_samples = len(data_trajectory) * len(data_trajectory[0])
    print(f"Number of subsampled documents: {num_samples}")

    streaming_data = StreamingDataset(
        local=args.download_local,
        split="train",
        shuffle=False,
    )

    samples = generate_samples(streaming_data,
                               data_trajectory=data_trajectory,
                               truncate_num_samples=None)
    columns = {'tokens': 'bytes', 'idx': 'int', 'uids': 'ndarray:int32'}

    upload_base = build_remote_base(num_holdout_tokens=args.holdout_num_tokens,
                                    dataset=args.dataset)
    upload_remote = os.path.join(
        upload_base, "pruned",
        f"{args.ref_num_tokens}-final-tokens-pruned-from-{proxy_run_base}-sd-{args.seed}"
    )

    print(f"Uploading samples to {upload_remote}")
    with MDSWriter(
            columns=columns,
            out=os.path.join(upload_remote, "train"),
            compression="zstd",
            max_workers=args.num_workers,
    ) as streaming_writer:
        for step, sample in enumerate(
                tqdm(samples, desc="prune", total=num_samples, leave=True)):
            streaming_writer.write(sample)
            if use_wandb and step % 1_000 == 0:
                wandb.log(({'step': step, 'progress': step / num_samples}))
