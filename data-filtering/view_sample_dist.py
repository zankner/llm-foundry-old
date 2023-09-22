import os
import tempfile
import random
from argparse import ArgumentParser
from collections import defaultdict
from typing import Dict, Optional, Iterable, List
import pickle

import wandb
from streaming import StreamingDataset
import torch
from tqdm import tqdm
from composer.utils import get_file
import matplotlib.pyplot as plt

from pretrain_utils import (CKPT_BASE, build_proxy_base)

torch.multiprocessing.set_sharing_strategy('file_system')

default_proportions = {
    "Pile-CC": 0.16033764323535438,
    "PubMed Central": 0.12679232727387502,
    "Books3": 0.11423307622919882,
    "OpenWebText2": 0.09005721466468448,
    "ArXiv": 0.1021089536218925,
    "Github": 0.0981499534730727,
    "FreeLaw": 0.05542629815635358,
    "StackExchange": 0.05583781478182942,
    "USPTO Backgrounds": 0.02866197012430996,
    "PubMed Abstracts": 0.024361277203368642,
    "Gutenberg (PG-19)": 0.0207452466409499,
    "OpenSubtitles": 0.016519108571200595,
    "Wikipedia (en)": 0.03659153576868448,
    "DM Mathematics": 0.02141010912586229,
    "Ubuntu IRC": 0.01165247234699381,
    "BookCorpus2": 0.00691818686672903,
    "EuroParl": 0.00870412929818324,
    "HackerNews": 0.006170615190153391,
    "YoutubeSubtitles": 0.008045373316346893,
    "PhilPapers": 0.003671285162979514,
    "NIH ExPorter": 0.0021772150740052352,
    "Enron Emails": 0.0014281938739721515
}


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
    #n_samples = 0
    #for sample_idx in data_trajectory:
        #if truncate_num_samples is not None and n_samples == truncate_num_samples:
            #return
        #n_samples += 1
        #sample = dataset[sample_idx]
        #print(sample)
        #sample_idx = sample["idx"]
        #sample = {"uids": sample["uids"]}
        #if sample_idx != sample["idx"]:
            #raise ValueError("All sample indices should be the same")
        #yield sample
    
    n_samples = 0
    for batch in data_trajectory:
        for sample_idx in batch:
            if truncate_num_samples is not None and n_samples == truncate_num_samples:
                return
            n_samples += 1
            sample = dataset[sample_idx]
            stored_sample_idx = sample["idx"]
            sample = {"uids": sample["uids"]}
            if sample_idx != stored_sample_idx:
                raise ValueError("All sample indices should be the same")
            yield sample


def fetch_data_trajectory(proxy_ckpt: str):
    # Loading the data trajectory
    with tempfile.NamedTemporaryFile() as tmp_file:
        get_file(proxy_ckpt, tmp_file.name, overwrite=True)
        proxy_ckpt = torch.load(tmp_file.name, map_location="cpu")

    data_trajectories = [
        alg_state[1]["data_trajectory"]
        for alg_state in proxy_ckpt["state"]["algorithms"]
        if alg_state[0] == "OnlineBatchSelection"
    ]
    assert len(
        data_trajectories
    ) == 1, f"Model ckpt must have one and only one OnlineBatchSelection algorithm state, instead found {len(data_trajectories)}"
    data_trajectory = data_trajectories[0]
    return data_trajectory


def fetch_uid_to_provenance(uid_to_provenance_path):
    # Loading the data trajectory
    with tempfile.NamedTemporaryFile() as tmp_file:
        get_file(uid_to_provenance_path, tmp_file.name, overwrite=True)
        with open(tmp_file.name, "rb") as f:
            uid_to_provenance = pickle.load(f)
    return uid_to_provenance


if __name__ == "__main__":

    parser = ArgumentParser()
    # Dataset args
    parser.add_argument("--dataset", type=str, default="pile")
    parser.add_argument("--download-local", type=str, required=True)
    parser.add_argument("--num-workers", type=int, default=64)

    # Holdout args
    parser.add_argument("--holdout-num-tokens",
                        type=str,
                        choices=["2B", "5B", "20B", "26B", "130B"])

    # Reference args
    parser.add_argument("--ref-model-size", type=str, choices=["125M", "250M"])
    parser.add_argument("--ref-num-tokens",
                        type=str,
                        choices=["2B", "5B", "20B", "26B", "130B"])

    # Proxy args
    parser.add_argument("--proxy-model-size",
                        type=str,
                        choices=["125M", "250M", "1B"])
    parser.add_argument("--proxy-num-tokens",
                        type=str,
                        choices=["2B", "5B", "20B", "26B", "130B"])
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

    # Provenance args
    parser.add_argument("--uid-to-provenance-path", type=str, required=True)
    parser.add_argument("--visualize-num-samples", type=int, required=True)

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
                                      args.num_pplx_filter, args.ref_num_tokens,
                                      args.ref_model_size)
    run_name = f"proxy-{args.dataset}-{proxy_run_base}-holdt-{args.holdout_num_tokens}"

    proxy_ckpt = os.path.join(CKPT_BASE, args.dataset, "proxy",
                              f"{run_name}-sd-{args.seed}", "ckpts",
                              "latest-rank0.pt.symlink")
    print(f"Loading proxy ckpt from {proxy_ckpt}")

    data_trajectory = fetch_data_trajectory(proxy_ckpt)
    data_trajectory = random.sample(data_trajectory, args.visualize_num_samples)
    num_samples = len(data_trajectory) * len(data_trajectory[0])
    print(f"Getting sampling probs for {num_samples} samples")

    streaming_data = StreamingDataset(
        local=args.download_local,
        split="train",
        shuffle=False,
    )

    samples = generate_samples(streaming_data,
                               data_trajectory=data_trajectory,
                               truncate_num_samples=None)

    uid_to_provenance = fetch_uid_to_provenance(args.uid_to_provenance_path)

    provenance_counts = defaultdict(int)
    for step, sample in enumerate(
            tqdm(samples, desc="prune", total=num_samples, leave=True)):

        for uid in sample["uids"]:
            provenance_counts[uid_to_provenance[uid]] += 1

    total_uids = sum(provenance_counts.values())
    provenance_counts = {
        k: v / total_uids for k, v in provenance_counts.items()
    }
    # Sort the dictionary by value in descending order
    sorted_data = sorted(provenance_counts.items(),
                         key=lambda x: x[1],
                         reverse=True)

    # Extract the keys (names) and values (counts)
    names, pruned_counts = zip(*sorted_data)
    og_counts = [default_proportions[name] for name in names]

    # Create a vertical bar plot
    plt.bar(names, pruned_counts, alpha=0.5, label="Subsampled")
    plt.bar(names, og_counts, alpha=0.5, label="Original")

    # Label the axes
    plt.ylabel('Dataset Proportion')
    plt.xlabel('Provenance')
    plt.xticks(rotation=90)
    plt.title(run_name)

    # Add legend
    plt.legend()

    # Show the plot
    plt.tight_layout()
    plt.savefig("sample-vis.png")

    # Printing out sampling probs
    dict_counts = {name: count for name, count in zip(names, pruned_counts)}
    print(dict_counts)