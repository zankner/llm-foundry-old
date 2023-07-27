import os
import platform
import tempfile
from argparse import ArgumentParser
from typing import Dict, Optional, Iterable, List

import wandb
from streaming import MDSWriter, StreamingDataset
import torch
import pathos.multiprocessing as multiprocessing
from torch.utils.data import DataLoader
from tqdm import tqdm
from composer.utils import get_file

torch.multiprocessing.set_sharing_strategy('file_system')


def get_sample(idx, dataset):
    return dataset[idx]


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
        #with multiprocessing.ProcessingPool() as pool:
        #    samps = pool.map(lambda idx: dataset[idx], batch)
        for sample_idx in batch:
            if truncate_num_samples is not None and n_samples == truncate_num_samples:
                return
            n_samples += 1
            sample = dataset[sample_idx]
            sample = {"tokens": sample["tokens"]}
            yield sample


if __name__ == "__main__":

    parser = ArgumentParser()
    # Dataset args
    parser.add_argument("--download-local", type=str, required=True)
    parser.add_argument("--num-workers", type=int, default=64)

    # Domain args
    parser.add_argument("--proxy-ckpt", type=str, required=True)
    parser.add_argument("--proxy-name", type=str, required=True)
    parser.add_argument("--pruned-num-tokens",
                        type=str,
                        required=True,
                        choices=["2B", "5B", "20B"])

    # Upload args
    parser.add_argument("--upload-base", type=str, required=True)

    # Misc
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-name", type=str, default=None)
    args = parser.parse_args()

    use_wandb = not args.no_wandb
    if use_wandb:
        assert args.wandb_name is not None, "Wandb name necessary if using for logging"

        wandb.init(name=args.wandb_name,
                   project="pre-process-data",
                   entity="mosaic-ml")

    # Loading the data trajectory
    with tempfile.NamedTemporaryFile() as tmp_file:
        get_file(os.path.join("oci://mosaicml-internal-checkpoints",
                              args.proxy_ckpt),
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
    num_samples = len(data_trajectory) * len(data_trajectory[0])
    print(f"Number of subsampled documents: {num_samples}")

    streaming_data = StreamingDataset(
        local=args.download_local,
        #split="train",
        shuffle=False,
    )

    samples = generate_samples(streaming_data,
                               data_trajectory=data_trajectory,
                               truncate_num_samples=None)
    columns = {'tokens': 'bytes'}

    upload_name = f"{args.pruned_num_tokens}-final-tokens-pruned-from-{args.proxy_name}"
    upload_remote = os.path.join("oci://mosaicml-internal-amortized-obs",
                                 args.upload_base, upload_name)
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
