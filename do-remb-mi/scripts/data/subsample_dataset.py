import os
import warnings
import platform
from argparse import ArgumentParser
from typing import Dict, Optional, Iterable

from streaming import MDSWriter, StreamingDataset, Stream
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerBase
import wandb
import numpy as np

GLOBAL_SEED = 42

PILE_DATA_SOURCES = [
    "Pile-CC", "PubMed Central", "Books3", "OpenWebText2", "ArXiv", "Github",
    "FreeLaw", "StackExchange", "USPTO Backgrounds", "PubMed Abstracts",
    "Gutenberg (PG-19)", "OpenSubtitles", "Wikipedia (en)", "DM Mathematics",
    "Ubuntu IRC", "BookCorpus2", "EuroParl", "HackerNews", "YoutubeSubtitles",
    "PhilPapers", "NIH ExPorter", "Enron Emails"
]


def build_dataloader(dataset, batch_size) -> DataLoader:
    # Multiple workers is only supported on linux machines
    if 'linux' in platform.platform().lower():
        num_workers = 64  # type: ignore
    else:
        num_workers = 0

    # If using multiple workers, configure each worker to prefetch as many samples as it can, up to
    # the aggregate device batch size
    # If not using workers, the torch DataLoader expects the default value for prefetch_factor,
    # which non-intuitively must be 2.
    prefetch_factor = max(1, 2 * batch_size //
                          num_workers) if num_workers > 0 else 2

    return DataLoader(
        dataset=dataset,
        sampler=None,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    )


def generate_samples(
        loader: DataLoader,
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
    for batch in loader:
        keys = list(batch.keys())
        current_bs = len(batch[keys[0]])
        for idx in range(current_bs):
            if truncate_num_samples is not None and n_samples == truncate_num_samples:
                return
            domain_idx = batch["domain_idx"][idx].item()
            n_samples += 1
            yield {
                **{
                    k: v[idx] for k, v in batch.items() if k != "domain_idx"
                }, "domain_idx": domain_idx
            }, domain_idx


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--remote-base", type=str, required=True)
    parser.add_argument("--local-base", type=str, default="/tmp/domain-local")
    parser.add_argument("--num-samples",
                        type=str,
                        required=True,
                        choices=["1K", "5K", "10K", "100K"])
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-domains", type=int, required=True)
    parser.add_argument("--splits", type=str, nargs="+", default=["train"])
    parser.add_argument("--sample-dist",
                        type=str,
                        required=True,
                        choices=["uniform", "original"])
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-name", type=str, default=None)
    args = parser.parse_args()

    use_wandb = not args.no_wandb
    if use_wandb:
        assert args.wandb_name is not None

        wandb.init(name=args.wandb_name,
                   project="doremi-preprocess",
                   entity="mosaic-ml")

    if args.num_samples == "100K":
        int_num_samples = 100_000 * args.batch_size
    elif args.num_samples == "1K":
        int_num_samples = 1_000 * args.batch_size
    elif args.num_samples == "5K":
        int_num_samples = 5_000 * args.batch_size
    elif args.num_samples == "10K":
        int_num_samples = 10_000 * args.batch_size

    for split in args.splits:
        print(f"Subsampling split {split}")
        streams = []
        for domain_idx in range(args.num_domains):
            if args.sample_dist == "uniform":
                proportion = 1 / args.num_domains
            if args.sample_dist == "original":
                proportion = None
            streams.append(
                Stream(remote=os.path.join(args.remote_base,
                                           f"domain-{domain_idx}"),
                       local=os.path.join(args.local_base,
                                          f"domain-{domain_idx}"),
                       split=split,
                       proportion=proportion))
        # Might change shuffling later if doing multiple seeds
        streaming_data = StreamingDataset(shuffle=True,
                                          shuffle_seed=GLOBAL_SEED,
                                          shuffle_algo="py1b",
                                          streams=streams)

        loader = build_dataloader(streaming_data, batch_size=512)
        samples = generate_samples(loader, truncate_num_samples=None)

        columns = {'tokens': 'bytes', 'domain_idx': 'int'}
        denominator = int_num_samples

        writers = [
            MDSWriter(columns=columns,
                      out=os.path.join("/tmp", "domains",
                                       f"domain-{domain_idx}", split),
                      compression="zstd")
            for domain_idx in range(args.num_domains)
        ]
        for step, (sample, domain_idx) in enumerate(
                tqdm(samples, desc=split, total=denominator, leave=True)):

            if step >= int_num_samples:
                break

            writers[domain_idx].write(sample)

            if use_wandb and step % 100 == 0:
                wandb.log(({'step': step, 'progress': step / denominator}))

        for writer in writers:
            writer.finish()
