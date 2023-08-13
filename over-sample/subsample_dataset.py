import os
import platform
from argparse import ArgumentParser
from typing import Dict, Optional, Iterable

from streaming import MDSWriter, StreamingDataset
import torch
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm

torch.multiprocessing.set_sharing_strategy('file_system')

SUBSAMPLE_DOMAIN_INFO = {
    'c4': {
        'truncate_samples':
            155613,
        'download_remote':
            'oci://mosaicml-internal-dataset-c4/sem-dedupe/23-04-05/dataset/pretokencat/EleutherAI-gpt-neox-20b/0pt8/',
        'download_split':
            'train',
        'upload_remote':
            'oci://mosaicml-internal-dataset-mpt-oversample/13B-available-tokens/c4/train'
    },
    'markdown': {
        'truncate_samples':
            165923,
        'download_remote':
            'oci://mosaicml-internal-datasets/stack-split-neox/markdown/',
        'download_split':
            None,
        'upload_remote':
            'oci://mosaicml-internal-dataset-mpt-oversample/13B-available-tokens/markdown/train'
    },
    'mc4': {
        'truncate_samples':
            3747115,
        'download_remote':
            'oci://mosaicml-internal-dataset-mc4/filtered/en/filter2-gptneox-tok/',
        'download_split':
            None,
        'upload_remote':
            'oci://mosaicml-internal-dataset-mpt-oversample/13B-available-tokens/mc4/train'
    },
    'redpajama': {
        'truncate_samples':
            1361326,
        'download_remote':
            'oci://mosaicml-internal-dataset-red-pajama/cc/gptneox-tok/',
        'download_split':
            None,
        'upload_remote':
            'oci://mosaicml-internal-dataset-mpt-oversample/13B-available-tokens/redpajama/train'
    },
    'redpajama-arxiv': {
        'truncate_samples':
            43538,
        'download_remote':
            'oci://mosaicml-internal-dataset-red-pajama/arxiv/gptneox-tok/',
        'download_split':
            None,
        'upload_remote':
            'oci://mosaicml-internal-dataset-mpt-oversample/13B-available-tokens/redpajama-arxiv/train'
    },
    'redpajama-books': {
        'truncate_samples':
            40327,
        'download_remote':
            'oci://mosaicml-internal-dataset-red-pajama/books/gptneox-tok/',
        'download_split':
            None,
        'upload_remote':
            'oci://mosaicml-internal-dataset-mpt-oversample/13B-available-tokens/redpajama-books/train'
    },
    'redpajama-stackexchange': {
        'truncate_samples':
            31837,
        'download_remote':
            'oci://mosaicml-internal-dataset-red-pajama/stackexchange/gptneox-tok/',
        'download_split':
            None,
        'upload_remote':
            'oci://mosaicml-internal-dataset-mpt-oversample/13B-available-tokens/redpajama-stackexchange/train'
    },
    'redpajama-wiki': {
        'truncate_samples':
            7557,
        'download_remote':
            'oci://mosaicml-internal-dataset-red-pajama/wikipedia/gptneox-tok/en/',
        'download_split':
            None,
        'upload_remote':
            'oci://mosaicml-internal-dataset-mpt-oversample/13B-available-tokens/redpajama-wiki/train'
    },
    's2': {
        'truncate_samples':
            75704,
        'download_remote':
            'oci://mosaicml-internal-datasets/s2orc/base/preconcat-gpt_neox/',
        'download_split':
            'train',
        'upload_remote':
            'oci://mosaicml-internal-dataset-mpt-oversample/13B-available-tokens/s2/train'
    },
    'stack': {
        'truncate_samples':
            718712,
        'download_remote':
            'oci://mosaicml-internal-datasets/stack-split-neox/',
        'download_split':
            None,
        'upload_remote':
            'oci://mosaicml-internal-dataset-mpt-oversample/13B-available-tokens/stack/train'
    }
}


def build_dataloader(dataset, batch_size, num_workers) -> DataLoader:
    # Multiple workers is only supported on linux machines
    if 'linux' in platform.platform().lower():
        num_workers = num_workers  # type: ignore
    else:
        num_workers = 0

    # If using multiple workers, configure each worker to prefetch as many samples as it can, up to
    # the aggregate device batch size
    # If not using workers, the torch DataLoader expects the default value for prefetch_factor,
    # which non-intuitively must be 2.
    prefetch_factor = max(1, 2 * batch_size //
                          num_workers) if num_workers > 0 else None

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
            n_samples += 1
            sample = {
                "tokens": batch["tokens"][idx],
            }
            yield sample


class TokensDataset(IterableDataset):
    """An IterableDataset that returns concatenated token samples for MDSWriter.

    Returns dicts of {'tokens': bytes, 'domain_idx': int}

    """

    def __init__(
        self,
        dataset: StreamingDataset,
    ):
        self.dataset = dataset

    def __iter__(self) -> Iterable[Dict[str, bytes]]:
        for sample in self.dataset:
            yield {"tokens": sample["tokens"]}


if __name__ == "__main__":

    parser = ArgumentParser()
    # Dataset args
    parser.add_argument("--download-local", type=str, default="/mnt/datateam/zack/tmp/download")
    parser.add_argument("--num-workers", type=int, default=64)

    # Domain args
    parser.add_argument("--domain", type=str, required=True, choices=list(SUBSAMPLE_DOMAIN_INFO.keys()))

    # Final args
    parser.add_argument("--available-num-tokens",
                        type=str,
                        required=True,
                        choices=["13B", "26B"])

    # Misc
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    print(f"Subsampling domain {args.domain}")

    domain_info = SUBSAMPLE_DOMAIN_INFO[args.domain]

    streaming_data = StreamingDataset(
        remote=domain_info["download_remote"],
        local=os.path.join(args.download_local, args.domain),
        split=domain_info["download_split"],
        shuffle=True,
        shuffle_algo="py1b",
        shuffle_seed=args.seed
    )
    dataset = TokensDataset(streaming_data)

    dataloader = build_dataloader(dataset, 512, args.num_workers)

    if args.available_num_tokens == "13B":
        available_num_tokens = 13_000_000_000
    else:
        raise ValueError(
            f"Invalid available num tokens: {args.available_num_tokens}")

    num_samples = domain_info["truncate_samples"]

    samples = generate_samples(dataloader, truncate_num_samples=num_samples)
    columns = {'tokens': 'bytes'}


    print(f"Uploading {num_samples} samples to {domain_info['upload_remote']}")
    with MDSWriter(
            columns=columns,
            out=domain_info["upload_remote"],
            compression="zstd",
            max_workers=args.num_workers,
    ) as streaming_writer:
        for step, sample in enumerate(
                tqdm(samples, desc="subsample", total=num_samples, leave=True)):
            streaming_writer.write(sample)
    
    print("="*20)
