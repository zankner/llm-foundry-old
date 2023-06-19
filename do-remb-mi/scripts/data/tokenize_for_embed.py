import argparse
import os
import copy
import platform
from typing import Dict, Iterable, Optional

from streaming import MDSWriter, StreamingDataset
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerBase
import wandb
import numpy as np


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
            n_samples += 1
            sample = {
                k: copy.deepcopy(v[idx]) for k, v in batch.items() if k != "uid"
            }
            sample["uid"] = copy.deepcopy(batch["uid"][idx].item())
            yield sample


class EmbedTokensDataset(IterableDataset):
    """An IterableDataset that returns token samples for MDSWriter.

    Returns dicts of {'tokens': bytes, 'token_type_ids': bytes,
                      'attention_mask':bytes, 'uid': int}

    To use data created by this class and written to MDS format:

    ```python
        import torch
        from streaming.base import StreamingDataset
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained('your/tokenizer')
        ds = StreamingDataset(local='mds-data-folder', split='val')

        # note, you need to copy the numpy array because the original is non-writeable
        # and torch does not support non-writeable tensors, so you get a scary warning and
        # if you do try to write to the tensor you get undefined behavior
        tokens = torch.from_numpy(np.frombuffer(ds[0]['tokens'], dtype=np.int64).copy())
        print(tokenizer.decode(tokens))
    ```
    """

    def __init__(
        self,
        dataset: StreamingDataset,
        tokenizer: PreTrainedTokenizerBase,
        prefix: str,
        max_length: int,
    ):
        self.dataset = dataset
        self.tokenizer = tokenizer
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        self.prefix = prefix
        self.max_length = max_length

    def __iter__(self) -> Iterable[Dict[str, bytes]]:

        for sample in self.dataset:
            uid = sample['uid']
            text = self.prefix + sample["text"]
            encoded = tokenizer(text,
                                truncation=True,
                                max_length=self.max_length,
                                return_overflowing_tokens=True,
                                padding=True)
            for input_ids, token_type_ids, attention_mask in zip(
                    encoded["input_ids"], encoded["token_type_ids"],
                    encoded["attention_mask"]):
                yield {
                    # convert to bytes to store in MDS binary format
                    'tokens': np.asarray(input_ids).tobytes(),
                    'token_type_ids': np.asarray(token_type_ids).tobytes(),
                    'attention_mask': np.asarray(attention_mask).tobytes(),
                    'uid': uid
                }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--remote", type=str, required=True)
    parser.add_argument("--local",
                        type=str,
                        default="/tmp/tokenize-for-embedding")
    parser.add_argument("--splits", type=str, nargs="+", default=["train"])
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--tokenizer", type=str, default="intfloat/e5-large")
    parser.add_argument("--prefix", type=str, default="query: ")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-name", type=str, default="tokenize-for-embed")
    args = parser.parse_args()

    if not args.no_wandb:
        wandb.init(name=args.wandb_name,
                   project="doremi-preprocess",
                   entity="mosaic-ml")

    tokenizer = AutoTokenizer.from_pretrained("intfloat/e5-base")
    tokenizer.model_max_length = int(1e30)

    for split in args.splits:
        print(f"Converting split {split}")
        data = StreamingDataset(remote=args.remote,
                                local=os.path.join(args.local, "streaming"),
                                split=split,
                                shuffle=False)
        n_samples = data.index.total_samples
        denominator = n_samples * 6212 // (512 * 4)
        data = EmbedTokensDataset(data,
                                  tokenizer,
                                  prefix=args.prefix,
                                  max_length=args.max_length)
        loader = build_dataloader(data, batch_size=args.max_length)
        samples = generate_samples(loader, truncate_num_samples=None)

        columns = {
            'tokens': 'bytes',
            'token_type_ids': 'bytes',
            'attention_mask': 'bytes',
            'uid': 'int'
        }

        with MDSWriter(columns=columns,
                       out=os.path.join(os.path.join(args.local, "processed"),
                                        split),
                       compression="zstd") as out:
            for step, sample in enumerate(
                    tqdm(samples, desc=split, total=denominator, leave=True)):
                out.write(sample)

                if step % 1_000 == 0 and not args.no_wandb:
                    wandb.log(({'step': step, 'progress': step / denominator}))
