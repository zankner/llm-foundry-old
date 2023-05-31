import os
import warnings
import platform
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Iterable, Optional, Union

import datasets as hf_datasets
from streaming import MDSWriter, StreamingDataset
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerBase
import wandb
import numpy as np


# When creating base version use index as uid for future reference
class StreamingDatasetIndexed(StreamingDataset):

    def __getitem__(self, index: int):
        """Get sample by global index.
        Args:
            index (int): Sample index.
        Returns:
            Dict[str, Any]: Column name with sample data.
        """
        shard, index_in_shard = self.index.find_sample(index)
        reader = self.shards[shard]
        return reader[index_in_shard], index


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
            yield {k: v[idx] for k, v in batch.items()}


class ConcatTokensDataset(IterableDataset):
    """An IterableDataset that returns token samples for MDSWriter.

    Returns dicts of {'tokens': bytes}

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
        dataset,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
        bos_text: str,
        eos_text: str,
        no_wrap: bool,
    ):
        self.dataset = dataset
        self.tokenizer = tokenizer
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        self.max_length = max_length
        self.bos_text = bos_text
        self.eos_text = eos_text
        self.should_wrap = not no_wrap

        self.bos_tokens = self.tokenizer(self.bos_text,
                                         truncation=False,
                                         padding=False,
                                         add_special_tokens=False)['input_ids']
        if len(self.bos_tokens) > 1:
            warnings.warn(
                f'You specified --concat_tokens with --bos_text, but your BOS text is not tokenizing to one token\
                , instead we got {self.bos_tokens}. Quit if this was in error.')

        self.eos_tokens = self.tokenizer(self.eos_text,
                                         truncation=False,
                                         padding=False,
                                         add_special_tokens=False)['input_ids']
        if len(self.eos_tokens) > 1:
            warnings.warn(
                f'You specified --concat_tokens with --eos_text, but your EOS text is not tokenizing to one token\
                , instead we got {self.eos_tokens}. Quit if this was in error.')

        eos_text_provided = self.eos_text != ''
        bos_text_provided = self.bos_text != ''
        test_text = self.tokenizer('')
        if len(test_text['input_ids']) > 0 and (eos_text_provided or
                                                bos_text_provided):
            message = 'both eos and bos' if eos_text_provided and bos_text_provided else (
                'eos_text' if eos_text_provided else 'bos_text')
            warnings.warn(
                f'The provided tokenizer adds special tokens, but you also specified {message}. This may result '
                'in duplicated special tokens. Please be sure this is what you intend.'
            )

    def __iter__(self) -> Iterable[Dict[str, bytes]]:

        buffer = []
        for sample in self.dataset:
            encoded = self.tokenizer(sample['text'],
                                     truncation=False,
                                     padding=False)
            iids = encoded['input_ids']
            buffer = buffer + self.bos_tokens + iids + self.eos_tokens
            while len(buffer) >= self.max_length:
                concat_sample = buffer[:self.max_length]
                buffer = buffer[self.max_length:] if self.should_wrap else []
                yield {
                    # convert to bytes to store in MDS binary format
                    'tokens': np.asarray(concat_sample).tobytes()
                }


# TODO: REFAC TO TAKE IN THE EMBEDDINGS FOR CLUSTERS
if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--remote", type=str, required=True)
    parser.add_argument("--local", type=str, default="/tmp/concat-local")
    parser.add_argument("--splits",
                        type=str,
                        nargs="+",
                        default=["train", "val", "test"])
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--tokenizer",
                        type=str,
                        default="EleutherAI/gpt-neox-20b")
    parser.add_argument("--eos-text", type=str, default="<|endoftext|>")
    parser.add_argument("--bos-text", type=str, default=None)
    parser.add_argument('--no-wrap', default=False, action='store_true')
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-name", type=str, default=None)
    args = parser.parse_args()

    if args.bos_text is None:
        args.bos_text = ''
    if args.eos_text is None:
        args.eos_text = ''

    use_wandb = not args.no_wandb
    if use_wandb:
        assert args.wandb_name is not None

        wandb.init(name=args.wandb_name,
                   project="doremi-preprocess",
                   entity="mosaic-ml")

    for split in args.splits:
        print(f"Converting split {split}")
        s3_data = StreamingDataset(remote=args.remote,
                                   local=args.local,
                                   split=split,
                                   shuffle=False)

        tokenizer = tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        tokenizer.model_max_length = int(1e30)

        data = ConcatTokensDataset(
            s3_data,
            tokenizer=tokenizer,
            max_length=args.max_length,
            bos_text=args.bos_text,
            eos_text=args.eos_text,
            no_wrap=args.no_wrap,
        )
        loader = build_dataloader(data, batch_size=512)
        samples = generate_samples(loader, truncate_num_samples=None)

        columns = {'tokens': 'bytes'}
        denominator = 210607728

        with MDSWriter(columns=columns,
                       out=os.path.join("/tmp", "pre-concat", split),
                       compression="zstd") as out:
            for step, sample in enumerate(
                    tqdm(samples, desc=split, total=denominator, leave=True)):
                out.write(sample)

                if step % 1_000 == 0:
                    wandb.log(({'step': step, 'progress': step / denominator}))
