import argparse
import datetime
import os
import logging
import pickle as pkl
from typing import Any, Callable, Dict, List, Mapping, Tuple, Union

import numpy as np
from streaming.base.dataset import StreamingDataset
import torch
from torch import device, Tensor
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.profiler import profile, record_function, ProfilerActivity
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '12355'
    os.environ['WORLD_SIZE'] = f'{world_size}'
    os.environ['RANK'] = f'{rank}'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


# Need dataset to return index, because we need to associate embedding w/ sample index if
# sample doesn't have UID
class StreamingDatasetIndexed(StreamingDataset):

    def __getitem__(self, index: int) -> Tuple[Dict[str, Any], int]:
        """Get sample by global index.
        Args:
            index (int): Sample index.
        Returns:
            Dict[str, Any]: Column name with sample data.
        """
        shard, index_in_shard = self.index.find_sample(index)
        reader = self.shards[shard]
        return reader[index_in_shard], index


# Utility function to send each key of a batch to the target device
def batch_to_device(batch, target_device: device):
    for key in batch:
        if isinstance(batch[key], Tensor):
            batch[key] = batch[key].to(target_device)
    return batch


# Utility function to subdivide a sequence of tokens into chunks of length ≤512. Accounts
# for length of instructions (which need to be prepended to each chunk)
def chunk_tokens_bert(text: List[str],
                      tokenizer: Callable,
                      chunk_size: int,
                      instruction: Union[str, None] = None):
    # Tokenize instruction (if we have one)
    if instruction is not None:
        instruction_tokenized = tokenizer(instruction,
                                          truncation=False,
                                          max_length=False,
                                          return_tensors='pt')
        # Strip the beginning ([CLS]) and final ([SEP]) tokens.
        instruction_tokenized = {
            k: v[:, 1:-1] for k, v in instruction_tokenized.items()
        }
        instruction_length = instruction_tokenized['input_ids'].shape[1]
    else:
        instruction_tokenized = None
        instruction_length = 0
    # Account for the size of the instruction when chunking - reduce chunk_size by
    # instruction_length
    max_length = chunk_size - instruction_length
    # Tokenize
    input_tokenized = tokenizer(text,
                                truncation=True,
                                max_length=max_length,
                                return_tensors='pt',
                                return_overflowing_tokens=True,
                                padding=True)
    # Insert the instruction into the tokenized input at index = 1 (after the [CLS]
    # token and before the input text)
    if instruction is not None:
        assert type(instruction_tokenized) is dict
        for k, instruction_value in instruction_tokenized.items():
            input_value = input_tokenized[k]
            input_tokenized[k] = torch.cat([
                input_value[:, 0:1],
                instruction_value.repeat(input_value.shape[0], 1),
                input_value[:, 1:]
            ], 1)
    return input_tokenized


# Collator for E5 model (what I'm using for text embeddings)
class E5Collator:

    def __init__(self,
                 tokenizer: Union[Callable, None] = None,
                 chunk_size: int = 512,
                 instruction: Union[str, None] = None,
                 uid_key: Union[str, None] = None) -> None:

        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.instruction = instruction
        self.uid_key = uid_key

    def __call__(self, samples: List) -> Dict:
        # Collect data into lists
        sample_indices = []
        texts = []
        uids = []
        # Iterate through samples
        for sample in samples:
            texts.append(sample[0]['text'])
            # Sample[1] is the sample index into the dataset
            sample_indices.append(sample[1])
            # Get UID if we have it
            if self.uid_key is not None:
                uids.append(sample[0][self.uid_key])
        assert self.tokenizer is not None
        # Chunk 'em
        samples_tokenized = chunk_tokens_bert(text=texts,
                                              tokenizer=self.tokenizer,
                                              chunk_size=self.chunk_size,
                                              instruction=self.instruction)
        # Count the number of chunks per sample
        _, counts = samples_tokenized['overflow_to_sample_mapping'].unique(
            return_counts=True)
        # Repeat the sample index for each chunk
        sample_indices = torch.tensor(sample_indices).repeat_interleave(
            repeats=counts)
        samples_tokenized.pop('overflow_to_sample_mapping')
        return {
            'samples': samples_tokenized,
            'indices': sample_indices,
            'uids': uids
        }


# Collator for MPT models
class MPTCollator:

    def __init__(self,
                 tokenizer: Union[Callable, None] = None,
                 chunk_size: int = 512,
                 instruction: Union[str, None] = None,
                 uid_key: Union[str, None] = None) -> None:

        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.instruction = instruction
        self.uid_key = uid_key

    def __call__(self, samples: List) -> Dict:
        # Collect data into lists
        sample_indices = []
        texts = []
        uids = []
        # Iterate through samples
        for sample in samples:
            texts.append(sample[0]['text'])
            # Sample[1] is the sample index into the dataset
            sample_indices.append(sample[1])
            # Get UID if we have it
            if self.uid_key is not None:
                uids.append(sample[0][self.uid_key])
        assert self.tokenizer is not None
        # Chunk 'em
        samples_tokenized = chunk_tokens_bert(text=texts,
                                              tokenizer=self.tokenizer,
                                              chunk_size=self.chunk_size,
                                              instruction=self.instruction)
        # Count the number of chunks per sample
        _, counts = samples_tokenized['overflow_to_sample_mapping'].unique(
            return_counts=True)
        # Repeat the sample index for each chunk
        sample_indices = torch.tensor(sample_indices).repeat_interleave(
            repeats=counts)
        samples_tokenized.pop('overflow_to_sample_mapping')
        return {
            'samples': samples_tokenized,
            'indices': sample_indices,
            'uids': uids
        }


def avg_pool_tokens(last_hidden_states: Tensor,
                    attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(
        ~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


# Function to average together embeddings for chunks from the same sample
def avg_sequences(seq_embeddings: Tensor, sample_indices: Tensor):
    curr_device = seq_embeddings.device
    try:
        sample_indices = sample_indices.to(curr_device)
    except Exception as e:
        print(
            f'sample_indices: {sample_indices}]\nwith shape {sample_indices.shape}]\non device {sample_indices.device}'
        )
        print(f'curr device: {curr_device}')
        raise e
    uniques, inverse = sample_indices.unique(return_inverse=True)
    reduce_inds = inverse.view(inverse.size(0),
                               1).expand(-1, seq_embeddings.size(1))
    mean_embeddings = torch.zeros(uniques.size(0),
                                  seq_embeddings.size(1),
                                  device=curr_device)
    mean_embeddings.scatter_reduce_(dim=0,
                                    index=reduce_inds,
                                    src=seq_embeddings,
                                    reduce='mean',
                                    include_self=False)
    return mean_embeddings


# Post-processing function for E5/bert model
def post_process_bert(last_hidden_state: Tensor, sample_indices: Tensor,
                      **kwargs):

    doc_embeddings = avg_sequences(last_hidden_state, sample_indices)
    doc_embeddings = torch.nn.functional.normalize(doc_embeddings, p=2, dim=1)

    return doc_embeddings


# Function to subdivide a batch into smaller batches
def subdivide_batch(batch: Mapping, batch_size: int):
    n_samples = batch['input_ids'].shape[0]
    # batches = []
    for i in range(0, n_samples, batch_size):
        if i + batch_size > n_samples:
            inds = (i, n_samples)
        else:
            inds = (i, i + batch_size)
        yield {k: v[inds[0]:inds[1], :] for k, v in batch.items()}
    # return batches


# E5, wrapped to conform to a more generic output format
class WrappedE5(torch.nn.Module):

    def __init__(self, model_name):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)

    @property
    def device(self):
        return next(self.parameters()).device

    def avg_pool_tokens(self, last_hidden_state: Tensor,
                        attention_mask: Tensor) -> Tensor:
        attention_mask = attention_mask.to(last_hidden_state.device)
        last_hidden = last_hidden_state.masked_fill(
            ~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        out = self.backbone(*args, **kwds)
        out['pooler_output'] = None
        out['last_hidden_state'] = self.avg_pool_tokens(
            out['last_hidden_state'], kwds['attention_mask'])
        return out


# TODO: Make this use save_dir
# Merge together np.memmap arrays from different ranks
def merge_memmaps(filename: str, rank0_filename: str,
                  destination_memmap: np.memmap) -> None:
    # Get all the memmap files
    memmap_files = [f for f in os.listdir() if filename in f]
    # Remove the rank 0 file
    memmap_files.remove(rank0_filename)
    # Iterate through the files
    for memmap_file in memmap_files:
        # Load the memmap file
        source_memmap = np.memmap(memmap_file,
                                  dtype='float32',
                                  mode="r",
                                  shape=destination_memmap.shape)
        # Find the indices that are not all zeros in the source memmap
        source_inds = source_memmap[:, 0].nonzero()[0]
        # Copy from the source to the main memmap
        destination_memmap[source_inds, :] = source_memmap[source_inds, :]
        destination_memmap.flush()
        # Remove the source memmap from memory and disk
        del source_memmap
        os.remove(memmap_file)


# TODO: Make this use save_dir
# TODO: Add sanity checks
# Merge together the ind2uid and uid2ind files from different ranks (NOT merging uid2ind
# with ind2uid)
def merge_ind2uids() -> None:
    # Get all the ind2uid files
    ind2uid_files = [f for f in os.listdir() if 'ind2uid' in f]
    # Map from {sample's index into dataset and embedding array: UID}
    ind2uid = {}
    # Map from {sample UID: index into dataset and embedding array}
    uid2ind = {}
    # Iterate through the files
    for ind2uid_file in ind2uid_files:
        with open(ind2uid_file, 'rb') as f:
            curr_ind2uid = pkl.load(f)
        # Update data structures
        ind2uid.update(curr_ind2uid)
        uid2ind.update({v: k for k, v in curr_ind2uid.items()})
        os.remove(ind2uid_file)
    # Write merged files
    with open('ind2uid.pkl', 'wb') as f:
        pkl.dump(ind2uid, f, protocol=pkl.HIGHEST_PROTOCOL)
    with open('uid2ind.pkl', 'wb') as f:
        pkl.dump(uid2ind, f, protocol=pkl.HIGHEST_PROTOCOL)


# Core function for doing inference
def do_the_thing(
    rank: int,
    world_size: int,
    streaming_remote: str,
    streaming_local: str,
    save_dir: str,
    split: Union[str, None],
    file_name: str,
    model_name: str,
    embedding_dim: int,
    batch_size_dataloader: int,
    batch_size_inference: int,
    collator: Callable,
    post_processing: Callable,
    device_ids: Union[List, None] = None,
    parallel_strategy: str = 'dp',
    uid_key: str = 'uid',
    **kwargs,
):

    if parallel_strategy == 'mp':
        setup(rank, world_size)
        dist.barrier()

    # Workaround for streaming OCI bug
    for r in range(world_size):
        if rank == r:
            dataset = StreamingDatasetIndexed(
                local=streaming_local,
                remote=streaming_remote,
                split=split,
                shuffle=False,
            )
        if parallel_strategy == 'mp':
            dist.barrier()

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size_dataloader,
                                             shuffle=False,
                                             collate_fn=collator,
                                             num_workers=8)

    # Instantiate model. You will have to change this if you want to use a different model
    # to get embeddings. (Sorry!)
    if rank == 0:
        model = WrappedE5(model_name).to(rank)
    if parallel_strategy == 'mp':
        # Barrier to stop multiple ranks from trying to download model simultaneously.
        dist.barrier()
    if rank > 0:
        model = WrappedE5(model_name).to(rank)
    if parallel_strategy == 'mp':
        dist.barrier()

    if parallel_strategy == 'dp':
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    model.eval()

    use_uid = False
    dataset_len = dataset.index.total_samples
    if rank == 0:
        logging.basicConfig(filename='rank0.log',
                            encoding='utf-8',
                            level=logging.DEBUG)
    if uid_key not in dataset.shards[0].column_names:
        if rank == 0:
            logging.warning(
                f"Key {uid_key} not found in dataset. Using index as uid.")
    else:
        use_uid = True
    # File that we write to that contains embeddings
    rank_filename = f'rank{rank}_{file_name}'
    # TODO: Make this use save_dir
    emb_array = np.memmap(rank_filename,
                          dtype='float32',
                          mode='w+',
                          shape=(int(dataset_len), embedding_dim))
    ind2uid = {}

    if rank == 0:
        pbar = tqdm(total=dataset_len)
        # total = dataset_len
    else:
        pbar = None
    for batch in dataloader:
        uids = batch['uids']
        sample_indices = batch['indices']
        batch = batch['samples']
        if parallel_strategy == 'mp':
            batch = batch_to_device(
                batch, model.device)  # Move this to microbatch level
        if isinstance(
                batch,
                Mapping) and batch['input_ids'].shape[0] > batch_size_inference:
            # microbatches = subdivide_batch(batch, batch_size_inference)
            microbatches_out = []
            with torch.no_grad():
                with torch.autocast(device_type='cuda',
                                    dtype=torch.float16,
                                    enabled=True):
                    for microbatch in subdivide_batch(batch,
                                                      batch_size_inference):
                        microbatch_out = model(**microbatch)
                        microbatch_out = batch_to_device(
                            microbatch_out, torch.device('cpu'))
                        microbatches_out.append(microbatch_out)
            # Aggregate microbatches
            out = microbatches_out[0]
            for i, microbatch_out in enumerate(microbatches_out[1:], 1):
                for k, v in microbatch_out.items():
                    if v is not None:
                        out[k] = torch.cat([out[k], v], 0)
                microbatches_out[i] = []
            del microbatches_out
        else:
            with torch.no_grad():
                with torch.autocast(device_type='cuda',
                                    dtype=torch.float16,
                                    enabled=True):
                    out = model(**batch)
                    out = batch_to_device(out, torch.device('cpu'))
                # out['last_hidden_state'] = avg_pool_tokens(out['last_hidden_state'], batch['attention_mask'])
        embeddings = post_processing(**out,
                                     **batch,
                                     sample_indices=sample_indices)
        sample_indices_unique = sample_indices.unique()
        if use_uid:
            ind2uid.update({
                ind.item(): uid for ind, uid in zip(sample_indices_unique, uids)
            })

        zero_embeddings = torch.where(embeddings[:, 0] == 0)[0]
        if len(zero_embeddings) > 0:
            logging.warning(
                f"zero-value embeddings for samples {sample_indices_unique[zero_embeddings]}"
            )

        emb_array[sample_indices_unique.numpy(), :] = embeddings.numpy()

        if rank == 0:
            assert type(pbar) is tqdm
            update_size = len(sample_indices_unique)
            if parallel_strategy == 'mp':
                update_size *= world_size
            pbar.update(update_size)
            current = pbar.format_dict['n']
            if current == update_size or current % (update_size * 10):
                total = pbar.format_dict['total']
                elapsed = str(
                    datetime.timedelta(
                        seconds=pbar.format_dict['elapsed'])).split('.')[0]
                est_total = str(
                    datetime.timedelta(seconds=pbar.format_dict["total"] /
                                       pbar.format_dict["rate"])).split(".")[0]
                logging.info(
                    f'{current} of {total} Samples ---- Elapsed: {elapsed} ---- Estimated Total: {est_total}'
                )

    if rank == 0:
        assert type(pbar) is tqdm

        pbar.close()
    emb_array.flush()
    if parallel_strategy == 'mp':
        dist.barrier()

    if use_uid:
        # TODO: Make this use save_dir
        with open(f'ind2uid_rank{rank}.pkl', 'wb') as f:
            pkl.dump(ind2uid, f, protocol=pkl.HIGHEST_PROTOCOL)
        dist.barrier()

    if rank == 0:
        if parallel_strategy == 'mp':
            print("Merging arrays from different ranks")
            merge_memmaps(filename=file_name,
                          rank0_filename=rank_filename,
                          destination_memmap=emb_array)
        os.rename(rank_filename, file_name)
        if use_uid:
            merge_ind2uids()
    if parallel_strategy == 'mp':
        dist.barrier()
        cleanup()


# Post-processing functions to choose from. Add your own if you want to use a different
# embedding model!
POST_PROCESSING_FXNS = {
    'post_processing_bert': post_process_bert,
}

# Dataloader collators to choose from. Add your own if you want to use a different
# embedding model!
COLLATORS = {
    'mpt': MPTCollator,
}

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Get token wise losses for streaming dataset')
    parser.add_argument('--streaming_remote', type=str)
    parser.add_argument('--streaming_local',
                        type=str,
                        default="/tmp/streaming_dataset")
    parser.add_argument('--save_dir', default=None,
                        type=str)  # TODO: Make this work
    parser.add_argument('--file_name', default='REF_LOSSES.npy', type=str)
    parser.add_argument('--split',
                        type=str,
                        help='Dataset split (e.g. "train" or "val")')
    parser.add_argument('--model_name', default='intfloat/e5-base', type=str)
    parser.add_argument('--tokenizer', default='intfloat/e5-base', type=str)
    parser.add_argument('--max_seq_length',
                        default=512,
                        type=int,
                        help="Model's maximum accepted sequence length")
    parser.add_argument('--embedding_dim', default=768, type=int)
    parser.add_argument('--batch_size_dataloader',
                        default=320,
                        type=int,
                        help='Batch size for dataloader')
    parser.add_argument(
        '--batch_size_inference',
        default=640,
        type=int,
        help=
        'Batch size for inference, because batch size after tokenization and chunking will be >> batch size of samples in dataloader'
    )
    parser.add_argument('--world_size',
                        default=torch.cuda.device_count(),
                        type=int)
    parser.add_argument('--post_processing_fxn',
                        default='post_processing_bert',
                        type=str,
                        help='Function to post-process model outputs')
    parser.add_argument('--collator', default='e5', type=str)
    parser.add_argument(
        '--parallel_strategy',
        default='mp',
        type=str,
        help="mp (multiprocessing) or dp (Data Parallel). Don't use dp.")
    parser.add_argument(
        '--uid_key',
        type=str,
        help=
        'unique identifier for samples. If not provided, will use sample index in dataset as uid.'
    )
    args = parser.parse_args()

    if args.streaming_remote.lower() == "none":
        args.streaming_remote = None

    embedding_dim = args.embedding_dim
    instruction = args.instruction
    model_name = args.model_name
    tokenizer_name = args.tokenizer

    if tokenizer_name.lower() != "none":
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    else:
        tokenizer = None

    if args.instruction.lower == "none":
        args.instruction = None

    # Get and initialize collator
    collator = COLLATORS[args.collator.lower()](tokenizer=tokenizer,
                                                chunk_size=args.max_seq_length,
                                                instruction=args.instruction,
                                                uid_key=args.uid_key)
    # Get post-processing function
    post_processing_fxn = POST_PROCESSING_FXNS[args.post_processing_fxn]

    world_size = args.world_size
    if args.parallel_strategy == 'mp':

        mp.spawn(do_the_thing,
                 args=[
                     world_size,
                     args.streaming_remote,
                     args.streaming_local,
                     args.save_dir,
                     args.split,
                     args.file_name,
                     args.model_name,
                     args.embedding_dim,
                     args.batch_size_dataloader,
                     args.batch_size_inference,
                     collator,
                     post_processing_fxn,
                     None,
                     args.parallel_strategy,
                     args.uid_key,
                 ],
                 nprocs=world_size)
    elif args.parallel_strategy == 'dp':
        device_ids = list(range(world_size))
        args.world_size = 1
        args.collator = collator
        do_the_thing(rank=0,
                     **vars(args),
                     post_processing=post_processing_fxn,
                     device_ids=device_ids)
