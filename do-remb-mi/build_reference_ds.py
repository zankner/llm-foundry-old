import argparse
import logging
import os
import pickle
from typing import Union

from torch.utils.data import DataLoader
from tqdm import tqdm

from streaming.base import MDSWriter, StreamingDataset

logger = logging.getLogger(__name__)
logging.basicConfig()
logger.setLevel(logging.INFO)


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


def running_average(avg: float, new_val: float, count: int):
    return avg * (count - 1) / count + new_val / count


def write_and_prune(
    streaming_remote: str,
    streaming_local: str,
    save_dir: str,
    name: str,
    similarity_file: str,
    keep_fraction: float,
    source_key: str,
    split: Union[str, None] = None,
    uid_key: Union[str, None] = None,
    quantiles_file: Union[str, None] = None,
):

    # similarity_filetype = None
    quantiles = None
    similarities = None
    if similarity_file.endswith('.pkl'):
        with open(similarity_file, 'rb') as handle:
            similarities = pickle.load(handle)
        logger.info(
            f'Loaded {similarity_file} with {len(similarities["similarities"])} similarities'
        )
        quantiles = similarities['quantiles']
        similarities = similarities['similarities']
    if quantiles is None:
        raise ValueError('Quantiles must be specified')
    if similarities is None:
        raise ValueError('Similarities must be specified')

    try:
        keep_threshold = quantiles[keep_fraction]
    except KeyError as e:
        logger.error(
            f'{keep_fraction} quantile not present in {similarity_file}. Please use one of:\n'
        )
        for q in quantiles.keys():
            print(q)
        return e

    def collate_fn(batch):
        return batch

    # Instantiate old dataset
    old_dataset = StreamingDatasetIndexed(local=streaming_local,
                                          remote=streaming_remote,
                                          split=split,
                                          shuffle=False)

    dataloader = DataLoader(old_dataset,
                            batch_size=64,
                            num_workers=8,
                            collate_fn=collate_fn)

    # Get params for new dataset from old dataset
    columns = {
        k: v for k, v in zip(old_dataset.shards[0].column_names,
                             old_dataset.shards[0].column_encodings)
    }
    compression = old_dataset.shards[0].compression
    hashes = old_dataset.shards[0].hashes
    size_limit = old_dataset.shards[0].size_limit

    # Write deduplicated samples
    found = 0
    removed = 0
    not_found = 0
    no_uid = 0
    data_stats = {
        'source': {},
        'n_samples': 0,
        'mean_length': 0,
        'not_found': 0,
        'no_uid': 0,
        'removed': 0,
    }
    i = 0
    with MDSWriter(out=save_dir,
                   columns=columns,
                   compression=compression,
                   hashes=hashes,
                   size_limit=size_limit) as out:
        for batch in tqdm(dataloader):
            for sample, index in batch:
                if uid_key is not None:
                    try:
                        key = sample[uid_key]
                    except KeyError:
                        logger.error(
                            f'Could not find key {uid_key} in sample {sample}')
                        no_uid += 1
                        break
                else:
                    key = index
                try:
                    if type(similarities) == set:
                        similarity = key in similarities
                        if not similarity:
                            not_found += 1
                    else:
                        similarity = similarities[key]  # type: ignore
                    found += 1
                    # If our similarity is a boolean, we keep the sample if it is True
                    if type(similarity) == bool:
                        keep = similarity
                    else:
                        if type(similarity) in [bytes, str]:
                            similarity = float(similarity)
                        keep = similarity < keep_threshold
                    if keep:
                        out.write(sample)
                        data_stats['n_samples'] += 1
                        sample_len = len(sample['text'])
                        data_stats['mean_length'] = running_average(
                            data_stats['mean_length'], sample_len, i + 1)
                        if source_key in sample.keys():
                            source = sample[source_key]
                            source_stats = data_stats['source'].get(
                                source, {
                                    'n_samples': 0,
                                    'mean_length': 0
                                })
                            source_stats['n_samples'] += 1
                            source_stats['mean_length'] = running_average(
                                source_stats['mean_length'], sample_len, i + 1)
                            data_stats['source'][source] = source_stats
                    else:
                        removed += 1
                except:
                    not_found += 1
                if i % 10000000 == 0:
                    if found != 0:
                        logger.info(
                            f'\nRemoved {removed} of {found} found samples {removed/found:.4f}'
                        )
                    logger.info(
                        f'Not found {not_found} of {i+1} samples {not_found/(i+1):.4f}'
                    )
                i += 1
    if found != 0:
        logger.info(
            f'\nRemoved {removed} of {found} found samples {removed/found:.4f}')
    logger.info(
        f'Not found {not_found} of {i+1} samples {not_found/(len(old_dataset)):.4f}'
    )
    data_stats['not_found'] = not_found
    data_stats['no_uid'] = no_uid
    data_stats['removed'] = removed

    savename = os.path.join(save_dir, 'data_stats.pkl')
    with open(savename, 'wb') as handle:
        pickle.dump(data_stats, handle, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info(f'\nSaved data stats to {savename}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Prune and write dataset as MDS streaming dataset')
    parser.add_argument('--streaming_remote', type=str, default="None")
    parser.add_argument('--streaming_local',
                        type=str,
                        default="/tmp/streaming_dataset")
    parser.add_argument('--save_dir', type=str)
    parser.add_argument('--name', type=str)
    parser.add_argument('--split', default=None)
    parser.add_argument('--similarity_file', type=str)
    parser.add_argument('--quantiles_file', type=str)
    parser.add_argument('--keep_fraction', type=float)
    parser.add_argument('--source_key',
                        type=str,
                        default='pile_set_name',
                        help='Key indicating data source of sample')
    parser.add_argument(
        '--uid_key',
        type=str,
        default='',
        help=
        'Key indicating unique id of sample. If nothing provided, will use sample index.'
    )
    args = parser.parse_args()

    # Convert split to None if empty
    if args.split is not None:
        args.save_dir = os.path.join(args.save_dir, args.split)
    # Convert uid_key to None if empty
    if args.uid_key == '':
        args.uid_key = None
    write_and_prune(**vars(args))