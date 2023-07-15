import numpy as np
from streaming import StreamingDataset
from transformers import AutoTokenizer


def get_samples(remote, split, tokenizer_name, num_samples=1, shuffle=False):
    ds = StreamingDataset(remote=remote, split=split, shuffle=shuffle)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    samples = []
    for i in range(num_samples):
        sample = ds[i]
        #print(sample["domain_idx"])
        samples.append(
            tokenizer.decode(
                np.frombuffer(sample['tokens'], dtype=np.int64).copy()))
    return samples


def get_sample_int_keys(sample, int_keys):
    int_keys_unpacked = {}
    for int_key in int_keys:
        int_keys_unpacked[int_key] = sample[int_key].item()
        del sample[int_key]
    return {**sample, **int_keys_unpacked}
