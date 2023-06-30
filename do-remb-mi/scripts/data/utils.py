def get_sample_int_keys(sample, int_keys):
    int_keys_unpacked = {}
    for int_key in int_keys:
        int_keys_unpacked[int_key] = sample[int_key].item()
        del sample[int_key]
    return {**sample, **int_keys_unpacked}