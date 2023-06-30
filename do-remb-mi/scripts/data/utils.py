
def get_sample_with_uid(sample):
    uid = sample["uid"].item()
    del sample["uid"]
    return {**sample, "uid": uid}