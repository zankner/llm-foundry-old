import argparse
import os
import tempfile

import torch
from composer.utils import get_file

from ckpt_utils import RemoteUploaderDownloader


def load_state_dict(model_ckpt: str):
    with tempfile.NamedTemporaryFile() as tmp:
        get_file(model_ckpt, tmp.name, overwrite=True)
        state_dict = torch.load(tmp.name, map_location="cpu")
    return state_dict


def update_and_upload_states(average_model_weights: str, upload_bucket: str,
                             model_ckpts: str, model_states: str):
    remote_uploader = RemoteUploaderDownloader(bucket_uri=upload_bucket)
    for model_ckpt, model_state in zip(model_ckpts, model_states):
        model_state["state"]["model"] = average_model_weights
        with tempfile.NamedTemporaryFile() as tmp:
            torch.save(model_state, tmp.name)
            remote_uploader.upload_file(model_ckpt, tmp.name)
    remote_uploader.post_close()


def average_models(model_states, merge_weights):
    model_weights = [
        state_dict["state"]["model"] for state_dict in model_states
    ]
    with torch.no_grad():
        merged = {}
        for key in model_weights[0]:
            merged[key] = torch.sum(torch.stack([
                sd[key] * weight
                for sd, weight in zip(model_weights, merge_weights)
            ]),
                                    axis=0)
    return merged


def merge_models(args):

    model_ckpts = [
        os.path.join(args.model_bucket, model_ckpt)
        for model_ckpt in args.model_ckpts
    ]
    if args.merge_weights is None:
        merge_weights = [1 / len(model_ckpts)] * len(model_ckpts)
    else:
        assert len(model_ckpts) == len(args.model_weights)

    model_states = [load_state_dict(model_ckpt) for model_ckpt in model_ckpts]
    average_model_weights = average_models(model_states, merge_weights)

    update_and_upload_states(average_model_weights, args.model_bucket,
                             args.model_ckpts, model_states)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-bucket", type=str, required=True)
    parser.add_argument("--model-ckpts", type=str, nargs="+",
                        required=True)  # Shouldn't include bucket prefix
    parser.add_argument("--merge-weights", type=float, nargs="+", default=None)
    args = parser.parse_args()

    merge_models(args)