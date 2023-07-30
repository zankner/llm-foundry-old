import argparse
import os
import tempfile

import torch
import pathlib
import shutil
import uuid
from composer.utils import get_file
from composer.loggers import RemoteUploaderDownloader as ComposerRemoteUploaderDownloader


class RemoteUploaderDownloader(ComposerRemoteUploaderDownloader):

    def upload_file(
        self,
        remote_file_name: str,
        file_path: pathlib.Path,
        *,
        overwrite: bool,
    ):
        copied_path = os.path.join(self._upload_staging_folder,
                                   str(uuid.uuid4()))
        os.makedirs(self._upload_staging_folder, exist_ok=True)
        shutil.copy2(file_path, copied_path)
        formatted_remote_file_name = self._remote_file_name(remote_file_name)
        with self._object_lock:
            if formatted_remote_file_name in self._logged_objects and not overwrite:
                raise FileExistsError(
                    f'Object {formatted_remote_file_name} was already enqueued to be uploaded, but overwrite=False.'
                )
            self._logged_objects[formatted_remote_file_name] = (copied_path,
                                                                overwrite)


def load_state_dict(model_ckpt: str):
    with tempfile.NamedTemporaryFile() as tmp:
        get_file(model_ckpt, tmp.name, overwrite=True)
        state_dict = torch.load(tmp.name)
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