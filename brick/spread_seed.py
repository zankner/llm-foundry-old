import argparse
import tempfile

from composer.utils import get_file

from ckpt_utils import RemoteUploaderDownloader


def spread_seed(args):
    remote_uploader = RemoteUploaderDownloader(bucket_uri=args.ckpt_bucket)

    with tempfile.NamedTemporaryFile() as seed_tmp:
        get_file(args.seed_ckpt, seed_tmp.name, overwrite=True)

        for tree_ckpt in args.tree_ckpts:
            remote_uploader.upload_file(tree_ckpt, seed_tmp.name)

    remote_uploader.post_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-ckpt", type=str, required=True)
    parser.add_argument("--tree-ckpts", type=str, nargs="+", required=True)
    args = parser.parse_args()

    spread_seed(args)