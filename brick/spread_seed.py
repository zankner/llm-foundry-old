import os
import argparse
import tempfile

from composer.utils import get_file
from composer.utils.object_store.oci_object_store import OCIObjectStore


def spread_seed(args):
    remote_uploader = OCIObjectStore(bucket=args.ckpt_bucket)

    seed_ckpt = os.path.join(f"oci://{args.ckpt_bucket}", args.seed_ckpt)

    with tempfile.NamedTemporaryFile() as seed_tmp:
        get_file(seed_ckpt, seed_tmp.name, overwrite=True)
        print(f"Downloaded from seed ckpt: {seed_ckpt}")
        for tree_ckpt in args.tree_ckpts:
            remote_uploader.upload_object(
                args.seed_ckpt,
                seed_tmp.name,
            )
            print(f"Uploaded to tree ckpt {tree_ckpt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-bucket", type=str, required=True)
    parser.add_argument("--seed-ckpt", type=str, required=True)
    parser.add_argument("--tree-ckpts", type=str, nargs="+", required=True)
    parser.add_argument("--tree-run-names", type=str, nargs="+", required=True)
    args = parser.parse_args()

    spread_seed(args)