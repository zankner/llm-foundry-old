import argparse
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-file", type=str, required=True)
    args = parser.parse_args()

    with open(args.index_file, "r") as f:
        index = json.load(f)

    for shard_idx in range(len(index["shards"])):
        index["shards"][shard_idx]["compression"] = None
        index["shards"][shard_idx]["zip_data"] = None

    with open(args.index_file, "w") as out:
        json.dump(index, out, sort_keys=True)