import re

import wandb
from omegaconf import OmegaConf as om

api = wandb.Api()
project_name = "mosaic-ml/amortized-selection"


def compute_averages(run_data, logger_keys):

    results = {}
    pat = re.compile('metrics/(.*?)/(\d+)-shot(/.*?)?/InContextLearning(.*)')
    for key in logger_keys:
        match = pat.match(key)
        val = run_data[key]

        if match:
            eval_name = match.group(1)
            num_shot = match.group(2)
            subcat = match.group(3)
            metric = match.group(4)

            if subcat is not None:
                subcat = subcat[1:]
                if f'metrics/{eval_name}/{num_shot}-shot/InContextLearning{metric}' not in results:
                    results[f'metrics/{eval_name}/{num_shot}-shot/InContextLearning{metric}'] = []
                results[
                    f'metrics/{eval_name}/{num_shot}-shot/InContextLearning{metric}'].append(
                        val)
            else:
                results[key] = [val]
    return {k: sum(v) / len(v) for k, v in results.items()}


if __name__ == "__main__":
    with open("scripts/eval/yamls/model_gauntlet.yaml", "r") as f:
        model_gauntlet = om.load(f)
    categories = model_gauntlet.model_gauntlet.categories
    for category in categories:
        for benchmark in category['benchmarks']:
            benchmark['weighting'] = 1

    filters = {"$and": [{"tags": "eval"}, {"tags": {"$ne": "prelim"}}]}
    runs = api.runs(project_name, filters=filters)
    for run in runs:
        group = run.group
        name = run.name
        history = run.history(pandas=False)
        logger_keys = [
            k for k in history[0].keys()
            if "metrics" in k and "gauntlet" not in k
        ]
        to_log = {}
        for eval_step in history:
            step = eval_step["_step"]
            new_metrics = compute_averages(eval_step, logger_keys)

            composite_scores = {}
            for category in categories:
                composite_scores[category['name']] = []
                for benchmark in category['benchmarks']:
                    key_pat = re.compile(
                        f"metrics/{benchmark['name']}/{benchmark['num_fewshot']}-shot/.*Accuracy"
                    )

                    matching_key = [
                        k for k in new_metrics.keys()
                        if key_pat.match(k) is not None
                    ]
                    if len(matching_key) == 0:
                        print(
                            f"Warning: couldn't find results for benchmark: {benchmark}"
                        )
                    else:
                        score = new_metrics[matching_key[0]]
                        score -= benchmark['random_baseline']
                        score /= 1.0 - benchmark['random_baseline']

                        composite_scores[category['name']].append({
                            'name': benchmark['name'],
                            'score': score,
                            'weighting': benchmark['weighting']
                        })

                total_weight = sum(
                    k['weighting'] for k in composite_scores[category['name']])
                composite_scores[category['name']] = sum(
                    k['score'] * (k['weighting'] / total_weight)
                    for k in composite_scores[category['name']])

            composite_scores = {
                f'metrics/normalized_model_gauntlet/{k}': v
                for k, v in composite_scores.items()
            }

            composite_scores[
                'metrics/normalized_model_gauntlet/category-average'] = sum(
                    composite_scores.values()) / len(composite_scores.values())

            to_log[step] = composite_scores

        new_run = wandb.init(project="amortized-selection",
                             name=f"normalized-{name}",
                             group=group,
                             tags=run.tags)

        for step, metrics in to_log.items():
            wandb.log(metrics, step=step)
        new_run.finish()