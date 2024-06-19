import argparse
import json
import numpy as np
import os
from tabulate import tabulate

from src.utils.constants import ENV_TO_SUITE

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Known tasks metrics calculation")
    parser.add_argument(
        "--logs_folder_path",
        type=str,
        help="Path to the folder which contains one sub-folder per known task.",
    )
    args = parser.parse_args()
    task_folders = os.listdir(args.logs_folder_path)
    assert len(task_folders) == 12

    adroit = []
    dmc = []
    metaworld = []
    for task_folder in task_folders:
        task_res_file = os.path.join(
            args.logs_folder_path, task_folder, "eval_log.json"
        )
        f = open(task_res_file)
        task_res = json.load(f)
        mean_score = task_res["mean_score"]

        if ENV_TO_SUITE[task_folder] == "adroit":
            adroit.append(mean_score)
        elif ENV_TO_SUITE[task_folder] == "dmc":
            dmc.append(mean_score)
        else:
            metaworld.append(mean_score)
    assert len(adroit) == 2
    assert len(dmc) == 5
    assert len(metaworld) == 5
    adroit_avg = np.mean(adroit).item()
    dmc_avg = np.mean(dmc).item()
    metaworld_avg = np.mean(metaworld).item()
    benchmarks_avg = np.mean([adroit_avg, dmc_avg, metaworld_avg]).item()
    tasks_avg = np.mean(adroit + dmc + metaworld).item()

    metrics = {}
    metrics["Adroit"] = adroit_avg
    metrics["DMC"] = dmc_avg
    metrics["MetaWorld"] = metaworld_avg
    metrics["Benchmarks avg"] = benchmarks_avg
    metrics["Tasks avg"] = tasks_avg

    print(tabulate(metrics.items()))
    with open(os.path.join(args.logs_folder_path, "metrics_all_tasks.json"), "w") as fp:
        json.dump(metrics, fp)
