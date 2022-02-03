from typing import Tuple

import os
import sys
import argparse

import numpy as np
import pandas as pd

CLASSIFICATION_METRICS = ["acc", "auc", "balacc", "logloss"]
REGRESSION_METRICS = ["mae", "r2", "rmse"]
METRICS = CLASSIFICATION_METRICS + REGRESSION_METRICS


def _get_mean_results_across_folds(df) -> pd.DataFrame:
    """Returns a dataframe with the task, id, metric and the mean values
    across folds

    [idx: 'task', 'id', 'metric', ... mean metrics across folds ...]
    """
    # Get the information about id and metric, only need info from first fold

    # [idx: task, id, metric]
    df_info = df[df["fold"] == 0][["task", "id", "metric"]].set_index("task")

    # [idx: task, ... mean metrics across folds ...]
    available_metrics = list(set(METRICS).intersection(set(df.columns)))
    df_means = df[["task"] + available_metrics].groupby(["task"]).mean()

    return df_info.join(df_means)


def generate_framework_def(
    user_dir: str,
    username: str,
    branch: str,
    commit: str,  # Not used in this setup but perhaps in a different one
):
    """Creates a framework definition to run an autosklearn repo.

    Technically we only use the commit to pull the targeted version but for
    naming consistency, we need to know the branch too.

    Parameters
    ----------
    user_dir: str
        The path to where the framework definition should be placed

    username: str
        The username of the auto-sklearn fork owner to pull from, used as

        * ssh: git@github.com:{username}/auto-sklearn

    branch: str
        Which branch of the fork to pull from
        e.g.
            development

    commit: str
        A full commit hash to use
        e.g.
            8b474a437ce980bd0909db59141b40d56f6d5688
            or
            #8b474a437ce980bd0909db59141b40d56f6d5688
    """
    assert (
        len(commit) == 41 and commit[0] == "#" or len(commit) == 40
    ), "Not a commit hash"

    # automlbenchmark requires the '#' to identify it's a commit rather than
    # a branch being targeted
    if commit[0] != "#":
        commit = "#" + commit

    # Tried commit and ssh repo but was getting errors with ssh
    # Tried commit and https but getting issues with commit ref
    # Using branch and https
    version = branch
    repo = f"https://github.com/{username}/auto-sklearn.git"

    # Create the framework file
    lines = "\n".join(
        [
            f"---",
            f"autosklearn_targeted:",
            f"  extends: autosklearn",
            f"  version: '{version}'",
            f"  repo: '{repo}'",
        ]
    )

    filepath = os.path.join(user_dir, "frameworks.yaml")
    with open(filepath, "w") as f:
        f.writelines(lines)


def create_comparison(
    baseline_csv_classification: str,
    baseline_csv_regression: str,
    targeted_csv_classification: str,
    targeted_csv_regression: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Creates a csv with comparisons between the baseline and results.

    Scores are averaged across folds for a given task.

    The baseline and results should have the following fields as given by
    automl benchmark:

    * ['id', 'task', 'framework', 'constraint', 'fold', 'type', 'result',
     'metric', 'mode', 'version', 'params', 'app_version', 'utc', 'duration',
     'training_duration', 'predict_duration', 'models_count', 'seed', 'info',
     'acc', 'auc', 'balacc', 'logloss', 'mae', 'r2', 'rmse']

    Parameters
    ----------
    baseline_csv_classification: str
        Path to the csv containing the baseline classification results

    baseline_csv_regression: str
        Path to the csv containing the baseline regression results

    targeted_csv_classification: str
        Path to the csv containing the targeted classification results

    targeted_csv_regression: str
        Path to the csv containing the targeted regression results

    Returns
    -------
    Tuple[df.DataFrame, df.DataFrame, df.DataFrame]
        Returns the mean results across folds for:

            baseline, targeted, comparisons

        Comparisons here is the difference between (targeted - baseline)
        Returns them in that specific order
    """

    # Load in data and get the means across folds
    df_baseline_classification = pd.read_csv(baseline_csv_classification)
    df_baseline_regression = pd.read_csv(baseline_csv_regression)
    df_baseline = pd.concat([df_baseline_classification, df_baseline_regression])
    df_baseline_means = _get_mean_results_across_folds(df_baseline)

    df_targeted_classification = pd.read_csv(targeted_csv_classification)
    df_targeted_regression = pd.read_csv(targeted_csv_regression)
    df_targeted = pd.concat([df_targeted_classification, df_targeted_regression])
    df_targeted_means = _get_mean_results_across_folds(df_targeted)

    # Find the set intersection of tasks they have in common
    common_tasks = set(df_baseline_means.index).intersection(
        set(df_targeted_means.index)
    )

    # Find the set of metrics that are comparable
    baseline_metrics = set(METRICS).intersection(set(df_baseline_means.columns))
    common_metrics = baseline_metrics.intersection(set(df_targeted_means.columns))

    # Calculate the differences for in common tasks, across all available metrics
    df_differences = (
        df_targeted_means.loc[common_tasks][common_metrics]
        - df_baseline_means.loc[common_tasks][common_metrics]
    )

    # Get the metric used for training and the dataset id of common tasks
    df_info = df_baseline_means.loc[common_tasks][["id", "metric"]]

    # Join together the info and the differences
    return df_baseline_means, df_targeted_means, df_info.join(df_differences)


def create_comparisons_markdown(
    baseline_means_csv: str,
    targeted_means_csv: str,
    compared_means_csv: str,
) -> str:
    """Creates markdown that can be posted to Github that shows
    a comparison between the baseline and the targeted branch.

    Parameters
    ----------
    baseline_means_csv: str
        path to the csv baseline results with their mean across folds.

    targeted_means_csv: str
        path to the csv with the results of the targeted branch with their mean
        across folds.

    compared_means_csv: str
        path to the csv containing the comparisons results

    Returns
    -------
    str
        The results written in markdown.
    """
    # Create colours and func to create the markdown for it
    colours = {
        "Worse": ["353536", "800000", "bd0000", "ff0000"],
        "Better": ["353536", "306300", "51a800", "6fe600"],
        "Good": "6fe600",
        "Bad": "ff0000",
        "Neutral": "353536",
        "NaN": "52544f",
    }

    def colour(kind, scale=None):
        c = colours[kind] if scale is None else colours[kind][scale]
        return f"![#{c}](https://via.placeholder.com/15/{c}/000000?text=+)"

    # Metrics, whether positive is better and the tolerances between each
    # Neutral, kind of good/bad, very good/bad etc...
    metric_tolerances = {
        "acc": {"positive_is_better": True, "tol": [0.001, 0.01, 0.2]},
        "auc": {"positive_is_better": True, "tol": [0.001, 0.01, 0.2]},
        "balacc": {"positive_is_better": True, "tol": [0.001, 0.01, 0.2]},
        "logloss": {"positive_is_better": False, "tol": [0.009, 0.01, 0.2]},
        "mae": {"positive_is_better": False, "tol": [0.001, 0.01, 0.2]},
        "r2": {"positive_is_better": True, "tol": [0.001, 0.01, 0.2]},
        "rmse": {"positive_is_better": False, "tol": [0.001, 0.01, 0.2]},
    }

    def is_good(score, metric):
        return (
            score > 0
            and metric_tolerances[metric]["positive_is_better"]
            or score < 0
            and not metric_tolerances[metric]["positive_is_better"]
        )

    def is_neutral(diff, baseline_score, metric):
        tolerance = metric_tolerances[metric]["tol"][0]
        if baseline_score == 0:
            baseline_score = 1e-10
        prc_diff = diff / baseline_score
        return prc_diff <= tolerance

    def tolerance_colour(baseline_value, comparison_value, metric):
        if np.isnan(baseline_value) or np.isnan(comparison_value):
            return colour("NaN")

        if baseline_value == 0:
            baseline_value = 1e-10

        prc_diff = comparison_value / baseline_value

        tolerances = metric_tolerances[metric]["tol"]
        if metric_tolerances[metric]["positive_is_better"]:
            diff_color = "Better" if prc_diff > 0 else "Worse"
        else:
            diff_color = "Better" if prc_diff < 0 else "Worse"

        if abs(prc_diff) < tolerances[0]:
            return colour(diff_color, 0)
        elif abs(prc_diff) < tolerances[1]:
            return colour(diff_color, 1)
        elif abs(prc_diff) < tolerances[2]:
            return colour(diff_color, 2)
        else:
            return colour(diff_color, 3)

    legend = {
        "B": "Baseline",
        "T": "Target Version",
        "**Bold**": "Training Metric",
        "/": "Missing Value",
        "---": "Missing Task",
    }
    legend.update(
        {
            key: colour(key)
            for key in set(colours.keys()) - set(["Worse", "Better", "Good", "Bad"])
        }
    )
    # Worse and better are handled seperatly

    compared = pd.read_csv(compared_means_csv, index_col="task")
    baseline = pd.read_csv(baseline_means_csv, index_col="task")
    targeted = pd.read_csv(targeted_means_csv, index_col="task")

    # Some things to keep track of for the textual summary
    n_performed_equally = 0
    n_performed_better = 0
    n_performed_worse = 0
    n_could_not_compare = 0

    headers = ["task", "metric"] + METRICS
    table_header = "|".join(headers)
    seperator = "|".join(len(headers) * ["---"])

    lines = [table_header, seperator]

    for task in baseline.index:

        # The chosen metric name and the csv column differ with neg_logloss and
        # logloss
        training_metric = baseline.loc[task]["metric"]
        if training_metric == "neg_logloss":
            training_metric = "logloss"

        # The baseline has tasks we can't compare with
        if task not in compared.index:
            line = "|".join([task, training_metric] + len(METRICS) * ["---"])
            lines.append(line)

        # We can compare for a given tasks
        else:

            # Create the entries for the line with each metric
            entries = []
            for metric in METRICS:

                entry = ""

                # If the metric does not exist for either, do fill it in as
                # missing
                if metric not in baseline.columns and metric not in compared.columns:
                    n_could_not_compare += 1
                    entry = "/"

                # If the metric exists in the baseline but not in the comparison
                elif metric in baseline.columns and not metric in compared.columns:
                    n_could_not_compare += 1
                    entry = "<br/>".join(
                        [f" B : {baseline.loc[task][metric]:.3f}", f" T : /"]
                    )

                # If the metric exists in the comparison but not in the baseline
                elif metric in compared.columns and not metric in baseline.columns:
                    n_could_not_compare += 1
                    entry = "<br/>".join(
                        [f" B : /", f" T : {targeted.loc[task][metric]:.3f}"]
                    )

                # The metric must exist in both
                else:
                    baseline_score = baseline.loc[task][metric]
                    compared_score = compared.loc[task][metric]
                    if is_neutral(compared_score, baseline_score, metric):
                        n_performed_equally += 1
                    elif is_good(compared_score, metric):
                        n_performed_better += 1
                    else:
                        n_performed_worse += 1

                    diff_colour = tolerance_colour(
                        baseline_score, compared_score, metric
                    )
                    entry = "<br/>".join(
                        [
                            f" B : {baseline.loc[task][metric]:.3f}",
                            f" T : {targeted.loc[task][metric]:.3f}",
                            f"{diff_colour}: {compared.loc[task][metric]:.3f}",
                        ]
                    )

                # Make the training metric entry bold
                if metric == training_metric:
                    entry = "<b>" + entry + "</b>"

                entries.append(entry)

            # Create the line
            line = "|".join([task, training_metric] + entries)
            lines.append(line)

    # Create the legend line
    score_scale = {
        "worse": "".join(
            colour("Worse", scale) for scale in range(len(colours["Worse"]) - 1, 0, -1)
        ),
        "better": "".join(
            colour("Better", scale) for scale in range(len(colours["Better"]))
        ),
    }
    score_scale = f'worse {score_scale["worse"] + score_scale["better"]} better'

    legend_str = "\t\t\t||\t\t".join(
        [score_scale] + [f"{key} - {text}" for key, text in legend.items()]
    )

    lines.append("")
    lines.append(legend_str)

    # Create a textual summary to go at the top
    compared_metrics = list(set(METRICS).intersection(compared.columns))
    compared_tasks = list(compared.index)
    non_compared_tasks = list(set(baseline.index) - set(compared_tasks))

    # Populate info about each metric
    per_metric_info = {}
    for metric in compared_metrics:
        n_compared = sum(compared[metric].notna())
        compared_average = compared[metric].sum() / n_compared
        baseline_average = baseline[metric].sum() / sum(baseline[metric].notna())

        item_colour = ""
        if is_neutral(compared_average, baseline_average, metric):
            item_colour = colour("Neutral")
        elif is_good(compared_average, metric):
            item_colour = colour("Good")
        else:
            item_colour = colour("Bad")

        per_metric_info[metric] = {
            "average": compared_average,
            "n_compared": n_compared,
            "colour": item_colour,
        }

    summary = "\n".join(
        [
            f"# Results",
            f"Overall the targeted versions performance across {len(compared_tasks)} task(s) and {len(compared_metrics)} metric(s)",
            f"",
            f"*  Equally on <b>{n_performed_equally}</b> comparisons",
            f"*  Better on <b>{n_performed_better}</b> comparisons",
            f"*  Worse on <b>{n_performed_worse}</b> comparisons",
            f"",
            f"There were <b>{len(non_compared_tasks)}</b> task(s) that could not be compared.",
            f"",
            f"The average change for each metric is:" f"",
        ]
        + [
            f"* <b>{metric}: </b> {info['colour']} {info['average']:.4f} across {info['n_compared']} task(s)"
            for metric, info in per_metric_info.items()
        ]
    )
    return "\n".join([summary] + [""] + lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Generates a framework definition for automlbenchmark so that we can target
    # auto-sklearn versions that are not our own
    parser.add_argument("--generate-framework-def", action="store_true")
    parser.add_argument("--user-dir", type=str)
    parser.add_argument("--owner", type=str)
    parser.add_argument("--branch", type=str)
    parser.add_argument("--commit", type=str)

    # For comparing results generated by automlbenchmark for:
    #  -> baseline results generated [--baseline-csv]
    #  -> targeted results generated [--target-csv]
    # by automlbenchmark and the target branch 'results' generated
    parser.add_argument("--compare-results", action="store_true")
    parser.add_argument("--baseline-csv-classification", type=str)
    parser.add_argument("--baseline-csv-regression", type=str)
    parser.add_argument("--targeted-csv-classification", type=str)
    parser.add_argument("--targeted-csv-regression", type=str)
    parser.add_argument("--baseline-means-to", type=str)
    parser.add_argument("--targeted-means-to", type=str)
    parser.add_argument("--compared-means-to", type=str)

    # For generating markdown that can be posted to github that shows the results
    parser.add_argument("--generate-markdown", action="store_true")
    parser.add_argument("--compared-means-csv", type=str)
    parser.add_argument("--baseline-means-csv", type=str)
    parser.add_argument("--targeted-means-csv", type=str)

    args = parser.parse_args()

    if args.generate_framework_def:

        assert args.owner and args.branch and args.commit and args.user_dir

        generate_framework_def(args.user_dir, args.owner, args.branch, args.commit)

    elif args.compare_results:

        assert all(
            [
                args.baseline_csv_classification,
                args.baseline_csv_regression,
                args.targeted_csv_classification,
                args.baseline_csv_regression,
                args.baseline_means_to,
                args.targeted_means_to,
                args.compared_means_to,
            ]
        )

        baseline_means, targeted_means, compared_means = create_comparison(
            baseline_csv_classification=args.baseline_csv_classification,
            baseline_csv_regression=args.baseline_csv_regression,
            targeted_csv_classification=args.targeted_csv_classification,
            targeted_csv_regression=args.targeted_csv_regression,
        )

        for df, path in [
            (baseline_means, args.baseline_means_to),
            (targeted_means, args.targeted_means_to),
            (compared_means, args.compared_means_to),
        ]:
            df.to_csv(path)

    elif args.generate_markdown:
        assert all(
            [args.baseline_means_csv, args.targeted_means_csv, args.compared_means_csv]
        )

        comparisons_table = create_comparisons_markdown(
            baseline_means_csv=args.baseline_means_csv,
            targeted_means_csv=args.targeted_means_csv,
            compared_means_csv=args.compared_means_csv,
        )
        print(comparisons_table)

    else:
        parser.print_help()
        sys.exit(1)
