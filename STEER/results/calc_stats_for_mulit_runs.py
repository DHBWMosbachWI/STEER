import os
from os.path import join
from dotenv import load_dotenv
load_dotenv(override=True)
import pandas as pd
import json
from sklearn.metrics import classification_report
import numpy as np


if __name__:"__main__":

    labeled_data_size = 1
    distance_threshold = 0.01
    path = join(
        os.environ["WORKING_DIR"], "emb_clus", "without_knn", "out",
        f"public_bi_clustering_n_classify_results_gen_train_data_{distance_threshold}_{labeled_data_size}_absolute_20.0"
    )
    scores = {
        "f1-scores_macro": [],
        "precisions_macro":[],
        "recalls_macro":[],
        "supports_macro": [],
        "f1-scores_weighted": [],
        "precisions_weighted": [],
        "recalls_weighted": [],
        "supports_weighted": []
    }


    for random_state in [1,2,3,4,5]:
        df_current = pd.read_csv(path+f"_{random_state}.csv")
        df_current = df_current[(df_current["already_labeled"] == False) & (df_current["predicted_type"] != "None")]
        current_class_report = classification_report(df_current["semanticType"],df_current["predicted_type"], output_dict=True)
        for metric in ["macro","weighted"]:
            scores[f"f1-scores_{metric}"].append(current_class_report[f"{metric} avg"]["f1-score"])
            scores[f"precisions_{metric}"].append(current_class_report[f"{metric} avg"]["precision"])
            scores[f"recalls_{metric}"].append(current_class_report[f"{metric} avg"]["recall"])
            scores[f"supports_{metric}"].append(current_class_report[f"{metric} avg"]["support"])

    df_scores = pd.DataFrame(
        np.array([
            scores["f1-scores_macro"], scores["precisions_macro"],
            scores["recalls_macro"], scores["supports_macro"],
            scores["f1-scores_weighted"], scores["precisions_weighted"],
            scores["recalls_weighted"], scores["supports_weighted"]
        ]), index=scores.keys())
    df_scores["mean"] = df_scores.mean(axis=1)
    df_scores["std"] = df_scores.std(axis=1)
    df_scores["var"] = df_scores.var(axis=1)

    df_scores.to_csv(path+"_mean.csv")
