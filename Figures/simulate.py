import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

warnings.filterwarnings('ignore')

mid_cnfs = np.array([1 / 6, 1 / 2, 5 / 6])
hu_cnfs_map = {"l": mid_cnfs - 1 / 6, "m": mid_cnfs, "h": mid_cnfs + 1 / 6}

data = Path("data")


def get_acc(grp, ai_bins, hu_cnfs):
    # naive max confidence slating accuracy
    grp["mcs_hu"] = hu_cnfs[grp["cnf_bin_hu"] - 1]  # 1, 2, 3
    grp["mcs_cor"] = np.where(grp["mcs_hu"] >= grp["cnf_ai"], grp["correct_hu"], grp["correct_ai"])

    # random accuracy
    rand_idx = np.random.rand(len(grp)) >= 0.5
    grp["rand_cor"] = np.where(rand_idx, grp["correct_hu"], grp["correct_ai"])

    # calibrate
    cal_hu = grp.groupby("cnf_bin_hu")["correct_hu"].transform("mean")

    grp["cnf_bin_ai"] = pd.cut(grp["cnf_ai"], np.linspace(0, 1, ai_bins + 1))
    cal_ai = grp.groupby("cnf_bin_ai")["correct_ai"].transform("mean")

    # bayes-optimal accuracies (equal to posterior log odds)
    grp["bayes_cor"] = np.where(cal_hu >= cal_ai, grp["correct_hu"], grp["correct_ai"])

    # get covariance between human and AI correctness indicators
    cov_hm = np.cov(grp["correct_hu"], grp["correct_ai"])[0, 1]
    rho_hm = np.corrcoef(grp["correct_hu"], grp["correct_ai"])[0, 1]

    return pd.Series(
        {"mcs_acc": grp["mcs_cor"].mean(), "bayes_acc": grp["bayes_cor"].mean(), "rand_acc": grp["rand_cor"].mean(),
         "ai_acc": grp["correct_ai"].mean(), "hu_acc": grp["correct_hu"].mean(), "cov_hm": cov_hm, "rho_hm": rho_hm,
         "ai_auc": roc_auc_score(grp["correct_ai"], grp["cnf_ai"]),
         "hu_auc": roc_auc_score(grp["correct_hu"], grp["cnf_bin_hu"])})


hu = pd.read_csv(data / "human_only_classification_6per_img_preprocessed.csv")
hu["cnf_bin_hu"] = hu["confidence_int"]  # human confidence is already binned

hu_sel = hu[["correct", "participant_id", "image_name", "noise_level", "cnf_bin_hu"]]

acc_lst = []
for p in tqdm(data.glob("hai*.csv")):
    ai = pd.read_csv(p)

    # create human-ai pairs
    ai["cnf_ai"] = ai.iloc[:, -16:].max(axis=1)

    ai_sel = ai[["correct", "model_name", "image_name", "noise_level", "cnf_ai"]]

    com = pd.merge(hu_sel, ai_sel, on=["image_name", "noise_level"], suffixes=("_hu", "_ai"))

    ep = p.stem.split("_")[1]
    if "epoch" in ep:
        ep = ep[-2:]

    for ai_bins in [3, 5, 10]:
        for hu_cnfs_key in ["l", "m", "h"]:
            hu_cnfs = hu_cnfs_map[hu_cnfs_key]

            print(p, ai_bins, hu_cnfs)

            acc = com.groupby(["participant_id", "model_name"]).apply(
                lambda grp: get_acc(grp, ai_bins, hu_cnfs)).reset_index()

            acc["epoch"] = ep

            acc["ai_bins"] = ai_bins
            acc["hu_cnfs"] = hu_cnfs_key

            acc_lst.append(acc)

df = pd.concat(acc_lst)
df.rename(columns={"noise_level": "noise", "model_name": "model"}, inplace=True)
df.replace({"epoch": {"00": "<1", "01": "1"}, "noise": {-10: "original", 0: "monochrome"}}, inplace=True)
df.to_csv("acc.csv", index=False)
