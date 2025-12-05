import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

warnings.filterwarnings('ignore')

ai_bins = 8

data = Path("data")


def get_acc(grp):
    # calibrate
    cal_hu = grp.groupby("cnf_bin_hu")["correct_hu"].transform("mean")
    cal_ai = grp.groupby("cnf_bin_ai")["correct_ai"].transform("mean")

    # compare accuracies (equal to posterior log odds)
    grp["correct"] = np.where(cal_hu > cal_ai, grp["correct_hu"], grp["correct_ai"])

    return pd.Series(
        {"sim_acc": grp["correct"].mean(), "ai_acc": grp["correct_ai"].mean(), "hu_acc": grp["correct_hu"].mean(),
         "ai_auc": roc_auc_score(grp["correct_ai"], grp["cnf_ai"]),
         "hu_auc": roc_auc_score(grp["correct_hu"], grp["cnf_bin_hu"])})


hu = pd.read_csv(data / "human_only_classification_6per_img_preprocessed.csv")
hu["cnf_bin_hu"] = hu["confidence_int"]  # human confidence is already binned

hu_sel = hu[["correct", "participant_id", "image_name", "noise_level", "cnf_bin_hu"]]

acc_lst = []
for p in tqdm(data.glob("hai*.csv")):
    print(p)
    ai = pd.read_csv(p)

    # create human-ai pairs
    ai["cnf_ai"] = ai.iloc[:, -16:].max(axis=1)
    ai["cnf_bin_ai"] = pd.cut(ai["cnf_ai"], np.linspace(0, 1, ai_bins + 1))

    ai_sel = ai[["correct", "model_name", "image_name", "noise_level", "cnf_bin_ai", "cnf_ai"]]

    com = pd.merge(hu_sel, ai_sel, on=["image_name", "noise_level"], suffixes=("_hu", "_ai"))

    acc = com.groupby(["participant_id", "model_name"]).apply(get_acc).reset_index()

    ep = p.stem.split("_")[1]
    if "epoch" in ep:
        ep = ep[-2:]

    acc["epoch"] = ep
    acc_lst.append(acc)

df = pd.concat(acc_lst)
df.rename(columns={"noise_level": "noise", "model_name": "model"}, inplace=True)
df.replace({"epoch": {"00": "<1", "01": "1"}, "noise": {-10: "original", 0: "monochrome"}}, inplace=True)
df.to_csv("acc.csv", index=False)
