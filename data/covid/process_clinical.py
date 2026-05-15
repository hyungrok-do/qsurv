import os
from datetime import datetime
import pandas as pd

IMG_ROOT = "./COVID-19-NY-SBU"
df = pd.read_csv("./deidentified_overlap_tcia.csv.cleaned.csv_20210806.csv")

cols = [
    "to_patient_id",
    "last.status",
    "visit_start_datetime",
    "length_of_stay",
]
df = df[cols]

df["visit_start_datetime"] = pd.to_datetime(df["visit_start_datetime"])
df["length_of_stay"] = pd.to_numeric(df["length_of_stay"], errors="coerce")

# last.status → event_indicator (discharged=0, deceased=1)
df["event_indicator"] = df["last.status"].map({
    "discharged": 0,
    "deceased": 1,
})

VALID_STUDY_TYPES = [
    "CHEST AP PORT",
    "CHEST AP VIEWONLY",
    "CHEST AP PORTABLE",
    "CHEST AP VIEW ONLY",
]


def get_closest_diff_days(patient_id: str, visit_dt: pd.Timestamp) -> int:

    patient_dir = os.path.join(IMG_ROOT, str(patient_id))
    if not os.path.isdir(patient_dir):
        return 99999999

    vdate = visit_dt.date()
    date_diffs = []

    for name in os.listdir(patient_dir):

        if not any(study_type in name for study_type in VALID_STUDY_TYPES):
            continue

        parts = name.split("-")
        if len(parts) < 3:
            continue

        date_str = "-".join(parts[:3])  # "01-02-1901"
        try:
            img_date = datetime.strptime(date_str, "%m-%d-%Y").date()
        except ValueError:
            continue

        diff_days = (img_date - vdate).days
        date_diffs.append(diff_days)

    if not date_diffs:
        return 99999999

    closest_diff = min(date_diffs, key=lambda x: abs(x))
    return closest_diff


def calc_observed_time(row):
    diff_days = get_closest_diff_days(row["to_patient_id"], row["visit_start_datetime"])
    return row["length_of_stay"] - diff_days


df["observed_time"] = df.apply(calc_observed_time, axis=1)

df.to_csv("clinical_processed.csv", index=False)
