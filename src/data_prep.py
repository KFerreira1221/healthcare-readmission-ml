import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def to_utc_naive(s: pd.Series) -> pd.Series:
    """
    Convert a datetime-like Series to timezone-naive UTC.
    Handles tz-aware strings like '...+00:00' and tz-naive timestamps.
    """
    dt = pd.to_datetime(s, errors="coerce", utc=True)
    return dt.dt.tz_convert(None)


def build_readmission_label(encounters: pd.DataFrame, days: int = 30) -> pd.DataFrame:
    """
    Create a readmission label:
    readmitted_30d = 1 if patient has another encounter within X days of the end date.
    """
    encounters = encounters.copy()

    # Normalize datetimes to tz-naive UTC (prevents tz-aware vs tz-naive comparison errors)
    if "START" not in encounters.columns or "STOP" not in encounters.columns:
        raise KeyError("encounters.csv must contain START and STOP columns.")
    encounters["START"] = to_utc_naive(encounters["START"])
    encounters["STOP"] = to_utc_naive(encounters["STOP"])

    encounters = encounters.sort_values(["PATIENT", "START"])

    # Next encounter start for same patient
    encounters["NEXT_START"] = encounters.groupby("PATIENT")["START"].shift(-1)

    # Days until next visit
    encounters["DAYS_TO_NEXT"] = (encounters["NEXT_START"] - encounters["STOP"]).dt.days

    encounters["readmitted_30d"] = (
        encounters["DAYS_TO_NEXT"].notna()
        & (encounters["DAYS_TO_NEXT"] >= 0)
        & (encounters["DAYS_TO_NEXT"] <= days)
    ).astype(int)

    return encounters


def _safe_datetime(df: pd.DataFrame, col: str) -> pd.Series:
    """
    Safe datetime conversion that always returns tz-naive UTC.
    """
    if col in df.columns:
        return to_utc_naive(df[col])
    return pd.to_datetime(pd.Series([pd.NaT] * len(df)), errors="coerce")


def add_history_features(
    enc: pd.DataFrame,
    conditions: pd.DataFrame | None,
    meds: pd.DataFrame | None,
    lookback_days: int = 365,
) -> pd.DataFrame:
    """
    For each encounter, count conditions/meds for the same patient occurring in the lookback window
    before encounter START date.
    """
    df = enc.copy()

    # Ensure encounter START is normalized
    df["START"] = to_utc_naive(df["START"])
    df["enc_id"] = range(len(df))  # stable join key

    # Default zeros
    df["conditions_365d"] = 0
    df["unique_conditions_365d"] = 0
    df["meds_365d"] = 0
    df["unique_meds_365d"] = 0

    window = pd.Timedelta(days=lookback_days)

    # ---- Conditions ----
    if conditions is not None and len(conditions) > 0:
        c = conditions.copy()

        # Synthea often has PATIENT + START
        c["START"] = _safe_datetime(c, "START")

        # Use DESCRIPTION as "type" if present
        c["COND_NAME"] = c["DESCRIPTION"] if "DESCRIPTION" in c.columns else "COND"

        # Join on PATIENT, then filter by time window
        merged = df[["enc_id", "PATIENT", "START"]].merge(
            c[["PATIENT", "START", "COND_NAME"]].rename(columns={"START": "C_START"}),
            on="PATIENT",
            how="left",
        )

        in_window = merged[
            (merged["C_START"].notna())
            & (merged["C_START"] <= merged["START"])
            & (merged["C_START"] >= merged["START"] - window)
        ]

        cond_counts = in_window.groupby("enc_id").size().rename("conditions_365d_new")
        uniq_cond = (
            in_window.groupby("enc_id")["COND_NAME"].nunique().rename("unique_conditions_365d_new")
        )

        df = df.join(cond_counts, on="enc_id")
        df["conditions_365d"] = df["conditions_365d_new"].fillna(0).astype(int)
        df = df.drop(columns=["conditions_365d_new"])

        df = df.join(uniq_cond, on="enc_id")
        df["unique_conditions_365d"] = df["unique_conditions_365d_new"].fillna(0).astype(int)
        df = df.drop(columns=["unique_conditions_365d_new"])

    # ---- Medications ----
    if meds is not None and len(meds) > 0:
        m = meds.copy()

        m["START"] = _safe_datetime(m, "START")
        m["MED_NAME"] = m["DESCRIPTION"] if "DESCRIPTION" in m.columns else "MED"

        merged = df[["enc_id", "PATIENT", "START"]].merge(
            m[["PATIENT", "START", "MED_NAME"]].rename(columns={"START": "M_START"}),
            on="PATIENT",
            how="left",
        )

        in_window = merged[
            (merged["M_START"].notna())
            & (merged["M_START"] <= merged["START"])
            & (merged["M_START"] >= merged["START"] - window)
        ]

        med_counts = in_window.groupby("enc_id").size().rename("meds_365d_new")
        uniq_med = in_window.groupby("enc_id")["MED_NAME"].nunique().rename("unique_meds_365d_new")

        df = df.join(med_counts, on="enc_id")
        df["meds_365d"] = df["meds_365d_new"].fillna(0).astype(int)
        df = df.drop(columns=["meds_365d_new"])

        df = df.join(uniq_med, on="enc_id")
        df["unique_meds_365d"] = df["unique_meds_365d_new"].fillna(0).astype(int)
        df = df.drop(columns=["unique_meds_365d_new"])

    return df


def basic_feature_table(encounters: pd.DataFrame) -> pd.DataFrame:
    """
    Create final feature table.
    """
    df = encounters.copy()

    # Encounter length
    df["encounter_length_hours"] = (df["STOP"] - df["START"]).dt.total_seconds() / 3600.0

    # Encounter class/category
    if "ENCOUNTERCLASS" in df.columns:
        df["encounter_class"] = df["ENCOUNTERCLASS"].fillna("UNKNOWN")
    elif "TYPE" in df.columns:
        df["encounter_class"] = df["TYPE"].fillna("UNKNOWN")
    else:
        df["encounter_class"] = "UNKNOWN"

    keep = [
        "PATIENT",
        "encounter_length_hours",
        "encounter_class",
        "conditions_365d",
        "unique_conditions_365d",
        "meds_365d",
        "unique_meds_365d",
        "readmitted_30d",
    ]

    # Ensure columns exist even if optional files missing
    for col in keep:
        if col not in df.columns:
            df[col] = 0

    out = df[keep].dropna(subset=["readmitted_30d"]).copy()

    # Replace any inf/nan numeric issues
    out["encounter_length_hours"] = out["encounter_length_hours"].replace([pd.NA, float("inf"), -float("inf")], 0)
    out["encounter_length_hours"] = out["encounter_length_hours"].fillna(0)

    return out


def main():
    encounters_path = RAW_DIR / "encounters.csv"
    if not encounters_path.exists():
        raise FileNotFoundError(f"Missing {encounters_path}. Put Synthea encounters.csv in data/raw/")

    encounters = pd.read_csv(encounters_path)
    labeled = build_readmission_label(encounters, days=30)

    # Optional files
    cond_path = RAW_DIR / "conditions.csv"
    meds_path = RAW_DIR / "medications.csv"

    conditions = pd.read_csv(cond_path) if cond_path.exists() else None
    meds = pd.read_csv(meds_path) if meds_path.exists() else None

    enriched = add_history_features(labeled, conditions, meds, lookback_days=365)
    features = basic_feature_table(enriched)

    out_path = OUT_DIR / "readmission_dataset.csv"
    features.to_csv(out_path, index=False)
    print(f"Saved processed dataset to: {out_path} (rows={len(features)})")


if __name__ == "__main__":
    main()