"""Feature engineering for all three credit risk datasets.

FeatureEngineer exposes fit_transform() (call on training data only)
and transform() (apply identical transforms to test/stream windows)
to prevent data leakage across time windows.
"""

import random
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

np.random.seed(42)
random.seed(42)

CONFIGS_DIR = Path(__file__).resolve().parents[3] / "experiments" / "configs"
CONFIGS_DIR.mkdir(parents=True, exist_ok=True)

# Reference date for IEEE-CIS TransactionDT
_IEEE_REFERENCE_DATE = datetime(2017, 11, 30)

# Aggregation columns for IEEE-CIS
_AGG_COLS = ["card1", "card2", "card3", "card4", "card5", "card6", "P_emaildomain"]

# Categoricals to target-encode for IEEE-CIS
_TARGET_ENC_COLS = [
    "ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain",
    "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9", "DeviceType",
]

# Free email providers
_FREE_EMAIL_DOMAINS = {"gmail.com", "yahoo.com", "hotmail.com", "outlook.com"}

# V-feature interaction pairs for CC Fraud
_V_INTERACTIONS = [("V1", "V2"), ("V1", "V3"), ("V2", "V3"), ("V1", "V4"), ("V2", "V4")]


class FeatureEngineer:
    """Fit on training data, apply same transforms to test/stream windows.

    Usage:
        fe = FeatureEngineer(dataset="ieee_cis")
        X_train = fe.fit_transform(train_df, y_train)
        X_test  = fe.transform(test_df)
    """

    def __init__(
        self,
        dataset: str,
        prior_weight: float = 300.0,
        v_corr_threshold: float = 0.95,
        top_k_features: int = 200,
    ):
        """
        Args:
            dataset: One of 'ieee_cis', 'gmsc', 'ccfraud'.
            prior_weight: Smoothing weight for target encoding.
            v_corr_threshold: Drop V-columns with pairwise correlation above this.
            top_k_features: Keep top K features by LightGBM importance after engineering.
        """
        if dataset not in ("ieee_cis", "gmsc", "ccfraud"):
            raise ValueError(f"Unknown dataset: {dataset}")
        self.dataset = dataset
        self.prior_weight = prior_weight
        self.v_corr_threshold = v_corr_threshold
        self.top_k_features = top_k_features

        # State fitted on training data
        self._agg_stats: dict = {}
        self._target_enc_maps: dict = {}
        self._global_mean: float = 0.0
        self._v_cols_to_drop: list = []
        self._label_encoders: dict = {}
        self._selected_features: list = []
        self._amount_mean: float = 0.0
        self._amount_std: float = 1.0
        self._fitted = False

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def fit_transform(self, df: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Fit on training data and return engineered features."""
        self._fitted = False
        if self.dataset == "ieee_cis":
            out = self._fit_transform_ieee(df, y)
        elif self.dataset == "gmsc":
            out = self._fit_transform_gmsc(df, y)
        else:
            out = self._fit_transform_ccfraud(df, y)
        self._fitted = True
        return out

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted transforms to new data (no fitting)."""
        if not self._fitted:
            raise RuntimeError("Call fit_transform() before transform()")
        if self.dataset == "ieee_cis":
            return self._transform_ieee(df)
        elif self.dataset == "gmsc":
            return self._transform_gmsc(df)
        else:
            return self._transform_ccfraud(df)

    def streaming_fit_transform(
        self, windows: list[pd.DataFrame], targets: list[pd.Series]
    ) -> list[pd.DataFrame]:
        """Process windows sequentially, fitting only on cumulative past data.

        Feature aggregations for window i are computed using windows 0..i-1 combined.
        Prevents look-ahead leakage in streaming evaluation.
        """
        results = []
        for i, (win_df, win_y) in enumerate(zip(windows, targets)):
            if i == 0:
                out = self.fit_transform(win_df, win_y)
            else:
                past_df = pd.concat(windows[:i], ignore_index=True)
                past_y = pd.concat(targets[:i], ignore_index=True)
                self.fit_transform(past_df, past_y)
                out = self.transform(win_df)
            results.append(out)
        return results

    # ------------------------------------------------------------------ #
    #  IEEE-CIS                                                            #
    # ------------------------------------------------------------------ #

    def _fit_transform_ieee(self, df: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        df = df.copy()
        self._global_mean = y.mean()
        df = self._ieee_temporal(df)
        df = self._ieee_amount(df)
        df = self._ieee_fit_agg(df)
        df = self._ieee_email(df)
        df = self._ieee_d_columns(df)
        df = self._ieee_fit_v_reduction(df)
        df = self._ieee_fit_target_enc(df, y)
        df = self._ieee_label_encode(df, fit=True)
        df = self._ieee_fill_missing(df)
        df = self._ieee_fit_select_features(df, y)
        return df[self._selected_features]

    def _transform_ieee(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = self._ieee_temporal(df)
        df = self._ieee_amount(df)
        df = self._ieee_apply_agg(df)
        df = self._ieee_email(df)
        df = self._ieee_d_columns(df)
        df = self._ieee_drop_v_cols(df)
        df = self._ieee_apply_target_enc(df)
        df = self._ieee_label_encode(df, fit=False)
        df = self._ieee_fill_missing(df)
        available = [c for c in self._selected_features if c in df.columns]
        missing_cols = [c for c in self._selected_features if c not in df.columns]
        for col in missing_cols:
            df[col] = -999
        return df[self._selected_features]

    def _ieee_temporal(self, df: pd.DataFrame) -> pd.DataFrame:
        df["datetime"] = _IEEE_REFERENCE_DATE + pd.to_timedelta(
            df["TransactionDT"], unit="s"
        )
        df["day_of_week"] = df["datetime"].dt.dayofweek
        df["hour_of_day"] = df["datetime"].dt.hour
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
        return df

    def _ieee_amount(self, df: pd.DataFrame) -> pd.DataFrame:
        df["TransactionAmt_log"] = np.log1p(df["TransactionAmt"])
        df["TransactionAmt_decimal"] = df["TransactionAmt"] % 1
        df["TransactionAmt_is_round"] = (df["TransactionAmt"] % 1 == 0).astype(int)
        return df

    def _ieee_fit_agg(self, df: pd.DataFrame) -> pd.DataFrame:
        self._agg_stats = {}
        for col in _AGG_COLS:
            if col not in df.columns:
                continue
            grp = df.groupby(col)["TransactionAmt"]
            self._agg_stats[col] = {
                "count": grp.count(),
                "mean": grp.mean(),
                "std": grp.std().fillna(0),
                "max": grp.max(),
            }
        return self._ieee_apply_agg(df)

    def _ieee_apply_agg(self, df: pd.DataFrame) -> pd.DataFrame:
        for col, stats in self._agg_stats.items():
            if col not in df.columns:
                continue
            df[f"{col}_count"] = df[col].map(stats["count"]).fillna(0)
            df[f"{col}_amt_mean"] = df[col].map(stats["mean"]).fillna(
                stats["mean"].mean()
            )
            df[f"{col}_amt_std"] = df[col].map(stats["std"]).fillna(0)
            df[f"{col}_amt_max"] = df[col].map(stats["max"]).fillna(0)
        return df

    def _ieee_email(self, df: pd.DataFrame) -> pd.DataFrame:
        if "P_emaildomain" in df.columns:
            df["P_email_is_free"] = (
                df["P_emaildomain"].isin(_FREE_EMAIL_DOMAINS).astype(int)
            )
        if "R_emaildomain" in df.columns:
            df["R_email_is_free"] = (
                df["R_emaildomain"].isin(_FREE_EMAIL_DOMAINS).astype(int)
            )
        if "P_emaildomain" in df.columns and "R_emaildomain" in df.columns:
            df["email_match"] = (
                df["P_emaildomain"] == df["R_emaildomain"]
            ).astype(int)
        return df

    def _ieee_d_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        d_cols = [c for c in df.columns if c.startswith("D") and c[1:].isdigit()]
        for col in d_cols:
            df[f"{col}_log"] = np.log1p(df[col].clip(lower=0).fillna(0))
            df[f"{col}_missing"] = df[col].isna().astype(int)
        return df

    def _ieee_fit_v_reduction(self, df: pd.DataFrame) -> pd.DataFrame:
        v_cols = [c for c in df.columns if c.startswith("V") and c[1:].isdigit()]
        v_data = df[v_cols].fillna(-999)
        corr_matrix = v_data.corr().abs()
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        self._v_cols_to_drop = [
            col for col in upper.columns if any(upper[col] > self.v_corr_threshold)
        ]
        drop_path = CONFIGS_DIR / "ieee_v_cols_dropped.txt"
        drop_path.write_text("\n".join(self._v_cols_to_drop))
        print(
            f"V-feature reduction: dropping {len(self._v_cols_to_drop)} / {len(v_cols)} columns"
        )
        return self._ieee_drop_v_cols(df)

    def _ieee_drop_v_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        to_drop = [c for c in self._v_cols_to_drop if c in df.columns]
        return df.drop(columns=to_drop)

    def _ieee_fit_target_enc(self, df: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        self._target_enc_maps = {}
        for col in _TARGET_ENC_COLS:
            if col not in df.columns:
                continue
            tmp = pd.DataFrame({"col": df[col].fillna("unknown"), "target": y.values})
            stats = tmp.groupby("col")["target"].agg(["count", "mean"])
            smoothed = (
                stats["count"] * stats["mean"]
                + self.prior_weight * self._global_mean
            ) / (stats["count"] + self.prior_weight)
            self._target_enc_maps[col] = smoothed
        return self._ieee_apply_target_enc(df)

    def _ieee_apply_target_enc(self, df: pd.DataFrame) -> pd.DataFrame:
        for col, enc_map in self._target_enc_maps.items():
            if col not in df.columns:
                continue
            df[f"{col}_enc"] = (
                df[col].fillna("unknown").map(enc_map).fillna(self._global_mean)
            )
            df = df.drop(columns=[col])
        return df

    def _ieee_label_encode(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        id_cols = [c for c in df.columns if c.startswith("id_")]
        for col in id_cols:
            if col not in df.columns:
                continue
            df[col] = df[col].fillna("unknown").astype(str)
            if fit:
                le = LabelEncoder()
                # Include explicit "unknown" so transform can handle unseen values later
                fit_values = pd.concat(
                    [df[col], pd.Series(["unknown"], index=[-1])], ignore_index=True
                )
                le.fit(fit_values)
                df[col] = le.transform(df[col])
                self._label_encoders[col] = le
            else:
                if col in self._label_encoders:
                    le = self._label_encoders[col]
                    known = set(le.classes_)
                    df[col] = df[col].apply(lambda x: x if x in known else "unknown")
                    df[col] = le.transform(df[col])
        return df

    def _ieee_fill_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        num_cols = df.select_dtypes(include=[np.number]).columns
        df[num_cols] = df[num_cols].fillna(-999)
        obj_cols = df.select_dtypes(include="object").columns
        df[obj_cols] = df[obj_cols].fillna("unknown")
        return df

    def _ieee_fit_select_features(
        self, df: pd.DataFrame, y: pd.Series
    ) -> pd.DataFrame:
        import lightgbm as lgb

        drop_cols = [
            "TransactionID", "isFraud", "datetime", "TransactionDT"
        ]
        feature_cols = [
            c for c in df.columns
            if c not in drop_cols and df[c].dtype != object
        ]
        X = df[feature_cols].fillna(-999)

        model = lgb.LGBMClassifier(
            n_estimators=200,
            num_leaves=63,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        model.fit(X, y)
        importances = pd.Series(model.feature_importances_, index=feature_cols)
        top_features = (
            importances.sort_values(ascending=False)
            .head(self.top_k_features)
            .index.tolist()
        )
        self._selected_features = top_features

        # Save feature list for reproducibility
        feat_path = CONFIGS_DIR / "ieee_selected_features.txt"
        feat_path.write_text("\n".join(top_features))
        print(f"Feature selection: kept {len(top_features)} features")
        return df

    # ------------------------------------------------------------------ #
    #  Give Me Some Credit                                                 #
    # ------------------------------------------------------------------ #

    def _fit_transform_gmsc(self, df: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        df = df.copy()
        df = self._gmsc_clean(df)
        df = self._gmsc_derived(df)
        df = self._gmsc_pseudo_time(df)
        self._selected_features = [
            c for c in df.columns
            if c not in ("SeriousDlqin2yrs",)
        ]
        self._fitted = True
        return df[self._selected_features]

    def _transform_gmsc(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = self._gmsc_clean(df)
        df = self._gmsc_derived(df)
        df = self._gmsc_pseudo_time(df)
        available = [c for c in self._selected_features if c in df.columns]
        for col in self._selected_features:
            if col not in df.columns:
                df[col] = 0
        return df[self._selected_features]

    def _gmsc_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        if "age" in df.columns:
            df.loc[df["age"] == 0, "age"] = np.nan
            df["age"] = df["age"].fillna(df["age"].median())

        for col in [
            "NumberOfTime30-59DaysPastDueNotWorse",
            "NumberOfTimes90DaysLate",
            "NumberOfTime60-89DaysPastDueNotWorse",
        ]:
            if col in df.columns:
                df[col] = df[col].clip(upper=90)

        if "MonthlyIncome" in df.columns:
            df["MonthlyIncome"] = df["MonthlyIncome"].fillna(
                df["MonthlyIncome"].median()
            )
            df["MonthlyIncome_log"] = np.log1p(df["MonthlyIncome"])

        if "RevolvingUtilizationOfUnsecuredLines" in df.columns:
            df["RevolvingUtilizationOfUnsecuredLines"] = df[
                "RevolvingUtilizationOfUnsecuredLines"
            ].clip(upper=1.0)

        df = df.fillna(df.median(numeric_only=True))
        return df

    def _gmsc_derived(self, df: pd.DataFrame) -> pd.DataFrame:
        cols_30_59 = "NumberOfTime30-59DaysPastDueNotWorse"
        cols_90 = "NumberOfTimes90DaysLate"
        cols_60_89 = "NumberOfTime60-89DaysPastDueNotWorse"

        if all(c in df.columns for c in [cols_30_59, cols_90, cols_60_89]):
            df["total_late_payments"] = (
                df[cols_30_59] + df[cols_90] + df[cols_60_89]
            )

        if "DebtRatio" in df.columns and "MonthlyIncome" in df.columns:
            df["debt_to_income"] = df["DebtRatio"] * df["MonthlyIncome"].fillna(0)

        if "MonthlyIncome" in df.columns and "NumberOfDependents" in df.columns:
            df["income_per_dependent"] = df["MonthlyIncome"] / (
                df["NumberOfDependents"].fillna(0) + 1
            )

        if (
            "RevolvingUtilizationOfUnsecuredLines" in df.columns
            and "total_late_payments" in df.columns
        ):
            df["credit_utilization_risk"] = (
                df["RevolvingUtilizationOfUnsecuredLines"] * df["total_late_payments"]
            )

        if "age" in df.columns:
            df["age_band"] = pd.cut(
                df["age"],
                bins=[0, 25, 35, 45, 55, 65, 120],
                labels=False,
            ).astype(float)

        return df

    def _gmsc_pseudo_time(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create synthetic temporal ordering as a pseudo-time index."""
        if "age" in df.columns and "RevolvingUtilizationOfUnsecuredLines" in df.columns:
            df = df.sort_values(
                by=["age", "RevolvingUtilizationOfUnsecuredLines"],
                ascending=[True, False],
            ).reset_index(drop=True)
        df["pseudo_time_index"] = range(len(df))
        return df

    # ------------------------------------------------------------------ #
    #  Credit Card Fraud (ULB)                                            #
    # ------------------------------------------------------------------ #

    def _fit_transform_ccfraud(self, df: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        df = df.copy()
        df = self._ccfraud_temporal(df)
        df = self._ccfraud_amount(df, fit=True)
        df = self._ccfraud_interactions(df)
        self._selected_features = [
            c for c in df.columns if c not in ("Class",)
        ]
        self._fitted = True
        return df[self._selected_features]

    def _transform_ccfraud(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = self._ccfraud_temporal(df)
        df = self._ccfraud_amount(df, fit=False)
        df = self._ccfraud_interactions(df)
        for col in self._selected_features:
            if col not in df.columns:
                df[col] = 0
        return df[self._selected_features]

    def _ccfraud_temporal(self, df: pd.DataFrame) -> pd.DataFrame:
        if "Time" in df.columns:
            df["hour"] = (df["Time"] // 3600) % 24
            df["day"] = (df["Time"] // 86400).astype(int)
            df["is_night"] = df["hour"].between(0, 6).astype(int)
        return df

    def _ccfraud_amount(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        if "Amount" not in df.columns:
            return df
        if fit:
            self._amount_mean = df["Amount"].mean()
            self._amount_std = df["Amount"].std()
            if self._amount_std == 0:
                self._amount_std = 1.0
        df["Amount_log"] = np.log1p(df["Amount"])
        df["Amount_zscore"] = (df["Amount"] - self._amount_mean) / self._amount_std
        df["Amount_is_round"] = (df["Amount"] % 1 == 0).astype(int)
        return df

    def _ccfraud_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        for v1, v2 in _V_INTERACTIONS:
            if v1 in df.columns and v2 in df.columns:
                df[f"{v1}_{v2}"] = df[v1] * df[v2]
        return df
