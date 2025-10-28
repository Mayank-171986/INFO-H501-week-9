import pandas as pd
import numpy as np
import statistics

class GroupEstimate:
    def __init__(self, estimate="mean"):
        """
        Initialize the estimator with strategy: 'mean' or 'median'.
        """
        if estimate not in ("mean", "median"):
            raise ValueError("estimate must be 'mean' or 'median'")
        self.estimate_type = estimate
        self.group_estimates = {}

    def fit(self, X, y):
        """
        Fit the model using categorical DataFrame X and 1-D array y.
        Stores group-level estimates only.
        """
        if len(X) != len(y):
            raise ValueError("X and y must be the same length")
        if pd.isnull(y).any():
            raise ValueError("y must not contain missing values")

        df = X.copy()
        df["_target"] = y

        grouped = df.groupby(list(X.columns))

        for group_keys, group_df in grouped:
            values = group_df["_target"].tolist()
            if self.estimate_type == "mean":
                estimate_value = sum(values) / len(values)
            else:
                estimate_value = statistics.median(values)
            self.group_estimates[group_keys] = estimate_value

    def predict(self, X_):
        """
        Predict estimates for new observations in X_.
        Returns a NumPy array of estimates.
        """
        if isinstance(X_, pd.Series):
            X_ = X_.to_frame().T
        elif isinstance(X_, dict):
            X_ = pd.DataFrame([X_])
        elif isinstance(X_, list):
            X_ = pd.DataFrame(X_)
        elif not isinstance(X_, pd.DataFrame):
            raise TypeError("X_ must be a DataFrame, Series, dict, or list of dicts")

        missing_count = 0
        predictions = []

        for _, row in X_.iterrows():
            key = tuple(row[col] for col in X_.columns)
            estimate = self.group_estimates.get(key, np.nan)
            if pd.isna(estimate):
                missing_count += 1
            predictions.append(estimate)

        if missing_count > 0:
            print(f"{missing_count} group(s) were missing from training data.")

        return np.array(predictions)

    def __repr__(self):
        """
        Display stored group estimates.
        """
        return '\n'.join(
            f"{group}: {self.estimate_type}={value:.2f}"
            for group, value in self.group_estimates.items()
        )
