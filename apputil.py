import pandas as pd
import numpy as np
import statistics

"""
A class to estimate group-level statistics (mean or median) for categorical data.
"""
class GroupEstimate:
    def __init__(self, estimate):
        
        if estimate not in ("mean", "median"):
            raise ValueError("estimate must be 'mean' or 'median'")
        self.estimate_type = estimate
        self.group_estimates = {}

    """_
    fit(X, y): Fit the model using categorical DataFrame X and 1-D array y.
    """
    def fit(self, X, y):

        #Combine `X` and `y` into a shared pandas DataFrame
        if len(X) != len(y):
            raise ValueError("X and y must be the same length")

        df = X.copy()
        df["_target"] = y

        #Group the DataFrame by the columns in `X`
        grouped = df.groupby(list(X.columns))
    
        #Calculate the estimate for each group
        for group_keys, group_df in grouped:
            values = group_df["_target"].tolist()
            if self.estimate_type == "mean":
                estimate_value = sum(values) / len(values)
            else:
                estimate_value = statistics.median(values)
            self.group_estimates[group_keys] = estimate_value

        """
        Predict estimates for new observations in X_. Returns a NumPy array of estimates.
        """
    def predict(self, X_):

        missing_count = 0
        predictions = []

        X_ = pd.DataFrame([X_]) if not isinstance(X_, pd.DataFrame) else X_

        for _, row in X_.iterrows():
            key = tuple(row[col] for col in X_.columns)
            estimate = self.group_estimates.get(key, np.nan)
            if pd.isna(estimate):
                missing_count += 1
            predictions.append(estimate)

        if missing_count > 0:
            print(f"{missing_count} group(s) were missing from training data.")

        return np.array(predictions)

