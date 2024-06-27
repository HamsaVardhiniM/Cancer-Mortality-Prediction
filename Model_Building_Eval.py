import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split as tts
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pickle

# Function to split data into training and testing sets
def split_data(dataframe, target_column):
    X = dataframe.drop(target_column, axis=1)
    y = dataframe[target_column]
    X_train, X_test, y_train, y_test = tts(X, y, test_size=0.2, random_state=0)
    return X_train, X_test, y_train, y_test

# Function to train a linear regression model
def lr_model(x_train, y_train):
    x_train_with_intercept = sm.add_constant(x_train)
    lr = sm.OLS(y_train, x_train_with_intercept).fit()
    return lr

# Function to identify significant variables based on p-values
def identify_significant_vars(lr, p_value_threshold=0.05):
    print(lr.pvalues)
    print("R-squared:", lr.rsquared)
    print("Adjusted R-squared:", lr.rsquared_adj)
    significant_vars = [var for var in lr.pvalues.keys() if lr.pvalues[var] < p_value_threshold]
    if 'const' in significant_vars:
        significant_vars.remove('const')
    return significant_vars

# Function to calculate VIF and remove high VIF columns
def calculate_vif(X):
    X = sm.add_constant(X)
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif_data

def remove_high_vif_columns(X, threshold=10):
    while True:
        vif_data = calculate_vif(X)
        max_vif = vif_data["VIF"].max()
        if max_vif > threshold:
            feature_to_remove = vif_data.sort_values("VIF", ascending=False).iloc[0]["feature"]
            if feature_to_remove == "const":
                break
            print(f"Removing {feature_to_remove} with VIF: {max_vif}")
            X = X.drop(columns=[feature_to_remove])
        else:
            break
    return X

# Load your dataset
df = pd.read_csv("Cancer_Mortality_Cleaned_data.csv")

# Remove high VIF columns and train the model
X = df.drop('target_deathrate', axis=1)
Y = df['target_deathrate']

X_reduced = remove_high_vif_columns(X)
final_df = pd.concat([X_reduced, Y], axis=1)

# Split data into training and testing sets
x_train, x_test, y_train, y_test = split_data(final_df, "target_deathrate")

# Train a linear regression model on the training data
lr = lr_model(x_train, y_train)
summary = lr.summary()
print(summary)

# Save the trained model
with open("trained_model.pkl", "wb") as f:
    pickle.dump(lr, f)

