#!/usr/bin/env python

import os
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, make_scorer
from sklearn.model_selection import train_test_split, cross_validate, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

from scipy.stats import pearsonr

AGGREGATION_INTERVAL = os.environ.get("AGGREGATION_INTERVAL") # 15min
AGGREGATION_METHOD = os.environ.get("AGGREGATION_METHOD") # FULL_SERVICE or SINGLE_BUS
LAGS = int(os.environ.get("LAGS", 3))
AHEAD = int(os.environ.get("AHEAD")) # 1 next bin, 2 next 2 bins, etc.
DATASET_PATH = os.environ.get("DATASET_PATH")
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR"))

if AGGREGATION_INTERVAL is None \
    or AGGREGATION_METHOD is None \
    or AHEAD is None \
    or DATASET_PATH is None \
    or OUTPUT_DIR is None:
    raise ValueError("Missing environment variables")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

output_name = f"results-{AGGREGATION_INTERVAL}-ahead_{AHEAD}-{AGGREGATION_METHOD}.csv"

if os.path.exists(OUTPUT_DIR / output_name):
    print(f"Skipping {output_name}")
    exit(0)

print(f"Computing {AGGREGATION_INTERVAL} ahead-{AHEAD} {AGGREGATION_METHOD}")

####################
#### Experiment ####
####################

seed = 42
num_folds = 5

df = pd.read_csv(
    DATASET_PATH,
    usecols=["open_time", "in_people", "plate"],
    parse_dates=["open_time"],
).set_index('open_time')

if AGGREGATION_METHOD == "FULL_SERVICE":

    aggregated_df = df.resample(AGGREGATION_INTERVAL).agg({
        'in_people': 'sum',
    }).sort_values('open_time').reset_index()

    for i in range(AHEAD, LAGS + AHEAD):
        aggregated_df[f'lag_{i}_in_people'] = aggregated_df['in_people'].shift(i)

    aggregated_df = aggregated_df.dropna().reset_index(drop=True)

elif AGGREGATION_METHOD == "SINGLE_BUS":

    aggregated_df = df.groupby('plate').resample(AGGREGATION_INTERVAL).agg({
        'in_people': 'sum',
    }).sort_values(by=['open_time', 'plate']).dropna().reset_index()

    aggregated_df['op_bus'] = aggregated_df.groupby('open_time').cumcount()

    for plate in aggregated_df['plate'].unique():
        plate_df = aggregated_df[aggregated_df['plate'] == plate]
        for i in range(AHEAD, LAGS + AHEAD):
            aggregated_df.loc[plate_df.index, f'lag_{i}_in_people'] = plate_df['in_people'].shift(i)

    aggregated_df = aggregated_df.dropna().reset_index(drop=True)

    aggregated_df = aggregated_df.drop(columns=['plate'])
    aggregated_df = pd.get_dummies(aggregated_df, columns=['op_bus'])

else:
    raise ValueError("Invalid aggregation method")

def sin_transformer(period):
    return FunctionTransformer(lambda x: np.sin(x / period * 2 * np.pi))
def cos_transformer(period):
    return FunctionTransformer(lambda x: np.cos(x / period * 2 * np.pi))

df = aggregated_df.copy()

df["month_sin"] = sin_transformer(12).fit_transform(df['open_time'].dt.month)
df["month_cos"] = cos_transformer(12).fit_transform(df['open_time'].dt.month)

df["day_sin"] = sin_transformer(31).fit_transform(df['open_time'].dt.day)
df["day_cos"] = cos_transformer(31).fit_transform(df['open_time'].dt.day)

df["day_of_week_sin"] = sin_transformer(7).fit_transform(df['open_time'].dt.day_of_week)
df["day_of_week_cos"] = cos_transformer(7).fit_transform(df['open_time'].dt.day_of_week)

df["hour_sin"] = sin_transformer(24).fit_transform(df['open_time'].dt.hour)
df["hour_cos"] = cos_transformer(24).fit_transform(df['open_time'].dt.hour)

df["minute_sin"] = sin_transformer(60).fit_transform(df['open_time'].dt.minute)
df["minute_cos"] = cos_transformer(60).fit_transform(df['open_time'].dt.minute) 

X = df.drop(columns=['in_people', 'open_time'])
y = df['in_people']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = seed)

# Scorers
scorers = {
    "mae": make_scorer(mean_absolute_error),
    "pearson": make_scorer(lambda y_true, y_pred: pearsonr(y_true, y_pred)[0]),
}

# Dummy MAE
dummy_mae = -mean_absolute_error(df[f'lag_{AHEAD}_in_people'], df['in_people'])
dummy_person = pearsonr(df[f'lag_{AHEAD}_in_people'], df['in_people'])[0]

names = ['Dummy']
mae_results = [dummy_mae]
pearson_results = [dummy_person]

def standard_scaled_model(model):
    return make_pipeline(StandardScaler(), model)

# Regressors
models = []
models.append(('LR' , standard_scaled_model(LinearRegression())))
models.append(('KNN' , standard_scaled_model(KNeighborsRegressor())))
models.append(('SVR' , standard_scaled_model(SVR())))
models.append(('MLP' , standard_scaled_model(MLPRegressor(max_iter=5000, random_state=seed))))
models.append(('RF',  standard_scaled_model(RandomForestRegressor(random_state=seed))))

for name, model in models:
    kfold = KFold(n_splits=num_folds, random_state=seed, shuffle=True)

    res = cross_validate(model, X_train, y_train, cv=kfold, scoring=scorers)

    maes = -np.array(res['test_mae'])
    pearsons = np.array(res['test_pearson'])

    names.append(name)
    mae_results.append(maes.mean())
    pearson_results.append(pearsons.mean())
    print(f"End: {name}")

########################
#### End Experiment ####
########################

with open(OUTPUT_DIR / output_name, 'w') as f:
    for name, mae, pearson in zip(names, mae_results, pearson_results):
        f.write(f"{name};{mae:.2f};{pearson:.2f}\n")
