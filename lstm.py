
from glob import glob
import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
import matplotlib.pyplot as plt


train_data = pd.DataFrame()
test_data = pd.DataFrame()
prediction_length = 100
files = glob("./Data/Simulated data/*.csv")
for i, file in enumerate(files):
    # print(file[22:-4])
    serie = pd.read_csv(file)
    serie.insert(0, "item_id", file[22:-4], True)
    # print(serie.head())
    train_data = pd.concat([train_data, serie[:-prediction_length]])
    test_data = pd.concat([test_data, serie])

# print(len(train_data))
# print(train_data.head())
# print(len(test_data))
# print(test_data.head())

# train_data = TimeSeriesDataFrame(train_data, id_column="item_id", timestamp_column="timestamp")
# predictor = TimeSeriesPredictor(prediction_length=prediction_length, path="./Models", target="target", eval_metric="WQL")
# predictor.fit(train_data, presets="best_quality", time_limit=600)
predictor = TimeSeriesPredictor.load(path="./Models")
predictions = predictor.predict(train_data)
# predictions.head()

# PLOT
# plt.figure(figsize=(20, 6))
plt.figure()

item_id = "t1"
y_past = train_data.loc[train_data["item_id"] == item_id]["target"]
y_pred = predictions.loc[item_id]
y_test = test_data.loc[test_data["item_id"] == item_id]["target"][-prediction_length:]

plt.plot(y_past, label="Past time series values")
plt.plot(y_pred["mean"], label="Mean forecast")
plt.plot(y_test, label="Future time series values")

plt.fill_between(
    y_pred.index, y_pred["0.1"], y_pred["0.9"], color="red", alpha=0.1, label=f"10%-90% confidence interval"
)
plt.legend()
plt.show()
predictor.leaderboard(test_data)