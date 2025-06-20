
import json
from create_estimator import get_results, convert_to_datetime
from sagemaker.predictor import Predictor
import sagemaker
import json
import pandas as pd
from datetime import timedelta
import matplotlib.pyplot as plt

def convert_df_to_ar_payload(df) -> dict:

    return {
        "instances": [
            {
                "start": df["datetime"].min().strftime("%Y-%m-%d %H:%M:%S"),
                "target": df["count_in"].tolist()
            }
        ],
        "configuration": {
            "num_samples": 100,
            "output_types": ["mean", "quantiles"],
            "quantiles": ["0.1", "0.5", "0.9"]
        }
    }



def send_request_to_endpoint(payload):

    predictor = Predictor(
        endpoint_name="ben-innovation-deepar-forecast",  
        sagemaker_session=sagemaker.Session()
    )

    response = predictor.predict(
        json.dumps(payload),
        initial_args={"ContentType": "application/json"}
    )

    return json.loads(response)


def forecast_to_dataframe(recent_df: pd.DataFrame, raw_prediction: dict) -> pd.DataFrame:
    last_time = pd.to_datetime(recent_df["datetime"].max())
    start_time = last_time + timedelta(hours=1)

    pred = raw_prediction["predictions"][0]
    mean = pred["mean"]
    q10 = pred["quantiles"]["0.1"]
    q50 = pred["quantiles"]["0.5"]
    q90 = pred["quantiles"]["0.9"]

    timestamps = pd.date_range(start=start_time, periods=len(mean), freq="H")

    return pd.DataFrame({
        "datetime": timestamps,
        "mean": mean,
        "p10": q10,
        "p50": q50,
        "p90": q90,
    })


import matplotlib.pyplot as plt

def plot_forecast_with_history(recent_df: pd.DataFrame, forecast_df: pd.DataFrame):
    plt.figure(figsize=(12, 6))

    plt.plot(recent_df["datetime"], recent_df["count_in"], label="Actual", color="black")

    plt.plot(forecast_df["datetime"], forecast_df["mean"], label="Forecast (mean)", linestyle="--")

    plt.fill_between(
        forecast_df["datetime"],
        forecast_df["p10"],
        forecast_df["p90"],
        color="blue",
        alpha=0.2,
        label="90% confidence interval"
    )

    plt.xlabel("Time")
    plt.ylabel("Count of cyclists")
    plt.title("Cyclists counted and predictions for next 24 hours using AWS DeepAR")
    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    # plt.xticks(rotation=45)

    plt.savefig("recent_and_forecast.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    recent_df = get_results(months="6")
    recent_df = convert_to_datetime(recent_df)
    recent_df = recent_df.iloc[-5*24:]
    payload = convert_df_to_ar_payload(recent_df)
    raw_predictions = send_request_to_endpoint(payload)
    forecast_df = forecast_to_dataframe(recent_df, raw_predictions)
    plot_forecast_with_history(recent_df, forecast_df)

