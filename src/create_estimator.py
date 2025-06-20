# snippet from here https://dev.socrata.com/foundry/opendata.camden.gov.uk/it3h-aqrf

import pandas as pd
from sodapy import Socrata
import sagemaker
import os
from sagemaker.estimator import Estimator

from helpers import upload_to_s3, write_deepar_json

s3_bucket_name = "ben-innovation-bucket"
local_path = "train.json"

def get_results(months: str) -> pd.DataFrame:
    # Unauthenticated client only works with public data sets. Note 'None'
    # in place of application token, and no username or password:
    client = Socrata("opendata.camden.gov.uk", None)
    results = client.get("it3h-aqrf", 
        select="count_in, hour, date, week, month, year", 
        where=f"name = 's62_kentishTownRd_road_cam' and year='2025' and month in ({months})", limit=720)

    # for one month, we expect this to be of size 30 days * 24 hours ~= 720

    return pd.DataFrame.from_records(results)


def convert_to_datetime(df: pd.DataFrame) -> pd.DataFrame:

    df["datetime"] = pd.to_datetime(
    df["year"].astype(str) + "-" +
    df["month"].astype(str).str.zfill(2) + "-" +
    df["date"].astype(str).str.zfill(2) + " " +
    df["hour"]
    )

    df[["count_in"]] = df[["count_in"]].astype(float) 

    df.sort_values("datetime", inplace=True)

    return df.drop(columns=["year", "month", "date", "hour"])



def create_and_train_estimator(s3_bucket_name: str) -> None:
    """First creates an estimator, then trains it on our dataset."""

    sess   = sagemaker.session.Session()
    role   = os.environ["SAGEMAKER_EXECUTION_ROLE"]
    image  = sagemaker.image_uris.retrieve("forecasting-deepar", sess.boto_region_name)

    est = Estimator(
        image_uri=image,
        role=role,
        instance_type="ml.m5.xlarge",
        instance_count=1,
        output_path=f"s3://{sess.default_bucket()}/deepar-output",
        hyperparameters={
            "time_freq": "H",
            "prediction_length": 24,
            "context_length": 72,
            "epochs": 10
        },
    )
    est.fit({"train": f"s3://{s3_bucket_name}/time_series/train/"})



if __name__  == "__main__":
    results_df = get_results(months="5")
    results_df = convert_to_datetime(results_df)
    write_deepar_json(results_df, local_path)
    upload_to_s3(local_path=local_path, bucket=s3_bucket_name, key="time_series/train/train.json")
    create_and_train_estimator(s3_bucket_name=s3_bucket_name)
