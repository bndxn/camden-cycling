import boto3
import matplotlib.pyplot as plt
import pandas as pd
import json


def upload_to_s3(local_path: str, bucket: str, key: str) -> None:
    try:
        s3 = boto3.client("s3")
        s3.upload_file(Filename=local_path, Bucket=bucket, Key=key)
        print(f"Uploaded {local_path} to s3://{bucket}/{key}")
    except Exception as e:
        print(f"Failed to upload {local_path} to s3://{bucket}/{key}: {e}")


def plot_df(df: pd.DataFrame) -> None:

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(df["datetime"], df["count_in"],  label="count_in")

    ax.set_xlabel("Datetime")
    ax.set_ylabel("Count")
    ax.set_title("Count In – May 2025")
    ax.legend()

    fig.tight_layout()
    fig.savefig("count_i.png", dpi=300)  # high-resolution PNG
    plt.show()


def write_deepar_json(df: pd.DataFrame, output_path: str):
    df = df.sort_values("datetime")
    record = {
        "start": df["datetime"].iloc[0].strftime("%Y-%m-%d %H:%M:%S"),
        "target": df["count_in"].tolist()
    }
    with open(output_path, "w") as f:
        json.dump(record, f)