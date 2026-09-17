# /// script
# requires-python = ">=3.11"
# dependencies = ["boto3"]
# ///
"""Usage from CloudWatch, plus the account's model invocation logging status.

Bedrock publishes per-model metrics to the AWS/Bedrock namespace within
minutes of each call, hours before the same usage shows up in Cost
Explorer. This reads the last N hours of them for MODEL_ID. Invocation
logging (full request/response capture to CloudWatch Logs or S3) is off by
default; this only reports whether it is configured, it does not change it.
"""

import datetime
import sys

import boto3
import botocore.exceptions

from config import MODEL_ID, REGION

METRICS = ["Invocations", "InputTokenCount", "OutputTokenCount", "InvocationThrottles"]


def metric_totals(hours: int) -> dict[str, float]:
    cloudwatch = boto3.client("cloudwatch", region_name=REGION)
    end = datetime.datetime.now(datetime.timezone.utc)
    response = cloudwatch.get_metric_data(
        MetricDataQueries=[
            {
                "Id": f"m{index}",
                "Label": name,
                "MetricStat": {
                    "Metric": {
                        "Namespace": "AWS/Bedrock",
                        "MetricName": name,
                        "Dimensions": [{"Name": "ModelId", "Value": MODEL_ID}],
                    },
                    "Period": 3600,
                    "Stat": "Sum",
                },
            }
            for index, name in enumerate(METRICS)
        ],
        StartTime=end - datetime.timedelta(hours=hours),
        EndTime=end,
    )
    return {result["Label"]: sum(result["Values"]) for result in response["MetricDataResults"]}


def invocation_logging_status() -> str:
    config = boto3.client("bedrock", region_name=REGION).get_model_invocation_logging_configuration()
    logging_config = config.get("loggingConfig")
    if not logging_config:
        return "off"
    targets = [key for key in ("cloudWatchConfig", "s3Config") if key in logging_config]
    return "on -> " + ", ".join(targets)


if __name__ == "__main__":
    window_hours = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    print(f"[{MODEL_ID} @ {REGION}, last {window_hours}h]\n")
    try:
        for metric, total in metric_totals(window_hours).items():
            print(f"{metric:>20}: {total:,.0f}")
        print(f"\ninvocation logging: {invocation_logging_status()}")
    except botocore.exceptions.ClientError as error:
        code = error.response["Error"]["Code"]
        print(f"{code}: {error.response['Error']['Message']}", file=sys.stderr)
        sys.exit(1)
