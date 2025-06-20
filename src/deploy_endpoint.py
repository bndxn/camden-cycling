"""Deploy as serverless realtime endpoint."""

import sagemaker
from sagemaker.model import Model
from sagemaker.serverless import ServerlessInferenceConfig
from sagemaker import Session, image_uris
import os
from dotenv import load_dotenv

load_dotenv()

model_artifact = os.getenv("MODEL_ARTEFACT")

region = Session().boto_region_name
image = image_uris.retrieve("forecasting-deepar", region)

model = Model(
    image_uri=image,
    model_data=model_artifact,
    role=os.getenv("SAGEMAKER_EXECUTION_ROLE")
,
    sagemaker_session=sagemaker.Session()
)

predictor = model.deploy(
    endpoint_name="ben-innovation-deepar-forecast",
    serverless_inference_config=ServerlessInferenceConfig(
        memory_size_in_mb=2048,
        max_concurrency=1
    )
)
