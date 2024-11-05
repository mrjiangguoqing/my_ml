from ray import serve
import mlflow
from transformers import pipeline
from starlette.requests import Request
from typing import Dict, List

@serve.deployment
class TextGenerationService:
    def __init__(self):
        mlflow.set_tracking_uri(uri="http://release-name-mlflow.default.svc.cluster.local:5000")
        model_uri = "/data/model_data/model/model_data/14/4a5f12b720534457a67aa8c5934a8fd3/artifacts/text_generator"  # 替换为实际的 model_uri
        self.model = mlflow.pyfunc.load_model(model_uri)
        self.default_params = {"temperature": 0.4}

    async def __call__(self, request: Request) -> Dict:
        input_data = await request.json()
        text_inputs = input_data.get("texts", [])
        params = input_data.get("params", self.default_params)
        return self.model.predict(text_inputs, **params)

# 启动服务并绑定
text_generation_service = TextGenerationService.bind()
