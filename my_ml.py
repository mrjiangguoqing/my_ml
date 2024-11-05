from ray import serve
import mlflow
from transformers import pipeline
from starlette.requests import Request
from typing import Dict, List

@serve.deployment
class TextGenerationService:
    def __init__(self, model_uri: str):
        self.model = mlflow.pyfunc.load_model(model_uri)
        self.default_params = {"max_length": 512, "do_sample": True, "temperature": 0.4}

    async def __call__(self, request: Request) -> Dict:
        input_data = await request.json()
        text_inputs = input_data.get("texts", [])
        params = input_data.get("params", self.default_params)
        return self.model.predict(text_inputs, **params)

# 启动服务并绑定模型
model_uri = "runs:/4a5f12b720534457a67aa8c5934a8fd3/text_generator"  # 替换为实际的 model_uri
text_generation_service = TextGenerationService.bind(model_uri)
