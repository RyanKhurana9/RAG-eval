import json
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from src.rag_pipeline import RAGPipline
from deepeval.metrics import ToxicityMetric

GOLDEN_PATH = "goldens/toxicity_goldens.json"
from deepeval.models import OllamaModel
model=OllamaModel(
     model="llama3.1:8b"
)

