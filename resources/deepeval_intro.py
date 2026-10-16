from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric# check how relevant the answer is to the query
from deepeval.models import OllamaModel
model = OllamaModel(
    model="llama3.1:8b",
)
case_1 = LLMTestCase(
    input="What is the capital of France?",
    actual_output="The capital of France is Paris.",
)
case_2=LLMTestCase(
     input="What is the capital of France?",
    actual_output="France is a beautiful country famous for its food and wine.",
)


# --- One metric, judged by an LLM
metric=AnswerRelevancyMetric(threshold=0.7,model=model, include_reason=True)
#threshold=0.7 means the the answer will be accepted only if it score>=0.7
# if the score is less than 0.7 the anser fails
evaluate(test_cases=[case_1,case_2],metrics=[metric])