"""
evals/eval_generator.py
=======================
Component-level evaluation of the GENERATOR, in isolation.

Faithfulness: of the claims in the generated answer, how many are supported
by the context it was given? (Did the generator make things up?)

ISOLATION: we feed the generator the GOLDEN context (the known-good chunks
from the faithfulness dataset), NOT the retriever's output. So a low score
is purely the generator's fault --- the context was already correct.

    python -m evals.eval_generator
"""
from dotenv import load_dotenv
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric,AnswerRelevancyMetric
from src.generator import generate
from evals.harness import load_goldens
from deepeval.models import OllamaModel

load_dotenv()
GOLDEN_PATH="goldens/faithfulness_dataset.json"
model=OllamaModel( model="llama3.1:8b")# model as judge
THRESHOLD=0.7# only claim with a score above 0.7 is considerd faithful

def run():
# 1. LOAD the faithfulness golden set (query + ideal_context)
    golden_set=load_goldens(GOLDEN_PATH)
 # 2. RUN THE GENERATOR on the GOLDEN context (isolation), build one test case each
    test_cases=[]
    for g in golden_set:
        context=g['ideal_context']
        answer=generate(g['query'],context)
        test_cases.append(
            LLMTestCase(
                input=g['query'],
                actual_output=answer,# answer we are judging
                retrieval_context=context# faithfullness being judged on the basis of these retrieved docs
            )
        )
# 3. THE METRICS --- decompose actual_output into claims, attribute each to context
    metrics=[
        FaithfulnessMetric(
            threshold=THRESHOLD,
            model=model,
            include_reason=True    # prints WHY each score --- shows which claims were unsupported

        ),
        AnswerRelevancyMetric(
            threshold=THRESHOLD,
            model=model,
            include_reason=True, 
        )
    ]
# 4. EVALUATE --- runs the metrics on every case, prints a report
    result=evaluate(test_cases=test_cases,metrics=metrics)
    print(result)


if __name__ == "__main__":
    run()



