from src.rag_pipeline import RAGPipline
from langsmith import traceable
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import(
    AnswerRelevancyMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric
)
from evals.harness import load_goldens
from deepeval.models import OllamaModel
llm_as_judge=OllamaModel(
      model="llama3.1:8b"
)
GOLDENS_PATH="goldens/faithfulness_dataset.json"
THRESHOLD=0.7

def run(rag):
     
     # 1. LOAD queries (we only need the queries --- context comes from the pipeline now)
     goldens=load_goldens(GOLDENS_PATH)# json 
     test_cases=[]
     for golden in goldens:
          result=rag.invoke(golden['query'])
          test_cases.append(
               input=golden['query'],
               actual_output=result['answer'],# what the generator produced
               retrieval_context=result['context']# what the generator produced
          )
     metrics=[
          AnswerRelevancyMetric(model=llm_as_judge,threshold=THRESHOLD,include_reason=True),
          FaithfulnessMetric(model=llm_as_judge,include_reason=True,threshold=THRESHOLD),
          ContextualRelevancyMetric(model=llm_as_judge,include_reason=True,threshold=THRESHOLD)
          
     ]
          
          
        
