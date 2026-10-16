from src.reranker import RerankingRetriever
from langsmith import traceable
from src.generator import generate



class RAGPipline:
    def __init__(self,fetch_k=10,top_k=5):
        self.retrieved=RerankingRetriever(fetch_k=fetch_k,top_k=top_k)

    def invoke(self,query):
        # step1 get the top k retrieved documents using the reranking retriever
        retrieved_docs=self.retrieved.invoke(query)

        # send these retrieved doc to the generator to produce an answer
        context=[doc.page_content for doc in retrieved_docs]

        answer=generate(query,context)

           # return all three legs of the triad so the eval harness can score them
        return {
            "query":query,
            "context":context,
            "answer":answer
        }
if __name__=="__main__":
    rag_pipeline=RAGPipline(fetch_k=10,top_k=5)
    query="what is online eval?"
    print("........ Pipeline started ........")
    result=rag_pipeline.invoke(query)
    print("Query:",result["query"])
    print("*"*60)
    print("Context:",result["context"])
    print("*"*60)
    print("Answer:",result["answer"])
    print("*"*60)
  

