from sentence_transformers import CrossEncoder
from langsmith import traceable
from .retriever import load_store
model =CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
class RerankingRetriever:
    def __init__(self,fetch_k=10,top_k=5):
        self.store=load_store()# chroma vector store
        self.fetch_k=fetch_k# how many the bi-encoder brings back (over-retrieve)
        self.top_k=top_k# top out of the k fetched results to return after reranking
        self.reranker=model
    @traceable(run_type="retriever", name="RerankingRetriever")
    def invoke(self,query):
        potential_candidates=self.store.similarity_search(query,k=self.fetch_k)# find out thr top k candidates using the bi-encoder as both using differnent encoders 
        # in cross encoder we send the raw query and raw document text in pair in order to get a relevance score for each document
        
        # step-1 fetch the top k candidates using the bi-encoder
        candidates=self.store.similarity_search(query,k=self.fetch_k)
        # rerank the candidates using the cross encoder
        # cross encoder takes a list of tuples (query,document)
        rerank_input=[(query,doc.page_content) for doc in candidates]

        # send these pairs to the cross encoder to get the relevance scores
        scores=self.reranker.predict(rerank_input)

        # sort by scores and keep the top k results
        sorted_candidates=sorted(zip(candidates,scores),key=lambda x:x[1],reverse=True)
        keep_top_k=sorted_candidates[:self.top_k]
        return [doc for doc,_ in keep_top_k]
    