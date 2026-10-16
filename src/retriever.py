import os 
import glob
import re
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma# store the embeddings in a local directory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

DATA_DIR="data"# directory where the text files are stored
DB_DIR="chroma_store"


#1) Load -- read each transcript, throw away the VTT timestamps
#read each transcript, throw away the VTT timestamps
def load_transcripts(data_dir):
    docs=[]
    for file_path in glob.glob(os.path.join(data_dir,'*.vtt')):
        lines=[]
        for line in open(file_path):
            line=line.strip()
            if line=="WEBVTT" or not line or "-->" in line:
                continue
            lines.append(line)
        text=" ".join(lines)
        session_id=re.search(r"Session[ _]*(\d+)",file_path).group(1)
        docs.append(Document(page_content=text,metadata={"SessionID":session_id}))
    print(len(docs))
    return docs



# 2. BUILD ---- chunk, embed once, and keep it on disk so we don't re-embed
def load_store():
  
    embedding=OllamaEmbeddings(model="nomic-embed-text")
    if os.path.exists(DB_DIR):
        return Chroma(persist_directory=DB_DIR,embedding_function=embedding)
    docs=load_transcripts(DATA_DIR)
    chunks=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=150).split_documents(docs)

    return Chroma.from_documents(chunks,embedding,persist_directory=DB_DIR)# returns a Chroma vector store that can be used for retrieval


def build_retriever():
    return load_store().as_retriever(search_kwargs={"k":5})


if __name__=="__main__":
     
     retriever = build_retriever()

     results = retriever.invoke("what is regression testing?")
     for i in results:
         print(f"SessionID: {i.metadata['SessionID']}, Content: {i.page_content[:200]}...")

    


            
            

    
