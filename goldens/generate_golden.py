
import random,json,re,os,glob
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from deepeval.models import OllamaModel
from deepeval.synthesizer import Synthesizer # generate goldens from contexts
DATA_DIR="data"
DB_DIR="chroma_store"
model = OllamaModel(
    model="llama3.1:8b",
)

def load_chunks():
    texts=[]
    for file_path in glob.glob(os.path.join(DATA_DIR,"*.vtt")):
        lines=[]# read each transcript 
        for line in open(file_path):
            line=line.strip()
            if line=="WEBVTT" or not line or "-->" in line:
                continue
            lines.append(line)
        texts.append(" ".join(lines))
    splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=150)
    return splitter.split_text("\n\n".join(texts))

# --- generate ---
chunks=load_chunks()
sample=random.sample(chunks,min(4,len(chunks)))# randomly sample 16 chunks out of the total chunks
# then give thses chunks the synthesizer to generate a golden answer for each chunk
synthesizer=Synthesizer(model=model)
contexts=[[c] for c in sample]

goldens=synthesizer.generate_goldens_from_contexts(
    contexts=contexts,
    include_expected_output=True,
      max_goldens_per_context=1,   
)

rows = []
for i, g in enumerate(goldens, 1):
    rows.append({
        "id": f"g{i:03d}",
        "query": g.input,
        "ideal_answer": g.expected_output,
        "source": "TODO-verify",        # Synthesizer won't know the session -- you fill this
    })

with open("goldens/retriever_deepeval_goldens.json", "w") as f:
    json.dump(rows, f, indent=2, ensure_ascii=False)

print(f"wrote {len(rows)} DRAFT goldens -> goldens/component_goldens_draft.json")
print("!! REVIEW EVERY ONE before using: check grounding, trim padding, fix leading questions.")
        
    
