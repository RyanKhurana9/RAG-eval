""""
dump_chunks.py — export all stored chunks from the Chroma store to JSON.

Assumes your retriever code lives in src/retriever.py and exposes load_store()
(which opens the persisted Chroma store without rebuilding).

Run from the project root:
    python dump_chunks.py

Produces chunks_dump.json — every chunk's id, text, and metadata — so you can
scan them (e.g. to hand-pick ideal_context for the faithfulness dataset).
"""
# basically move the chunks from chroma to a json file so that we can use it to create faithfulness_golden_dataset
import json
from pydoc import doc
from src.retriever import load_store
from collections import Counter

store=load_store()# load the chroma vector store
data = store._collection.get(include=["metadatas", "documents"])# get the data from the store
#._collection is used to directly access the underlying collection of chroma (directly access the stored Chroma data) and get is used to access the stored data in the collection. The include parameter specifies which fields to retrieve from the collection. In this case, we want to retrieve the metadatas, documents, and ids of the stored chunks.
dump=[
        {"id": i, "text": d, "meta": m}
    for i,d,m in zip(data['ids'],data['documents'],data['metadatas'])
]


# SOrt the chunks by their ID to ensure consistent ordering in the output file
dump.sort(key=lambda c:(str(c['meta'].get("session","")),c['id']))
# key allows us to specify a function that extracts a comparison key for each element in the list
# NOW THE DUMP IS SORTED BY SESSION AND THEN BY ID.
"""
So:

session 2 → chunk_3
session 1 → chunk_2
session 1 → chunk_1
session 2 → chunk_1

After sorting:

session 1 → chunk_1
session 1 → chunk_2
session 2 → chunk_1
session 2 → chunk_3

This makes related chunks appear together.
"""

# Now dump these sorted chunks to a JSON file
with open("chunks_dump.json","w") as f:
    json.dump(dump,f,indent=2)
print(f"Dumped {len(dump)} chunks to chunks_dump.json")  

counter=Counter(c['meta'].get("session","?")for c in dump)
print("Chunks per session:",dict(counter))

