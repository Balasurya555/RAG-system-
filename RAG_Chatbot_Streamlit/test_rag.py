import time
import os
import sys

# Setup env
os.environ["GOOGLE_API_KEY"] = "fake_key_just_for_import"

import app

print("Measuring initialization time (PDF loading, splitting, embedding, Chroma)...")
t0 = time.time()
pipeline, error = app.load_rag()
t1 = time.time()

if error:
    print(f"Error loading RAG: {error}")
    sys.exit(1)

init_time = t1 - t0
print(f"Initialization took {init_time:.2f} seconds")
print(f"Number of chunks: {pipeline['num_chunks']}")

print("Measuring retrieval time...")
t2 = time.time()
docs = pipeline["retriever"].invoke("What is a pulsar?")
t3 = time.time()
retrieval_time = t3 - t2
print(f"Retrieval took {retrieval_time:.2f} seconds")
print(f"Retrieved {len(docs)} documents.")
