from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List
from api.model import get_embedding_model
import asyncio
import os
app = FastAPI()

# Create a queue with a maximum size of 10
request_queue = asyncio.Queue(maxsize=int(os.getenv("MAX_PARALLEL_REQUESTS", 10)))

class EmbedRequest(BaseModel):
    model_name: str
    texts: List[str]

@app.post("/embed")
async def embed(request: EmbedRequest):
    try:
        # Check if the queue is full before putting a new request
        if request_queue.full():
            print("Queue is full. Waiting for a slot...")
        
        await request_queue.put(None)
        print(f"Request added to queue. Current queue size: {request_queue.qsize()}")
        
        assert request.model_name in ["bge", "bce"]
        print(f"Processing request: {request.model_name}, {request.texts}")
        embedding_model = get_embedding_model(request.model_name)
        embeddings = embedding_model.embed_documents(request.texts)
        return {"embeddings": embeddings}
    except AssertionError as _:
        return JSONResponse(status_code=400, content={"error": f"Invalid embedding name `{request.model_name}`, only `bge` and `bce` are supported"})
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": f"Failed to get embedding model: {e}"})
    finally:
        request_queue.get_nowait()
        print(f"Request completed. Current queue size: {request_queue.qsize()}")

@app.post("/embed_test")
async def embed_test(request: EmbedRequest):
    print(request)
    return {"message": "Hello World"}