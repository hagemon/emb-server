import requests
from typing import List
from langchain_core.embeddings import Embeddings

class CustomAPIEmbeddings(Embeddings):
    def __init__(self, model_name: str, api_url: str):
        self.model_name = model_name
        self.api_url = api_url

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if isinstance(texts, str):
            texts = [texts]
        print(texts)
        response = requests.post(
            self.api_url,
            json={
                "model_name": self.model_name,
                "texts": texts,
            },
        )
        if response.status_code != 200:
            raise ValueError(f"Failed to get embeddings: {response.status_code}")
        return response.json()['embeddings']  # Adjust this based on the response format of your API

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]
    

if __name__ == "__main__":
    model_name = "bge"
    api_url = "http://localhost:2333/embed"

    embeddings = CustomAPIEmbeddings(model_name, api_url)

    # embed five sentences in different threads
    import concurrent.futures

    def embed_sentence(sentence):
        return embeddings.embed_query(sentence)

    sentences = [
        "Hello, world!",
        "This is a test1.",
        "This is a test2.",
        "This is a test3.",
        "This is a test4.",
    ]*10

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(embed_sentence, sentence) for sentence in sentences]
        for future in concurrent.futures.as_completed(futures):
            print(future.result())
  
    
