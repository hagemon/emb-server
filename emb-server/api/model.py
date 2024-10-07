import os
import threading
from langchain_huggingface import HuggingFaceEmbeddings
from api.config import config


_embedding_models = {}
_loading_locks = {}

def get_embedding_model(embedding_name="bge"):
    global _embedding_models, _loading_locks

    if embedding_name in _embedding_models:
        return _embedding_models[embedding_name]

    if embedding_name not in _loading_locks:
        _loading_locks[embedding_name] = threading.Lock()

    with _loading_locks[embedding_name]:
        # Double-check if the model was loaded while waiting for the lock
        if embedding_name in _embedding_models:
            return _embedding_models[embedding_name]

        encode_kwargs = {"normalize_embeddings": True}
        model_paths: dict = config.get("model_path", {})
        base_path = config.get("base_path", '')
        
        try:
            path = model_paths.get(embedding_name, None)
            assert path is not None
            path = os.path.join(base_path, path)
            assert os.path.exists(path)
        except AssertionError as _:
            raise ValueError(f"Model path `{path}` not found")

        embedding_model = HuggingFaceEmbeddings(
            model_name=path, encode_kwargs=encode_kwargs
        )

        _embedding_models[embedding_name] = embedding_model

    return embedding_model


if __name__ == "__main__":
    model = get_embedding_model("bge")
    print(model)
