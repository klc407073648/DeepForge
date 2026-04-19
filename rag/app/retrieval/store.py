from pathlib import Path
from typing import Any

import chromadb
from chromadb.api import ClientAPI


def get_chroma_client(persist_dir: Path) -> ClientAPI:
    persist_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(persist_dir))


def get_or_create_collection(client: ClientAPI, name: str) -> Any:
    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )
