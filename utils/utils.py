import time
from typing import List

import faiss
import numpy as np
from langchain.embeddings import OpenAIEmbeddings


class OpenAIEmbeddingsWithRateLimit(OpenAIEmbeddings):
    def embed_documents(
        self,
        documents: List[str],
        chunk_size: int = 100,
        sleep_time: int = 30,
        max_retries: int = 5,
    ) -> List[List[float]]:
        """
        既存のembed_documentsメソッドをオーバーライドします。
        chunk_sizeに分けて埋め込みを実行し、Rate Limit時に待機するように変更しています。
        """

        embeddings = []
        for i in range(0, len(documents), chunk_size):
            print(i)
            chunk = documents[i : i + chunk_size]

            retries = 0
            while retries < max_retries:
                try:
                    embeddings_chunk = super().embed_documents(chunk)
                    embeddings.extend(embeddings_chunk)
                    break
                except Exception as e:  # FIXME: API制限エラーの具体的な例外クラスに置き換えてください
                    print(
                        f"API制限エラーが発生しました({retries+1}回目): {e}.\n\n{sleep_time}秒間待機します。"
                    )
                    time.sleep(sleep_time)
                    retries += 1

            if retries == max_retries:
                raise Exception("API制限エラーが続いています。処理を中断します。")

        return embeddings


class FaissDriver:
    def __init__(self):
        pass

    @staticmethod
    def array_to_index(vectors: np.ndarray) -> faiss.Index:
        if len(vectors.shape) != 2:
            raise ValueError("Expected a 2D array")

        dim = vectors.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(vectors)
        return index

    @staticmethod
    def index_to_array(index: faiss.Index, dtype="float32") -> np.ndarray:
        if not isinstance(index, faiss.Index):
            raise ValueError("Expected a faiss.Index instance")

        num_vectors = index.ntotal
        vectors = index.reconstruct_batch(range(num_vectors))
        return vectors.astype(dtype)

    @staticmethod
    def read_index(path: str) -> None:
        index = faiss.read_index(path)
        return index

    @staticmethod
    def write_index(index: faiss.Index, path: str) -> None:
        faiss.write_index(index, path)

    @staticmethod
    def get_indexes_and_distances(
        base_vectors: np.ndarray,
        query_vectors: np.ndarray | None = None,
        num_neighbors: int = 3,
    ) -> dict:
        """
        base_vectors  の shape: (num_base_vectors , dim)
        query_vectors の shape: (num_query_vectors, dim)
        distances     の shape: (num_query_vectors, num_neighbors)
        ids           の shape: (num_query_vectors, num_neighbors)
        """
        base_index = FaissDriver.array_to_index(base_vectors)

        if query_vectors is None:
            query_vectors = FaissDriver.index_to_array(base_index)

        distances, ids = base_index.search(x=query_vectors, k=num_neighbors)
        return {
            "distances": distances,
            "ids": ids,
        }
