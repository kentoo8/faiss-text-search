import numpy as np
import pandas as pd

from utils.utils import FaissDriver, OpenAIEmbeddingsWithRateLimit


def main():
    model = OpenAIEmbeddingsWithRateLimit()
    faiss_driver = FaissDriver()

    sample_texts = pd.read_csv("./data/sample.csv", usecols=["text"])
    embedded = np.array(model.embed_documents(sample_texts["text"]))

    distances_and_ids = faiss_driver.get_indexes_and_distances(
        base_vectors=embedded,
        num_neighbors=10,
    )
    ids = distances_and_ids["ids"]
    distances = distances_and_ids["distances"]

    print(ids)
    print(distances)


if __name__ == "__main__":
    main()
