"""
vector_store.py
---------------
Handles everything related to embeddings:
  - Building / querying a FAISS vector store from chunked documents.
  - Training a lightweight LogisticRegression guardrail that classifies
    whether a user query is medical (label=1) or off-topic (label=0).
"""

import asyncio
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# ---------------------------------------------------------------------------
# Embedding model (shared singleton)
# ---------------------------------------------------------------------------
_EMBEDDING_MODEL_NAME = "NeuML/pubmedbert-base-embeddings"


def load_embedding_model(model_name: str = _EMBEDDING_MODEL_NAME) -> HuggingFaceEmbeddings:
    """Return a HuggingFaceEmbeddings instance for *model_name*."""
    return HuggingFaceEmbeddings(model_name=model_name)


# ---------------------------------------------------------------------------
# FAISS vector store
# ---------------------------------------------------------------------------
def build_vector_store(
    docs: list[Document],
    embedding_model: HuggingFaceEmbeddings,
) -> FAISS:
    """
    Build a FAISS vector store from *docs* using *embedding_model*.

    Parameters
    ----------
    docs            : list[Document]       Chunked documents.
    embedding_model : HuggingFaceEmbeddings

    Returns
    -------
    FAISS  A searchable vector store.
    """
    return FAISS.from_documents(docs, embedding_model)


def get_retriever(vector_store: FAISS, k: int = 3):
    """Return a LangChain retriever that fetches the top-*k* chunks."""
    return vector_store.as_retriever(search_kwargs={"k": k})


# ---------------------------------------------------------------------------
# Async embedding helper
# ---------------------------------------------------------------------------
async def _embed_queries_async(
    queries: list[str],
    embedding_model: HuggingFaceEmbeddings,
) -> list[list[float]]:
    tasks = [embedding_model.aembed_query(q) for q in queries]
    return await asyncio.gather(*tasks)


def embed_queries(
    queries: list[str],
    embedding_model: HuggingFaceEmbeddings,
) -> list[list[float]]:
    """Synchronous wrapper around the async batch embedder."""
    return asyncio.run(_embed_queries_async(queries, embedding_model))


# ---------------------------------------------------------------------------
# Guardrail classifier
# ---------------------------------------------------------------------------
def train_guardrail(
    good_embeds: list[list[float]],
    poor_embeds: list[list[float]],
    test_size: float = 0.3,
    random_state: int = 42,
) -> LogisticRegression:
    """
    Train a binary LogisticRegression on pre-computed embeddings.

    Parameters
    ----------
    good_embeds  : embeddings of on-topic (medical) queries  → label 1
    poor_embeds  : embeddings of off-topic / harmful queries → label 0
    test_size    : fraction reserved for evaluation
    random_state : reproducibility seed

    Returns
    -------
    LogisticRegression  Fitted classifier.
    """
    X = poor_embeds + good_embeds
    y = [0] * len(poor_embeds) + [1] * len(good_embeds)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    clf = LogisticRegression()
    clf.fit(X_train, y_train)

    print(f"Train accuracy : {clf.score(X_train, y_train):.3f}")
    print(f"Test  accuracy : {clf.score(X_test,  y_test):.3f}")

    return clf


def is_medical_query(
    query: str,
    embedding_model: HuggingFaceEmbeddings,
    classifier: LogisticRegression,
) -> bool:
    """
    Return True if *query* is classified as a valid medical question.

    Parameters
    ----------
    query           : str
    embedding_model : HuggingFaceEmbeddings
    classifier      : fitted LogisticRegression from :func:`train_guardrail`
    """
    embed = embedding_model.embed_query(query)
    prediction = classifier.predict([embed])[0]
    return bool(prediction == 1)
