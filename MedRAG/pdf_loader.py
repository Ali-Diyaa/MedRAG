"""
pdf_loader.py
-------------
Reads a PDF file and returns LangChain Document objects ready for
embedding.  Tables and plain text are extracted separately so the
vector store keeps them distinguishable by metadata.
"""

import pdfplumber
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def load_pdf(file_path: str) -> list[Document]:
    """
    Extract text and tables from every page of *file_path*.

    Parameters
    ----------
    file_path : str
        Absolute or relative path to the PDF file.

    Returns
    -------
    list[Document]
        One Document per text-block or table found on each page.
        Metadata contains ``{"page": int, "type": "text"|"table"}``.

    Raises
    ------
    FileNotFoundError
        If *file_path* does not exist.
    RuntimeError
        If pdfplumber cannot open the file.
    """
    raw_documents: list[Document] = []

    with pdfplumber.open(file_path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            # --- Plain text ------------------------------------------------
            text = page.extract_text()
            if text and text.strip():
                raw_documents.append(
                    Document(
                        page_content=text,
                        metadata={"page": page_number, "type": "text"},
                    )
                )

            # --- Tables -------------------------------------------------------
            for table in page.extract_tables():
                table_text = "\n".join(
                    " | ".join(str(cell) for cell in row) for row in table
                )
                if table_text.strip():
                    raw_documents.append(
                        Document(
                            page_content=table_text,
                            metadata={"page": page_number, "type": "table"},
                        )
                    )

    return raw_documents


def split_documents(
    documents: list[Document],
    chunk_size: int = 800,
    chunk_overlap: int = 50,
) -> list[Document]:
    """
    Split *documents* into smaller chunks suitable for embedding.

    Parameters
    ----------
    documents   : list[Document]  Raw documents from :func:`load_pdf`.
    chunk_size  : int             Maximum characters per chunk.
    chunk_overlap : int           Overlap between consecutive chunks.

    Returns
    -------
    list[Document]  Chunked documents preserving original metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", ";", ",", " ", ""],
    )
    return splitter.split_documents(documents)


def load_and_split(file_path: str, **kwargs) -> list[Document]:
    """Convenience wrapper: load then split in one call."""
    raw = load_pdf(file_path)
    return split_documents(raw, **kwargs)
