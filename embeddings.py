import glob
import json
import re

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

KB_GLOB = "data/knowledge_base/*.md"
CASES_PATH = "data/resolved_cases.json"
INDEX_PATH = "embeddings"

FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)$", re.DOTALL)


def parse_frontmatter(raw_text: str):
    """
    Very small YAML-frontmatter parser: no external dependency needed
    since the KB frontmatter is flat key: value pairs (with one list field, tags).
    Returns (metadata_dict, body_text).
    """
    match = FRONTMATTER_RE.match(raw_text)
    if not match:
        return {}, raw_text

    front, body = match.group(1), match.group(2)
    metadata = {}
    for line in front.splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if value.startswith("[") and value.endswith("]"):
            value = [v.strip() for v in value[1:-1].split(",") if v.strip()]
        metadata[key] = value

    return metadata, body.strip()


def load_knowledge_base(pattern: str = KB_GLOB) -> list[Document]:
    docs = []
    paths = sorted(glob.glob(pattern))

    if not paths:
        raise FileNotFoundError(
            f"No files matched '{pattern}'. Check that the knowledge_base "
            "folder from the assignment zip is at data/knowledge_base/."
        )

    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            raw_text = f.read()

        metadata, body = parse_frontmatter(raw_text)

        doc = Document(
            page_content=body,
            metadata={
                "source_id": metadata.get("document_id", path),
                "title": metadata.get("title", ""),
                "status": metadata.get("status", "unknown"),
                "updated": metadata.get("updated", ""),
                "origin": "knowledge_base",
                "file_path": path,
            },
        )
        docs.append(doc)

    print(f"Loaded {len(docs)} knowledge-base documents from {pattern}")
    return docs


def load_resolved_cases(path: str = CASES_PATH) -> list[Document]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    docs = []
    for case in data.get("cases", []):
        lines = [f"Title: {case.get('title', '')}"]

        if case.get("symptoms"):
            lines.append("Symptoms: " + "; ".join(case["symptoms"]))
        if case.get("resolution"):
            lines.append("Resolution steps: " + "; ".join(case["resolution"]))
        if case.get("important_limit"):
            lines.append("Important limit: " + case["important_limit"])
        if case.get("superseded_reason"):
            lines.append("Superseded reason: " + case["superseded_reason"])

        page_content = "\n".join(lines)

        doc = Document(
            page_content=page_content,
            metadata={
                "source_id": case.get("case_id", "UNKNOWN-CASE"),
                "title": case.get("title", ""),
                "status": case.get("status", "unknown"),  
                "product_version": case.get("product_version", ""),
                "origin": "resolved_case",
                "source_documents": case.get("source_documents", []),
            },
        )
        docs.append(doc)

    print(f"Loaded {len(docs)} resolved cases from {path}")
    return docs


def build_index():
    kb_docs = load_knowledge_base()
    case_docs = load_resolved_cases()
    all_docs = kb_docs + case_docs

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100,
    )
    
    chunks = splitter.split_documents(all_docs)
    print(f"Split {len(all_docs)} documents into {len(chunks)} chunks")

    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")

    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(INDEX_PATH)

    print(f"FAISS index built and saved to '{INDEX_PATH}/'")
    return vectorstore


if __name__ == "__main__":
    build_index()