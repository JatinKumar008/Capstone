from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from dataclasses import dataclass
from typing import List


@dataclass
class Chunk:
    chunk_id: str
    text: str
    doc_type: str
    source_file: str


def load_and_chunk_documents() -> List[Chunk]:
    sub_splitter = RecursiveCharacterTextSplitter(
        chunk_size=650,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " "]
    )
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "H1"), ("##", "H2"), ("###", "H3")],
        strip_headers=False,
    )

    doc_config = [
        ("data/raw/product_terms.txt", "product_terms"),
        ("data/raw/fee_schedule.txt", "fees"),
        ("data/raw/eligibility_rules.txt", "eligibility")
    ]

    all_chunks = []
    for filepath, doc_type in doc_config:
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()

        n_sec = 0
        for sec in header_splitter.split_text(text):
            n_sec += 1
            prefix = " > ".join(v for _, v in sorted(sec.metadata.items()))
            body = sec.page_content.strip()
            if not body:
                continue
            if len(body) <= sub_splitter._chunk_size:
                pieces = [body]
            else:
                pieces = sub_splitter.split_text(body)
            for i, piece in enumerate(pieces):
                piece = piece.strip()
                if prefix and prefix not in piece:
                    piece = f"{prefix}\n{piece}"
                all_chunks.append(Chunk(
                    chunk_id=f"{doc_type}_{len(all_chunks):03d}",
                    text=piece,
                    doc_type=doc_type,
                    source_file=filepath
                ))
        print(f"✅ {doc_type}: {sum(1 for c in all_chunks if c.source_file == filepath)} chunks ({n_sec} sections)")

    return all_chunks