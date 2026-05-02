"""文档加载：从 txt/md/pdf 抽出纯文本，供分块与建索引使用。"""
from pathlib import Path
from pypdf import PdfReader

def load_text_file(path: Path) -> str:
    """UTF-8 读取文本文件，非法字节替换为 �。"""
    return path.read_text(encoding="utf-8", errors="replace")


def load_pdf(path: Path) -> str:
    """逐页抽取 PDF 正文，页面之间用两个换行连接。"""
    reader = PdfReader(str(path))
    parts: list[str] = []
    for page in reader.pages:
        t = page.extract_text()
        if t:
            parts.append(t)
    return "\n\n".join(parts)


def load_document(path: Path) -> str:
    """按后缀分发到文本或 PDF 加载器；不支持的格式抛出 ValueError。"""
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md", ".markdown"}:
        return load_text_file(path)
    if suffix == ".pdf":
        return load_pdf(path)
    raise ValueError(f"Unsupported file type: {suffix}")


def safe_source_name(filename: str) -> str:
    """取文件名（不含上级路径），用作 Chroma metadata 里的 source。"""
    return Path(filename).name
