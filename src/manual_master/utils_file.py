from mimetypes import guess_type
from pathlib import Path

from charset_normalizer import from_bytes
from langchain_community.document_loaders import BSHTMLLoader, PyPDFLoader, TextLoader
from langchain_core.documents import Document
from magic import Magic

try:
    _MIME_MAGIC = Magic(mime=True)
    _ENCODING_MAGIC = Magic(mime_encoding=True)
except Exception:
    _MIME_MAGIC = None
    _ENCODING_MAGIC = None


def _detect_encoding_from_header(header: bytes) -> str:
    """
    Doc-Block: Guesses a text encoding from a raw byte header using
    charset-normalizer, for use when libmagic is unavailable or failed.

    Args:
        header (bytes): The leading bytes of the file to analyze.

    Returns:
        str: The best-guess encoding name, or 'utf-8' if detection is
            inconclusive.

    Raises:
        None: All detection failures fall back to 'utf-8'.
    """
    try:
        match = from_bytes(header).best()
        return match.encoding if match else 'utf-8'
    except Exception:
        return 'utf-8'


def detect_file_info(path_obj: Path, safe_path: str) -> tuple[str, str]:
    """
    Doc-Block: Identifies a file's MIME type and text encoding.

    Args:
        path_obj (Path): The pathlib object (used for extension checking).
        safe_path (str): The string path (potentially with \\\\?\\) for file opening.

    Returns:
        tuple[str, str]: A tuple containing (MimeType, Encoding). Defaults
            to a charset-normalizer guess (or 'utf-8') on partial failure,
            or ('error/unreadable', 'unknown') if the file can't be opened.

    Raises:
        None: All I/O and detection errors are caught internally.
    """
    if _MIME_MAGIC is not None and _ENCODING_MAGIC is not None:
        try:
            mime_type = _MIME_MAGIC.from_file(safe_path)
            encoding = _ENCODING_MAGIC.from_file(safe_path)
            return mime_type, encoding
        except Exception as e:
            print(f"  [Warn] Magic detection failed: {e}")

    try:
        with open(safe_path, 'rb') as f:
            header = f.read(1024)

        if header.startswith(b'%PDF'):
            return 'application/pdf', 'binary'
        if header.startswith(b'<?xml') or path_obj.suffix.lower() == '.xml':
            return 'text/xml', _detect_encoding_from_header(header)
        if b'<!DOCTYPE html' in header or b'<html' in header or path_obj.suffix.lower() in ['.html', '.htm']:
            return 'text/html', _detect_encoding_from_header(header)

        m_type, _ = guess_type(path_obj)
        return m_type or 'application/octet-stream', _detect_encoding_from_header(header)

    except IOError:
        return 'error/unreadable', 'unknown'


def load_file_content(
    path_obj: Path,
    safe_path: str,
    mime_type: str,
    root_path: Path | None,
    encoding: str = 'utf-8',
) -> list[Document]:
    """
    Doc-Block: Loads and parses content from a file based on its MIME type.

    Args:
        path_obj (Path): The original Path object.
        safe_path (str): The safe string path for I/O.
        mime_type (str): The detected MIME type used to select the loader.
        root_path (Path | None): The root directory of the scan, used to
            calculate relative paths.
        encoding (str): The detected text encoding to open the file with.
            Falls back to 'utf-8' if the value is unusable as a codec name
            (e.g. 'binary' or 'unknown').

    Returns:
        list[Document]: LangChain Document objects containing the text and
            metadata. Returns an empty list on failure or unsupported type.

    Raises:
        None: Loader errors are caught internally and logged.
    """
    docs = []
    text_encoding = encoding if encoding not in (None, 'unknown', 'binary') else 'utf-8'

    try:
        if 'pdf' in mime_type:
            loader = PyPDFLoader(safe_path)
            docs = loader.load()
        elif 'html' in mime_type:
            loader = BSHTMLLoader(safe_path, open_encoding=text_encoding)
            docs = loader.load()
        elif 'xml' in mime_type:
            try:
                loader = BSHTMLLoader(safe_path, open_encoding=text_encoding, bs_kwargs={"features": "xml"})
                docs = loader.load()
            except Exception:
                loader = TextLoader(safe_path, encoding=text_encoding)
                docs = loader.load()
        elif 'text' in mime_type or 'plain' in mime_type:
            loader = TextLoader(safe_path, encoding=text_encoding)
            docs = loader.load()

        for doc in docs:
            if root_path and path_obj.is_relative_to(root_path):
                doc.metadata['source'] = str(path_obj.relative_to(root_path))
            else:
                doc.metadata['source'] = path_obj.name
        return docs
    except Exception as e:
        print(f"  [Error] Failed to load {path_obj.name}: {e}")
        return []