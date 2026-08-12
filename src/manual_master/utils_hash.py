import hashlib
from pathlib import Path


def scan_content_and_hash(source_root: Path) -> tuple[str, list[Path]]:
    """
    Scans the directory for relevant files and generates a deterministic hash
    based on the file contents (header + tail + size), NOT the filename or directory path.

    This implements "Content-Based Addressing" for the database cache.
    Renaming the folder or files will NOT break the cache/hash as long as content matches.

    Args:
        source_root (Path): The root directory or file to scan.

    Returns:
        tuple[str, list[Path]]: A tuple containing the (master_hex_hash, list_of_valid_paths).
    """
    signatures = []
    valid_files = []

    # Extensions to consider for the hash. We filter here for speed during the hash phase.
    # We purposefully exclude .exe, .dll, etc. to avoid hashing massive binaries unnecessarily.
    relevant_extensions = {'.pdf', '.txt', '.html', '.htm', '.xml', '.md'}

    if not source_root.exists():
        return hashlib.sha256(b'empty').hexdigest(), []

    items_to_scan = [source_root] if source_root.is_file() else [p for p in source_root.rglob('*') if p.is_file()]

    print(f"Scanning content signature for {len(items_to_scan)} items...")

    for path in items_to_scan:
        # Fast extension check
        if path.suffix.lower() not in relevant_extensions:
            continue

        try:
            # 1. Get Size
            size = path.stat().st_size

            # 2. Read Head (and Tail) safely as binary
            # Reading tail ensures we distinguish files that share a common header (like PDF templates)
            with open(path, "rb") as f:
                header = f.read(1024)

                # Logic to read tail if file is large enough
                tail = b''
                if size > 2048:
                    f.seek(-1024, 2) # Seek to 1024 bytes before end
                    tail = f.read(1024)

            # 3. Create component hash (Size + Header + Tail)
            # We encode size to ascii bytes to safely mix with binary data
            # Filename is EXCLUDED from the hash to allow renaming
            file_data = str(size).encode('ascii') + header + tail
            signatures.append(hashlib.sha256(file_data).hexdigest())
            valid_files.append(path)
        except (IOError, OSError) as e:
            # If unreadable, we skip it for the hash (and thus for ingestion)
            # OR we could add an error signature. Skipping is safer for stability.
            continue

    # 5. Master Hash
    # We hash the concatenated sorted signatures
    if not signatures:
        return hashlib.sha256(b'empty').hexdigest(), []

    master_hash = hashlib.sha256("".join(sorted(signatures)).encode('ascii')).hexdigest()
    return master_hash, valid_files
