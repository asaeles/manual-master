import os
from pathlib import Path


def resolve_path(path_str: str) -> Path:
    """
    Doc-Block: Resolves a path string into a concrete, absolute Path object.

    Args:
        path_str (str): The raw path string, potentially containing a
            leading '~' or environment variables (e.g. '%LOCALAPPDATA%' on
            Windows, '$VAR' on Linux/Mac).

    Returns:
        Path: A resolved, absolute pathlib.Path object. Returns the current
            directory ('.') if path_str is empty or falsy.

    Raises:
        None
    """
    if not path_str:
        return Path(".")
    return Path(os.path.expandvars(os.path.expanduser(path_str))).resolve()


def sizeof_fmt(num: float, decimal_places: int = 2) -> str:
    """
    Doc-Block: Converts a byte count into a human-readable string.

    Args:
        num (float): The size in bytes.
        decimal_places (int): Number of decimal places to display.

    Returns:
        str: Formatted string representing the size (e.g. '10.50MB').

    Raises:
        None
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB', 'PB']:
        if num < 1024.0 or unit == 'PB':
            return f"{int(num)}{unit}" if unit == 'B' else f"{num:.{decimal_places}f}{unit}"
        num /= 1024.0
    return f"{num:.{decimal_places}f}PB"


def get_safe_path_str(path: Path) -> str:
    """
    Doc-Block: Converts a Path object to a string safe for Windows file I/O.

    Args:
        path (Path): The pathlib object to convert.

    Returns:
        str: A string path safe for OS I/O operations. On Windows, absolute
            paths are prefixed with '\\\\?\\' to bypass the MAX_PATH (260
            char) limit. Falls back to str(path) if resolution fails.

    Raises:
        None: Resolution failures are caught internally.
    """
    try:
        abs_path = path.resolve()
        path_str = str(abs_path)
        if os.name == 'nt' and not path_str.startswith("\\\\?\\"):
            return f"\\\\?\\{path_str}"
        return path_str
    except Exception:
        return str(path)