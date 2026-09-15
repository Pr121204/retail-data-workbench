"""Robust CSV reading with deterministic encoding fallback.

The brief lists "mixed encodings/types" among the edge cases the system must
survive. Strategy: try utf-8, then utf-8-sig (BOM), then latin-1. latin-1
never fails (every byte maps to a codepoint), so behaviour is always
deterministic — worst case some characters render lossily, which is
preferable to crashing a whole run on one odd byte.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Union

import pandas as pd

_ENCODING_CANDIDATES = ("utf-8", "utf-8-sig", "latin-1")


def _decode_csv_bytes(raw: bytes, **kwargs: Any) -> pd.DataFrame:
    last_error: Exception | None = None
    for enc in _ENCODING_CANDIDATES:
        try:
            text = raw.decode(enc)
            return pd.read_csv(io.StringIO(text), **kwargs)
        except UnicodeDecodeError as exc:  # try next encoding
            last_error = exc
        except pd.errors.ParserError:
            raise  # genuine CSV structure problem — do not mask with encodings
        except Exception as exc:  # e.g. EmptyDataError, unterminated quotes
            raise exc
    raise last_error if last_error else ValueError("Unable to decode CSV input")


def read_csv_robust(source: Union[str, Path, bytes], **kwargs: Any) -> pd.DataFrame:
    """pd.read_csv that tolerates utf-8/utf-8-sig/latin-1 files and raw bytes.

    Raises the original parser errors for structurally broken CSVs, so callers
    can distinguish "encoding surprise" (handled) from "malformed file"
    (reported as a structured error).
    """
    if isinstance(source, bytes):
        return _decode_csv_bytes(source, **kwargs)
    raw = Path(source).read_bytes()
    return _decode_csv_bytes(raw, **kwargs)
