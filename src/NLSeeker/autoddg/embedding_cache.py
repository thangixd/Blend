from __future__ import annotations

import hashlib
import pickle
from pathlib import Path

import duckdb
import numpy as np

# Typing imports
from typing import Sequence


class EmbeddingCache:
    def __init__(self, *, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._con = duckdb.connect(str(self.path))
        self._con.execute(
            "CREATE TABLE IF NOT EXISTS emb_cache ("
            "  key VARCHAR PRIMARY KEY, "
            "  vector BLOB NOT NULL"
            ")"
        )

    @staticmethod
    def _embedder_signature(embedder) -> str:
        """Stable string identifying the embedder. Falls back to type-and-dim
        when ``embedder.signature()`` is not provided."""
        sig = getattr(embedder, "signature", None)
        if callable(sig):
            return str(sig())
        return f"{type(embedder).__name__}:dim={getattr(embedder, 'embedding_dim', '?')}"

    @staticmethod
    def _key(text: str, embedder_sig: str) -> str:
        """sha256(embedder_sig || \\x00 || text) - used as the row key."""
        h = hashlib.sha256()
        h.update(embedder_sig.encode("utf-8"))
        h.update(b"\x00")
        h.update(text.encode("utf-8"))
        return h.hexdigest()

    def encode(self, embedder, texts: Sequence[str]) -> np.ndarray:
        """Return embeddings for ``texts``, serving hits from the DuckDB cache.

        Misses are forwarded to ``embedder.encode`` in a single bulk call and stored.
        Return order matches the input ``texts`` order. Returns an empty
        ``(0,)`` array if ``texts`` is empty (no embedder call made).
        """
        if not texts:
            return np.empty((0,), dtype=np.float32)
        emb_sig = self._embedder_signature(embedder)
        keys = [self._key(t, emb_sig) for t in texts]
        # Bulk-fetch existing rows.
        placeholders = ",".join(["?"] * len(keys))
        rows = self._con.execute(
            f"SELECT key, vector FROM emb_cache WHERE key IN ({placeholders})",
            keys,
        ).fetchall()
        hit = {k: pickle.loads(v) for k, v in rows}

        miss_idx = [i for i, k in enumerate(keys) if k not in hit]
        if miss_idx:
            miss_texts = [texts[i] for i in miss_idx]
            miss_vecs = embedder.encode(miss_texts)
            self._con.executemany(
                "INSERT OR REPLACE INTO emb_cache (key, vector) VALUES (?, ?)",
                [
                    (keys[i], pickle.dumps(miss_vecs[j], protocol=4))
                    for j, i in enumerate(miss_idx)
                ],
            )
            for j, i in enumerate(miss_idx):
                hit[keys[i]] = miss_vecs[j]

        out = np.stack([hit[k] for k in keys])
        return out

    def close(self) -> None:
        """Close the underlying DuckDB connection. Idempotent."""
        if self._con is not None:
            self._con.close()
            self._con = None

    def __enter__(self) -> "EmbeddingCache":
        return self

    def __exit__(self, *_) -> None:
        self.close()


__all__ = ["EmbeddingCache"]
