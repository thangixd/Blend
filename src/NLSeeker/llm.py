import gc
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import numpy as np

from src.NLSeeker.config import NLSeekerConfig

# Typing imports
from typing import Any, Optional, Protocol, Sequence

LOG = logging.getLogger(__name__)

_BACKEND_CACHE: dict = {}
_CACHE_LOCK = threading.Lock()

# Number of in-flight HTTP requests against the OpenAI/vLLM endpoint per
# generate() call.
_OPENAI_LLM_CONCURRENCY = 256


def _is_ollama_endpoint(base_url: str | None) -> bool:
    """Detect Ollama by port - Ollama defaults to 11434 and rejects 'seed='."""
    if not base_url:
        return False
    return ":11434" in base_url


class LLMBackend(Protocol):
    def generate(
        self,
        prompts: Sequence[str],
        max_new_tokens: int,
        *,
        concurrency: int | None = None,
    ) -> list: ...

    @property
    def max_input_tokens(self) -> int: ...


class EmbedBackend(Protocol):
    def encode(self, texts: Sequence[str]) -> np.ndarray: ...

    @property
    def embedding_dim(self) -> int: ...

    @property
    def max_input_tokens(self) -> int: ...


@dataclass
class _LocalLLM:
    """Causal-LM (HF) wrapper with adaptive batch sizing on CUDA OOM."""

    model_path: str
    max_batch: int
    hf_token: str = ""

    def __post_init__(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

        set_seed(42)
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

        self._torch = torch
        token = self.hf_token or None
        LOG.info("Loading local LLM %s (this may download weights)", self.model_path)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path, token=token)
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._tokenizer.padding_side = "left"  # required for causal-LM batched gen

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
            token=token,
        )
        self._model.eval()
        self._current_batch = self.max_batch

    @property
    def max_input_tokens(self) -> int:
        cfg_window = getattr(self._model.config, "max_position_embeddings", None)
        return int(cfg_window or 32768)

    def generate(
        self,
        prompts: Sequence[str],
        max_new_tokens: int,
        *,
        concurrency: int | None = None,
    ) -> list:
        # ``concurrency`` is a no-op for the local HF backend (in-process
        # decoding); accepted only to satisfy the LLMBackend Protocol.
        del concurrency
        if not prompts:
            return []
        out = []
        idx = 0
        while idx < len(prompts):
            batch_size = max(1, min(self._current_batch, len(prompts) - idx))
            chunk = list(prompts[idx : idx + batch_size])
            try:
                out.extend(self._generate_chunk(chunk, max_new_tokens))
                if self._current_batch < self.max_batch:
                    self._current_batch = min(self.max_batch, self._current_batch + 1)
                idx += batch_size
            except self._torch.cuda.OutOfMemoryError:
                self._torch.cuda.empty_cache()
                gc.collect()
                if batch_size == 1:
                    raise
                self._current_batch = max(1, batch_size // 2)
                LOG.warning(
                    "CUDA OOM at batch=%d; retrying at batch=%d",
                    batch_size,
                    self._current_batch,
                )
        return out

    def _generate_chunk(self, prompts: list, max_new_tokens: int) -> list:
        torch = self._torch
        # Qwen-Instruct expects a chat-formatted user turn.
        messages = [
            self._tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for p in prompts
        ]
        encoded = self._tokenizer(
            messages,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_input_tokens - max_new_tokens - 16,
        ).to(self._model.device)
        with torch.inference_mode():
            output_ids = self._model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.0,
                pad_token_id=self._tokenizer.pad_token_id,
            )
        prompt_len = encoded["input_ids"].shape[1]
        completions = self._tokenizer.batch_decode(
            output_ids[:, prompt_len:], skip_special_tokens=True
        )
        return [c.strip() for c in completions]


@dataclass
class _LocalEmbedder:
    """sentence-transformers embedder with a fixed batch size."""

    embed_path: str
    hf_token: str = ""
    batch_size: int = 10

    def __post_init__(self) -> None:
        import torch
        from sentence_transformers import SentenceTransformer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        LOG.info("Loading local embedder %s on %s", self.embed_path, device)
        self._model = SentenceTransformer(
            self.embed_path,
            device=device,
            token=self.hf_token or None,
        )
        self._max_seq = int(getattr(self._model, "max_seq_length", 512) or 512)
        self._dim = int(self._model.get_sentence_embedding_dimension())

    @property
    def embedding_dim(self) -> int:
        return self._dim

    @property
    def max_input_tokens(self) -> int:
        return self._max_seq

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)
        vectors = self._model.encode(
            list(texts),
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        return np.asarray(vectors, dtype=np.float32)


@dataclass
class _OpenAILLM:
    """OpenAI chat.completions wrapper. Also drives Ollama's ``/v1`` when ``base_url`` is set."""

    model: str
    api_key: str
    base_url: str = ""
    _max_input_tokens: int = 8191
    tokenizer_id: str = ""
    _tokenizer: Optional[Any] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        import openai

        # Ollama ignores the api key but the SDK rejects an empty string;
        # supply a placeholder when only base_url is configured.
        effective_key = self.api_key or "ollama-local"
        if not self.base_url and not self.api_key:
            raise ValueError(
                "OpenAI backend selected but no api_key provided "
                "(set [NLSeeker].openai_api_key or OPENAI_API_KEY env)."
            )
        kwargs = {"api_key": effective_key}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        self._client = openai.OpenAI(**kwargs)

    @property
    def max_input_tokens(self) -> int:
        return self._max_input_tokens

    @property
    def tokenizer(self):
        if self._tokenizer is None and self.tokenizer_id:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_id)
        return self._tokenizer

    def _truncate_prompt(self, prompt: str, max_new_tokens: int) -> str:
        tok = self.tokenizer
        if tok is None:
            return prompt

        # Target rendered length, leaving a small safety headroom for any
        # template-rendering quirks the empty-user baseline cannot capture.
        target_total = self._max_input_tokens - max_new_tokens - 8

        def _render_len(text: str) -> int:
            try:
                ids = tok.apply_chat_template(
                    [{"role": "user", "content": text}],
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=False,
                )
                # Newer transformers can still hand back a BatchEncoding when
                # ``return_dict=False`` is silently ignored; pull the ids out.
                if hasattr(ids, "keys") and "input_ids" in ids:
                    ids = ids["input_ids"]
                return len(ids)
            except Exception:
                # Fallback: assume a 16-token chat-template overhead.
                return len(tok.tokenize(text)) + 16

        if _render_len(prompt) <= target_total:
            return prompt

        try:
            base_obj = tok.apply_chat_template(
                [{"role": "user", "content": ""}],
                tokenize=True,
                add_generation_prompt=True,
                return_dict=False,
            )
            if hasattr(base_obj, "keys") and "input_ids" in base_obj:
                base_obj = base_obj["input_ids"]
            base_len = len(base_obj)
        except Exception:
            base_len = 16
        tokens = tok.tokenize(prompt)
        budget = max(1, target_total - base_len)
        truncated = tok.convert_tokens_to_string(tokens[:budget])

        # Verification loop: re-render and shave 64 tokens at a time if the
        # round-trip inflated the count. Caps at a few iterations so a
        # pathological tokenizer cannot loop forever.
        for _ in range(8):
            if _render_len(truncated) <= target_total:
                return truncated
            budget = max(1, budget - 64)
            truncated = tok.convert_tokens_to_string(tokens[:budget])
        return truncated

    def generate(
        self,
        prompts: Sequence[str],
        max_new_tokens: int,
        *,
        concurrency: int | None = None,
    ) -> list:
        """Issue chat-completion prompts to the OpenAI-compat backend.

        concurrency=None (default) uses ThreadPoolExecutor(_OPENAI_LLM_CONCURRENCY).
        concurrency=1 issues prompts sequentially (no executor) - used by
        benchmark mode for PNEUMA-shape parity.
        concurrency>=2 caps the executor at that width.
        """
        prompt_list = list(prompts)
        n = len(prompt_list)
        if n == 0:
            return []

        # Warm up the lazy tokenizer property once before fanning out so
        # concurrent _truncate_prompt callers don't race on the
        # AutoTokenizer.from_pretrained() in self.tokenizer.
        if self.tokenizer_id:
            _ = self.tokenizer

        def _one(prompt: str) -> str:
            safe_prompt = self._truncate_prompt(prompt, max_new_tokens)
            kwargs = {
                "model": self.model,
                "messages": [{"role": "user", "content": safe_prompt}],
                "max_tokens": max_new_tokens,
                "temperature": 0.0,
            }
            # D16: send seed=42 unless we're talking to Ollama (which does not honour it).
            # vLLM honours seed= over OpenAI-compat; OpenAI-proper does too.
            if not _is_ollama_endpoint(self.base_url):
                kwargs["seed"] = 42
            resp = self._client.chat.completions.create(**kwargs)
            return (resp.choices[0].message.content or "").strip()

        if concurrency == 1:
            return [_one(p) for p in prompt_list]

        # Single-prompt or global-cap=1: serial dispatch keeps the call
        # structure identical to the simple loop (no executor overhead).
        if n == 1 or _OPENAI_LLM_CONCURRENCY <= 1:
            return [_one(p) for p in prompt_list]

        # Default concurrent path: ThreadPoolExecutor for batched OpenAI-compat
        # dispatch. concurrency=None falls back to the global
        # _OPENAI_LLM_CONCURRENCY ceiling; concurrency >= 2 uses that exact
        # value as the cap.
        cap = _OPENAI_LLM_CONCURRENCY if concurrency is None else concurrency
        workers = min(cap, n)
        out: list = [None] * n
        with ThreadPoolExecutor(max_workers=workers) as pool:
            # executor.map preserves submission order in the returned
            # iterator, so the i-th completion ends up in out[i] without
            # any explicit index threading.
            for i, completion in enumerate(pool.map(_one, prompt_list)):
                out[i] = completion
        return out


@dataclass
class _OpenAIEmbedder:
    model: str
    api_key: str
    base_url: str = ""
    batch_size: int = 256
    _dim: Optional[int] = None
    _max_input_tokens: int = 8191
    tokenizer_id: str = ""
    _tokenizer: Optional[Any] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        import openai

        effective_key = self.api_key or "ollama-local"
        kwargs = {"api_key": effective_key}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        self._client = openai.OpenAI(**kwargs)

    @property
    def embedding_dim(self) -> int:
        if self._dim is None:
            self._dim = len(self._client.embeddings.create(model=self.model, input=["ping"]).data[0].embedding)
        return self._dim

    @property
    def max_input_tokens(self) -> int:
        return self._max_input_tokens

    @property
    def tokenizer(self):
        if self._tokenizer is None and self.tokenizer_id:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_id)
        return self._tokenizer

    def _truncate(self, texts: Sequence[str]) -> list:
        tok = self.tokenizer
        if tok is None:
            return list(texts)

        budget = max(1, self._max_input_tokens - 8)
        out = []
        for t in texts:
            ids = tok.encode(t, add_special_tokens=False)
            if len(ids) <= budget:
                out.append(t)
            else:
                out.append(tok.decode(ids[:budget], skip_special_tokens=True))
        return out

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)
        safe_texts = self._truncate(texts)
        vectors = []
        for i in range(0, len(safe_texts), self.batch_size):
            chunk = list(safe_texts[i : i + self.batch_size])
            resp = self._client.embeddings.create(model=self.model, input=chunk)
            vectors.extend(item.embedding for item in resp.data)
        if self._dim is None:
            self._dim = len(vectors[0]) if vectors else 0
        return np.asarray(vectors, dtype=np.float32)


def build_backends(cfg: NLSeekerConfig) -> tuple:
    """Return an ``(llm, embedder)`` pair, cached per config signature."""
    key = cfg.signature()
    with _CACHE_LOCK:
        cached = _BACKEND_CACHE.get(key)
        if cached is not None:
            return cached

        if cfg.use_local_model:
            llm = _LocalLLM(
                model_path=cfg.llm_path,
                max_batch=cfg.max_llm_batch_size,
                hf_token=cfg.hf_token,
            )
            embedder = _LocalEmbedder(
                embed_path=cfg.embed_path,
                hf_token=cfg.hf_token,
            )
        else:
            llm = _OpenAILLM(
                model=cfg.openai_llm_model,
                api_key=cfg.openai_api_key,
                base_url=cfg.openai_base_url,
                _max_input_tokens=cfg.openai_llm_max_input_tokens,
                tokenizer_id=cfg.openai_llm_tokenizer_id,
            )
            if cfg.local_embedder:
                embedder = _LocalEmbedder(
                    embed_path=cfg.embed_path,
                    hf_token=cfg.hf_token,
                )
            else:
                embedder = _OpenAIEmbedder(
                    model=cfg.openai_embed_model,
                    api_key=cfg.openai_api_key,
                    base_url=cfg.openai_embed_base_url or cfg.openai_base_url,
                    _max_input_tokens=cfg.openai_embed_max_input_tokens,
                    tokenizer_id=cfg.openai_embed_tokenizer_id,
                )
        _BACKEND_CACHE[key] = (llm, embedder)
        return llm, embedder


def reset_backend_cache() -> None:
    """Clear the cache. Used by tests."""
    with _CACHE_LOCK:
        _BACKEND_CACHE.clear()
        gc.collect()


__all__ = ["LLMBackend", "EmbedBackend", "build_backends", "reset_backend_cache"]
