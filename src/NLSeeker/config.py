import configparser
import hashlib
import os
from dataclasses import dataclass, field, fields, replace
from pathlib import Path

# Typing imports
from typing import Any, Mapping


_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_CONFIG_PATH = _PROJECT_ROOT / "config" / "config.ini"


@dataclass(frozen=True)
class NLSeekerConfig:
    """Configuration for NLSeeker, loaded from the [NLSeeker] section of config.ini."""

    out_path: Path = field(default=_PROJECT_ROOT / "nl-out")
    index_name: str = "blend_nl_index"
    use_local_model: bool = True
    # Mixed mode: when ``use_local_model`` is False (LLM goes through OpenAI/
    # vLLM), ``local_embedder=True`` still loads the embedder via
    # sentence-transformers locally. No effect when use_local_model is True.
    local_embedder: bool = False
    llm_path: str = "Qwen/Qwen2.5-7B-Instruct"
    embed_path: str = "BAAI/bge-base-en-v1.5"
    hf_token: str = ""
    openai_api_key: str = ""
    openai_base_url: str = ""
    openai_embed_base_url: str = ""
    openai_llm_model: str = "gpt-4o-mini"
    openai_llm_max_input_tokens: int = 8191
    openai_llm_tokenizer_id: str = ""
    openai_embed_model: str = "text-embedding-3-small"
    openai_embed_max_input_tokens: int = 8191
    openai_embed_tokenizer_id: str = ""
    alpha: float = 0.5
    n: int = 5
    default_k: int = 10
    max_llm_batch_size: int = 50

    @classmethod
    def load(cls, config_path: Path = None, overrides: Mapping[str, Any] = None) -> "NLSeekerConfig":
        """Load defaults, then layer values from the ini file and ``overrides``."""
        cfg = cls()
        path = Path(config_path) if config_path is not None else _DEFAULT_CONFIG_PATH

        if path.exists():
            parser = configparser.ConfigParser()
            parser.read(path)
            if parser.has_section("NLSeeker"):
                cfg = cfg._merge(dict(parser.items("NLSeeker")))

        if overrides:
            cfg = cfg._merge(overrides)

        if not cfg.out_path.is_absolute():
            cfg = replace(cfg, out_path=(_PROJECT_ROOT / cfg.out_path).resolve())

        if not cfg.use_local_model and not cfg.openai_api_key:
            env_key = os.environ.get("OPENAI_API_KEY", "")
            if env_key:
                cfg = replace(cfg, openai_api_key=env_key)

        return cfg

    def vector_index_path(self) -> Path:
        return self.out_path / "indexes" / "vector" / self.index_name

    def fulltext_index_path(self) -> Path:
        return self.out_path / "indexes" / "fulltext" / self.index_name

    def signature(self) -> str:
        """Stable hash over fields that determine engine identity. Secrets excluded."""
        material = "|".join(
            f"{name}={getattr(self, name)!r}"
            for name in (
                "out_path",
                "index_name",
                "use_local_model",
                "local_embedder",
                "llm_path",
                "embed_path",
                "openai_base_url",
                "openai_embed_base_url",
                "openai_llm_model",
                "openai_llm_max_input_tokens",
                "openai_llm_tokenizer_id",
                "openai_embed_model",
                "openai_embed_max_input_tokens",
                "openai_embed_tokenizer_id",
                "alpha",
                "n",
                "default_k",
                "max_llm_batch_size",
            )
        )
        return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]

    def _merge(self, raw: Mapping[str, Any]) -> "NLSeekerConfig":
        kwargs = {}
        valid = {f.name: f.type for f in fields(self)}
        for key, value in raw.items():
            name = key.strip()
            if name not in valid:
                continue
            target = valid[name]
            if target is bool or target == "bool":
                if isinstance(value, str):
                    kwargs[name] = value.strip().lower() in {"1", "true", "yes", "on"}
                else:
                    kwargs[name] = bool(value)
            elif target is int or target == "int":
                kwargs[name] = int(value)
            elif target is float or target == "float":
                kwargs[name] = float(value)
            elif target is Path or target == "Path":
                kwargs[name] = Path(str(value)).expanduser()
            else:
                kwargs[name] = str(value)
        return replace(self, **kwargs)


__all__ = ["NLSeekerConfig"]
