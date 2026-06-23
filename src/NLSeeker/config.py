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


    enabled_summary_types: frozenset = field(
        default_factory=lambda: frozenset({"COLUMN_NARRATION", "ROW_SAMPLE"})
    )
    context_sources: frozenset = field(default_factory=lambda: frozenset({"real"}))

    generate_content_profile: bool = True
    generate_semantic_profile: bool = True
    generate_topic: bool = True
    generate_ufd: bool = True
    generate_sfd: bool = True
    generate_narration_profiled: bool = True

    topic_into_ufd_sfd: bool = True
    index_topic_standalone: bool = False
    semantic_profile_group_size: int = 0
    force_regenerate: bool = False

    def __post_init__(self) -> None:
        # Lazy import to avoid db_schema <-> config cycle.
        from src.NLSeeker.db_schema import SummaryType

        coerced = frozenset(
            v if isinstance(v, SummaryType) else SummaryType(v)
            for v in self.enabled_summary_types
        )
        # frozen dataclass: route through object.__setattr__.
        object.__setattr__(self, "enabled_summary_types", coerced)
        object.__setattr__(self, "context_sources", frozenset(self.context_sources))

    @property
    def enabled_summary_types_sig(self) -> tuple:
        """Stable sorted tuple of enum values for use in signature(); frozenset repr is order-unstable."""
        return tuple(sorted(st.value for st in self.enabled_summary_types))

    @property
    def context_sources_sig(self) -> tuple:
        """Stable sorted tuple of source strings for use in signature(); frozenset repr is order-unstable."""
        return tuple(sorted(self.context_sources))

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
                # AutoDDG identity-affecting fields.
                # NOTE: frozenset repr is order-unstable across Python builds;
                # we sort to stabilize the signature.
                "enabled_summary_types_sig",
                "context_sources_sig",
                "generate_content_profile",
                "generate_semantic_profile",
                "generate_topic",
                "generate_ufd",
                "generate_sfd",
                "generate_narration_profiled",
                "topic_into_ufd_sfd",
                "index_topic_standalone",
                "semantic_profile_group_size",
                # force_regenerate is deliberately omitted - it's a transient flag,
                # not part of engine identity.
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
            elif name in {"enabled_summary_types", "context_sources"}:
                # Accept comma-separated strings (ini) or iterables (overrides dict).
                if isinstance(value, str):
                    items = {v.strip() for v in value.split(",") if v.strip()}
                else:
                    items = set(value)
                kwargs[name] = frozenset(items)
            else:
                kwargs[name] = str(value)
        return replace(self, **kwargs)


__all__ = ["NLSeekerConfig"]
