from __future__ import annotations

import json
from urllib.request import Request, urlopen


class EndpointError(RuntimeError):
    """vLLM endpoint failed health/model probe."""


def _get(url: str, *, timeout: float = 5.0) -> dict:
    req = Request(url, headers={"Accept": "application/json"})
    with urlopen(req, timeout=timeout) as resp:
        if resp.status != 200:
            raise EndpointError(f"{url}: HTTP {resp.status}")
        body = resp.read()
        if not body:
            return {}
        return json.loads(body)


def _probe_one(label: str, base_url: str, expected_model: str) -> dict:
    base = base_url.rstrip("/")
    # /health may live one level above /v1
    health_url = base.replace("/v1", "") + "/health"
    try:
        _get(health_url)
    except Exception as e:
        raise EndpointError(
            f"{label}: /health unreachable at {health_url} ({e}). "
            f"Run: curl -fsS {health_url}"
        ) from e
    models = _get(base + "/models")
    ids = [item.get("id") for item in models.get("data", [])]
    if expected_model not in ids:
        raise EndpointError(
            f"{label}: expected served-model-name {expected_model!r} on "
            f"{base_url}, observed {ids!r}"
        )
    return {"base_url": base_url, "model_id": expected_model, "models_seen": ids}


def probe_endpoints(
    *,
    llm_urls: "list[str] | str",
    llm_model: str,
    embed_url: "str | None" = None,
    embed_model: "str | None" = None,
) -> dict:
    if isinstance(llm_urls, str):
        urls = [llm_urls]
    else:
        urls = list(llm_urls)
    result: dict = {
        "llm": [_probe_one(f"llm[{i}]", url, llm_model) for i, url in enumerate(urls)],
    }
    if embed_url and embed_model:
        result["embed"] = _probe_one("embed", embed_url, embed_model)
    return result


__all__ = ["probe_endpoints", "EndpointError"]
