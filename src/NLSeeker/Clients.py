from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI, OpenAIError
from tokenizers import Tokenizer
from tqdm import tqdm

# Typing imports
from src.NLSeeker.Config import NLSeekerConfig
from typing import Dict, List, Optional

# The embedding endpoint's tokenizer adds [CLS] and [SEP] on top of whatever we send.
SPECIAL_TOKEN_OVERHEAD = 2


class _LazyTokenizer:
    """Counts and truncates tokens; fetches tokenizer.json on first use, never model weights."""

    def __init__(self, name: str) -> None:
        self._name = name
        self._tokenizer = None

    def count(self, text: str) -> int:
        return len(self._load().encode(text, add_special_tokens=False).ids)

    def truncate(self, text: str, budget: int) -> str:
        ids = self._load().encode(text, add_special_tokens=False).ids
        if len(ids) <= budget:
            return text
        return self._load().decode(ids[:budget])

    def _load(self) -> Tokenizer:
        if self._tokenizer is None:
            self._tokenizer = Tokenizer.from_pretrained(self._name)
        return self._tokenizer


class LLMClient:
    """Chat completions against an OpenAI-compatible endpoint."""

    def __init__(self, config: NLSeekerConfig) -> None:
        self._client = OpenAI(base_url=config.llm_base_url, api_key=config.llm_api_key)
        self._model = config.llm_model
        self._temperature = config.llm_temperature
        self._max_new_tokens = config.llm_max_new_tokens
        self._context_length = config.llm_context_length
        self._chat_template_overhead = config.llm_chat_template_overhead
        self._concurrency = config.llm_concurrency
        self._retry_attempts = config.llm_retry_attempts
        self._tokenizer = _LazyTokenizer(config.llm_tokenizer)

    def complete(self, conversations: List[List[Dict[str, str]]], max_new_tokens: Optional[int] = None,
                 desc: Optional[str] = None) -> List[str]:
        """Answers every conversation, in input order."""
        if max_new_tokens is None:
            max_new_tokens = self._max_new_tokens

        # The endpoint applies the chat template server-side, so its cost is configured rather
        # than measured with a local apply_chat_template.
        budget = self._context_length - self._chat_template_overhead - max_new_tokens
        conversations = [[{**head, 'content': self._tokenizer.truncate(head['content'], budget)}, *rest]
                         for head, *rest in conversations]

        def answer(conversation: List[Dict[str, str]]) -> str:
            return self._complete_one(conversation, max_new_tokens)

        if self._concurrency <= 1:
            return list(tqdm(map(answer, conversations), total=len(conversations), desc=desc))

        with ThreadPoolExecutor(max_workers=self._concurrency) as executor:
            return list(tqdm(executor.map(answer, conversations), total=len(conversations), desc=desc))

    def _complete_one(self, conversation: List[Dict[str, str]], max_new_tokens: int) -> str:
        for _ in range(max(self._retry_attempts - 1, 0)):
            try:
                return self._request(conversation, max_new_tokens)
            except OpenAIError:
                continue
        return self._request(conversation, max_new_tokens)

    def _request(self, conversation: List[Dict[str, str]], max_new_tokens: int) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=conversation,
            max_tokens=max_new_tokens,
            temperature=self._temperature,
            top_p=1.0,
            seed=42,
        )
        return response.choices[0].message.content


class EmbeddingClient:
    """Embeddings against an OpenAI-compatible endpoint."""

    def __init__(self, config: NLSeekerConfig) -> None:
        self._client = OpenAI(base_url=config.embedding_base_url, api_key=config.embedding_api_key)
        self._model = config.embedding_model
        self._batch_size = config.embedding_batch_size
        self._tokenizer = _LazyTokenizer(config.embedding_tokenizer)
        self.max_tokens = config.embedding_max_tokens

    def encode(self, documents: List[str]) -> List[List[float]]:
        """Embeds every document, in input order."""
        # A single summary block can exceed the window on its own, which a local
        # SentenceTransformer would truncate silently and an endpoint rejects instead.
        budget = self.max_tokens - SPECIAL_TOKEN_OVERHEAD
        documents = [self._tokenizer.truncate(document, budget) for document in documents]

        embeddings = []
        for start in range(0, len(documents), self._batch_size):
            response = self._client.embeddings.create(
                input=documents[start:start + self._batch_size],
                model=self._model,
            )
            embeddings.extend(item.embedding for item in response.data)
        return embeddings

    def count_tokens(self, text: str) -> int:
        return self._tokenizer.count(text)
