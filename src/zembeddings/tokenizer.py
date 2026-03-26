"""
Causal sliding-window tokenisation.
====================================

Encodes a raw text string into token IDs using ``tiktoken``, then
produces strictly causal sliding windows:

    window_t = tokens[t - W + 1 : t + 1]

where W = ``params["window"]["size"]`` and the stride defaults to 1.

No future tokens are ever visible — biological plausibility is
maintained by construction.

Reference
---------
- tiktoken: https://github.com/openai/tiktoken
- OpenAI token counting cookbook:
  https://cookbook.openai.com/examples/how_to_count_tokens_with_tiktoken
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import tiktoken


@dataclass
class Window:
    """A single causal window of tokens."""
    index: int            # 0-based position of the window's right edge
    token_ids: list[int]  # the raw token IDs in this window
    text: str             # decoded text for this window
    start_token: int      # index of the first token (inclusive)
    end_token: int        # index of the last token (inclusive)


@dataclass
class TokenizedTranscript:
    """Full tokenisation result for a transcript."""
    raw_text: str
    encoding_name: str
    token_ids: list[int]
    n_tokens: int
    windows: list[Window] = field(default_factory=list)


def tokenize(text: str, params: dict[str, Any]) -> TokenizedTranscript:
    """Tokenise *text* and generate causal sliding windows.

    Parameters
    ----------
    text : str
        Raw transcript text.
    params : dict
        The full experiment PARAMS dict (or a deep copy thereof).

    Returns
    -------
    TokenizedTranscript
        Contains the full token list and the list of ``Window`` objects.
    """
    encoding_name: str = params["window"]["encoding"]
    window_size: int = params["window"]["size"]
    stride: int = params["window"]["stride"]

    enc = tiktoken.get_encoding(encoding_name)
    token_ids: list[int] = enc.encode(text)
    n_tokens = len(token_ids)

    windows: list[Window] = []

    # Build causal windows:  the earliest valid window needs at least
    # ``window_size`` tokens to have accumulated.
    for right_edge in range(window_size - 1, n_tokens, stride):
        left_edge = right_edge - window_size + 1
        win_ids = token_ids[left_edge : right_edge + 1]
        win_text = enc.decode(win_ids)
        windows.append(
            Window(
                index=len(windows),
                token_ids=win_ids,
                text=win_text,
                start_token=left_edge,
                end_token=right_edge,
            )
        )

    return TokenizedTranscript(
        raw_text=text,
        encoding_name=encoding_name,
        token_ids=token_ids,
        n_tokens=n_tokens,
        windows=windows,
    )


def windows_to_texts(transcript: TokenizedTranscript) -> list[str]:
    """Extract the decoded text for each window (for embedding)."""
    return [w.text for w in transcript.windows]


def token_count(text: str, encoding_name: str = "cl100k_base") -> int:
    """Quick token count utility."""
    enc = tiktoken.get_encoding(encoding_name)
    return len(enc.encode(text))
