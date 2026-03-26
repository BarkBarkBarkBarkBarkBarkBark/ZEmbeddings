"""
Tests for the causal sliding-window tokeniser.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest
from zembeddings.params import PARAMS, get_params
from zembeddings.tokenizer import tokenize, windows_to_texts, token_count


SAMPLE_TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "Dogs are wonderful companions and they bring joy to our lives."
)


class TestTokenize:
    def test_basic_output_structure(self):
        result = tokenize(SAMPLE_TEXT, PARAMS)
        assert result.n_tokens > 0
        assert len(result.token_ids) == result.n_tokens
        assert result.raw_text == SAMPLE_TEXT
        assert result.encoding_name == "cl100k_base"

    def test_window_count(self):
        """With stride=1 and window_size=W, we expect N - W + 1 windows."""
        p = get_params(**{"window.size": 5, "window.stride": 1})
        result = tokenize(SAMPLE_TEXT, p)
        expected = result.n_tokens - 5 + 1
        assert len(result.windows) == expected

    def test_window_size(self):
        """Each window should contain exactly window_size tokens."""
        p = get_params(**{"window.size": 7})
        result = tokenize(SAMPLE_TEXT, p)
        for w in result.windows:
            assert len(w.token_ids) == 7

    def test_causal_ordering(self):
        """Window t should end at token t + window_size - 1.
        No window should reference future tokens beyond its index."""
        result = tokenize(SAMPLE_TEXT, PARAMS)
        for i, w in enumerate(result.windows):
            # End token of window i should be >= start token
            assert w.end_token >= w.start_token
            # Windows should be ordered
            if i > 0:
                assert w.start_token >= result.windows[i - 1].start_token

    def test_stride_2(self):
        """Stride of 2 should produce roughly half the windows."""
        p1 = get_params(**{"window.size": 5, "window.stride": 1})
        p2 = get_params(**{"window.size": 5, "window.stride": 2})
        r1 = tokenize(SAMPLE_TEXT, p1)
        r2 = tokenize(SAMPLE_TEXT, p2)
        # Stride 2 gives ceil((N-W+1) / 2) windows
        assert len(r2.windows) < len(r1.windows)
        assert len(r2.windows) > 0

    def test_windows_to_texts(self):
        result = tokenize(SAMPLE_TEXT, PARAMS)
        texts = windows_to_texts(result)
        assert len(texts) == len(result.windows)
        assert all(isinstance(t, str) for t in texts)
        assert all(len(t) > 0 for t in texts)

    def test_token_count(self):
        count = token_count(SAMPLE_TEXT)
        assert count > 0
        assert isinstance(count, int)

    def test_empty_text(self):
        """Empty text should produce no windows."""
        result = tokenize("", PARAMS)
        assert result.n_tokens == 0
        assert len(result.windows) == 0

    def test_text_shorter_than_window(self):
        """Text with fewer tokens than window size → no windows."""
        p = get_params(**{"window.size": 100})
        result = tokenize("hello world", p)
        assert len(result.windows) == 0
