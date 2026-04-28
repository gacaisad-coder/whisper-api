import os
import sys
import types
import unittest
from unittest.mock import patch


fake_faster_whisper = types.ModuleType("faster_whisper")


class DummyWhisperModel:
    pass


fake_faster_whisper.WhisperModel = DummyWhisperModel
sys.modules.setdefault("faster_whisper", fake_faster_whisper)

from app import transcribe_mac as tm


class SenseVoiceUtilsTests(unittest.TestCase):
    def test_parse_timestamps_supports_words_fallback_and_millisecond_normalization(self) -> None:
        item = {
            "timestamp": [[1200, 2500], ["!", 2600, 3900]],
            "words": ["こんにちは", "unused"],
        }

        parsed = tm._sensevoice_parse_timestamps(item)

        self.assertEqual(
            parsed,
            [
                ("こんにちは", 1.2, 2.5),
                ("!", 2.6, 3.9),
            ],
        )

    def test_join_tokens_keeps_cjk_compact_and_attaches_punctuation(self) -> None:
        tokens = ["私", "は", "AI", "です", "。", "(" , "テスト", ")"]

        text = tm._sensevoice_join_tokens(tokens)

        self.assertEqual(text, "私は AI です。(テスト)")

    def test_sentence_segments_are_monotonic_when_source_timestamps_go_backward(self) -> None:
        item = {
            "sentence_info": [
                {
                    "text": "first",
                    "timestamp": [["first", 2.0, 3.0]],
                },
                {
                    "text": "second",
                    "timestamp": [["second", 1.0, 1.5]],
                },
            ]
        }

        segments = tm._sensevoice_parse_sentence_segments(item)

        self.assertEqual(len(segments), 2)
        self.assertGreaterEqual(segments[1].start, segments[0].end)
        self.assertGreaterEqual(segments[1].end, segments[1].start)

    def test_env_parsers_and_runtime_generate_kwargs(self) -> None:
        with patch.dict(
            os.environ,
            {
                "SENSEVOICE_USE_ITN": "false",
                "SENSEVOICE_BATCH_SIZE_S": "90",
                "SENSEVOICE_MERGE_VAD": "true",
                "SENSEVOICE_MERGE_LENGTH_S": "22",
            },
            clear=False,
        ):
            kwargs = tm._sensevoice_generate_kwargs(language="ja")

        self.assertEqual(kwargs["language"], "ja")
        self.assertEqual(kwargs["use_itn"], False)
        self.assertEqual(kwargs["batch_size_s"], 90)
        self.assertEqual(kwargs["merge_vad"], True)
        self.assertEqual(kwargs["merge_length_s"], 22)
        self.assertEqual(kwargs["output_timestamp"], True)


if __name__ == "__main__":
    unittest.main()
