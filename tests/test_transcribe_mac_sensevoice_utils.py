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
    def test_prepare_sensevoice_audio_produces_wav_file(self) -> None:
        class FakeArray:
            def __init__(self, values):
                self.values = values

            def __len__(self):
                return len(self.values)

            def __iter__(self):
                return iter(self.values)

            def __truediv__(self, value):
                return FakeArray([v / value for v in self.values])

            def __mul__(self, value):
                return FakeArray([v * value for v in self.values])

            def astype(self, _dtype):
                return self

            def tobytes(self):
                return b"\x00\x00" * len(self.values)

        class FakeNumpy:
            int16 = object()

            @staticmethod
            def max(values):
                return max(abs(v) for v in values.values)

            @staticmethod
            def abs(values):
                return values

            @staticmethod
            def clip(values, *_args):
                return values

        class DummyWaveFile:
            def __init__(self):
                self.closed = False

            def __enter__(self):
                return self

            def __exit__(self, *args):
                self.closed = True

            def setnchannels(self, _v):
                pass

            def setsampwidth(self, _v):
                pass

            def setframerate(self, _v):
                pass

            def writeframes(self, _v):
                pass

        fake_audio = FakeArray([0.2, -0.3, 0.8])
        fake_audio_module = types.ModuleType("faster_whisper.audio")
        fake_audio_module.decode_audio = lambda _path: fake_audio
        with patch.dict(
            sys.modules,
            {
                "numpy": FakeNumpy,
                "faster_whisper.audio": fake_audio_module,
            },
            clear=False,
        ):
            with patch("wave.open", return_value=DummyWaveFile()):
                path = tm._prepare_sensevoice_audio("dummy.wav")

        self.assertTrue(path.endswith(".wav"))
        if os.path.exists(path):
            os.remove(path)

    def test_preprocess_toggle_reads_env(self) -> None:
        with patch.dict(os.environ, {"SENSEVOICE_PREPROCESS_ENABLED": "false"}, clear=True):
            self.assertEqual(tm._sensevoice_preprocess_enabled(), False)

        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(tm._sensevoice_preprocess_enabled(), True)

    def test_transcribe_with_sensevoice_enforces_monotonic_timestamps_across_result_items(self) -> None:
        class DummySenseVoiceModel:
            def generate(self, **kwargs):
                return [
                    {
                        "text": "first",
                        "sentence_info": [
                            {
                                "text": "first",
                                "timestamp": [["first", 2.0, 3.0]],
                            }
                        ],
                    },
                    {
                        "text": "second",
                        "sentence_info": [
                            {
                                "text": "second",
                                "timestamp": [["second", 1.0, 1.5]],
                            }
                        ],
                    },
                ]

        with patch.object(tm, "_get_sensevoice_model", return_value=DummySenseVoiceModel()):
            with patch.object(tm, "_sensevoice_preprocess_enabled", return_value=False):
                with patch.object(
                    tm,
                    "_sensevoice_runtime_config",
                    return_value={
                        "batch_size_s": 20.0,
                    "merge_vad": True,
                    "merge_length_s": 8.0,
                    "use_itn": True,
                },
            ):
                    _, _, _, segments, _ = tm._transcribe_with_sensevoice(
                        temp_path="dummy.wav",
                        model_name="funaudiollm/sensevoicesmall",
                        language="ja",
                        require_gpu=False,
                    )

        self.assertEqual(len(segments), 2)
        self.assertGreaterEqual(segments[1].start, segments[0].end)
        self.assertGreaterEqual(segments[1].end, segments[1].start)

    def test_parse_timestamps_auto_mode_keeps_small_integer_values_as_seconds(self) -> None:
        item = {
            "timestamp": [[0, 120], [120, 360]],
            "words": ["勇気", "手に"],
        }

        with patch.dict(os.environ, {}, clear=True):
            parsed = tm._sensevoice_parse_timestamps(item)

        self.assertEqual(parsed, [("勇気", 0.0, 120.0), ("手に", 120.0, 360.0)])

    def test_parse_timestamps_auto_mode_detects_milliseconds_with_large_values(self) -> None:
        item = {
            "timestamp": [[12000, 12500], ["!", 12600, 13900]],
            "words": ["こんにちは", "unused"],
        }

        with patch.dict(os.environ, {}, clear=True):
            parsed = tm._sensevoice_parse_timestamps(item)

        self.assertEqual(
            parsed,
            [
                ("こんにちは", 12.0, 12.5),
                ("!", 12.6, 13.9),
            ],
        )

    def test_parse_timestamps_explicit_ms_mode_converts_to_seconds(self) -> None:
        item = {
            "timestamp": [[0, 120], [120, 360]],
            "words": ["勇気", "手に"],
        }

        with patch.dict(os.environ, {"SENSEVOICE_TIMESTAMP_UNIT": "ms"}, clear=True):
            parsed = tm._sensevoice_parse_timestamps(item)

        self.assertEqual(parsed, [("勇気", 0.0, 0.12), ("手に", 0.12, 0.36)])

    def test_parse_timestamps_explicit_s_mode_keeps_second_values(self) -> None:
        item = {
            "timestamp": [[1200, 2500], ["!", 2600, 3900]],
            "words": ["こんにちは", "unused"],
        }

        with patch.dict(os.environ, {"SENSEVOICE_TIMESTAMP_UNIT": "s"}, clear=True):
            parsed = tm._sensevoice_parse_timestamps(item)

        self.assertEqual(
            parsed,
            [
                ("こんにちは", 1200.0, 2500.0),
                ("!", 2600.0, 3900.0),
            ],
        )

    def test_join_tokens_keeps_cjk_compact_and_attaches_punctuation(self) -> None:
        tokens = ["私", "は", "AI", "です", "。", "(" , "テスト", ")"]

        text = tm._sensevoice_join_tokens(tokens)

        self.assertEqual(text, "私は AI です。(テスト)")

    def test_join_tokens_matches_spec_example(self) -> None:
        self.assertEqual(tm._sensevoice_join_tokens(["勇気", "を", "手", "に", "。"]), "勇気を手に。")

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

        with patch.dict(os.environ, {}, clear=True):
            segments = tm._sensevoice_parse_sentence_segments(item)

        self.assertEqual(len(segments), 2)
        self.assertGreaterEqual(segments[1].start, segments[0].end)
        self.assertGreaterEqual(segments[1].end, segments[1].start)

    def test_sentence_segments_auto_mode_uses_item_level_ms_evidence_for_short_blocks(self) -> None:
        item = {
            "timestamp": [[12000, 12600], [12600, 13900]],
            "words": ["こんにちは", "世界"],
            "sentence_info": [
                {
                    "text": "こんにちは",
                    "timestamp": [["こんにちは", 120, 360]],
                }
            ],
        }

        with patch.dict(os.environ, {}, clear=True):
            segments = tm._sensevoice_parse_sentence_segments(item)

        self.assertEqual(len(segments), 1)
        self.assertAlmostEqual(segments[0].start, 0.12)
        self.assertAlmostEqual(segments[0].end, 0.36)

    def test_build_segments_are_monotonic_when_tokens_go_backward(self) -> None:
        tokens = [
            ("first", 0.0, 1.0),
            (".", 1.0, 1.2),
            ("second", 0.2, 0.8),
            (".", 0.8, 1.0),
        ]

        segments = tm._sensevoice_build_segments(tokens)

        self.assertEqual(len(segments), 2)
        self.assertGreaterEqual(segments[1].start, segments[0].end)
        self.assertGreaterEqual(segments[1].end, segments[1].start)

    def test_env_parsers_and_runtime_config(self) -> None:
        with patch.dict(
            os.environ,
            {
                "SENSEVOICE_USE_ITN": "false",
                "SENSEVOICE_BATCH_SIZE_S": "90.5",
                "SENSEVOICE_MERGE_VAD": "true",
                "SENSEVOICE_MERGE_LENGTH_S": "22.25",
            },
            clear=True,
        ):
            cfg = tm._sensevoice_runtime_config()

        self.assertEqual(cfg["use_itn"], False)
        self.assertEqual(cfg["batch_size_s"], 90.5)
        self.assertEqual(cfg["merge_vad"], True)
        self.assertEqual(cfg["merge_length_s"], 22.25)

    def test_env_bool_false_value(self) -> None:
        with patch.dict(os.environ, {"SENSEVOICE_USE_ITN": "false"}, clear=True):
            self.assertEqual(tm._env_bool("SENSEVOICE_USE_ITN", True), False)

    def test_runtime_config_defaults_match_accuracy_first_plan(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            cfg = tm._sensevoice_runtime_config()

        self.assertEqual(cfg["batch_size_s"], 20.0)
        self.assertEqual(cfg["merge_vad"], True)
        self.assertEqual(cfg["merge_length_s"], 8.0)
        self.assertEqual(cfg["use_itn"], True)

    def test_runtime_config_invalid_numeric_env_falls_back_to_defaults(self) -> None:
        with patch.dict(
            os.environ,
            {
                "SENSEVOICE_BATCH_SIZE_S": "0",
                "SENSEVOICE_MERGE_LENGTH_S": "-1",
            },
            clear=True,
        ):
            with self.assertLogs(tm.logger, level="WARNING") as logs:
                cfg = tm._sensevoice_runtime_config()

        self.assertEqual(cfg["batch_size_s"], 20.0)
        self.assertEqual(cfg["merge_length_s"], 8.0)
        joined_logs = "\n".join(logs.output)
        self.assertIn("SENSEVOICE_BATCH_SIZE_S", joined_logs)
        self.assertIn("SENSEVOICE_MERGE_LENGTH_S", joined_logs)


if __name__ == "__main__":
    unittest.main()
