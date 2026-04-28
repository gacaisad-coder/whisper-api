from __future__ import annotations

import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from app import transcribe_mac as tm


class QwenModelDetectionTests(unittest.TestCase):
    def test_detects_qwen3_asr_repo_ids(self):
        self.assertTrue(tm._is_qwen3_asr_model("mlx-community/Qwen3-ASR-1.7B-4bit"))
        self.assertTrue(tm._is_qwen3_asr_model("Qwen/Qwen3-ASR-1.7B"))
        self.assertFalse(tm._is_qwen3_asr_model("small"))


class QwenTranscribeWrapperTests(unittest.TestCase):
    def test_maps_qwen_result_to_existing_segment_schema(self):
        fake_utils = types.ModuleType("mlx_audio.stt.utils")
        fake_generate = types.ModuleType("mlx_audio.stt.generate")
        fake_utils.load_model = lambda _name: object()

        class FakeResult:
            text = "こんにちは"
            language = "ja"
            time_stamps = [{"start_time": 0.0, "end_time": 1.2, "text": "こんにちは"}]

        fake_generate.generate_transcription = lambda **_kwargs: FakeResult()

        with patch.dict(
            sys.modules,
            {
                "mlx_audio": types.ModuleType("mlx_audio"),
                "mlx_audio.stt": types.ModuleType("mlx_audio.stt"),
                "mlx_audio.stt.utils": fake_utils,
                "mlx_audio.stt.generate": fake_generate,
            },
            clear=False,
        ):
            text, language, duration, segments = tm._transcribe_with_qwen3_mlx_audio(
                temp_path="/tmp/a.wav",
                model_name="mlx-community/Qwen3-ASR-1.7B-4bit",
                language="ja",
            )

        self.assertEqual(text, "こんにちは")
        self.assertEqual(language, "ja")
        self.assertEqual(duration, 1.2)
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0].text, "こんにちは")

    def test_missing_mlx_audio_dependency_raises_actionable_error(self):
        with patch.dict(sys.modules, {"mlx_audio.stt.generate": None, "mlx_audio.stt.utils": None}, clear=False):
            with self.assertRaises(RuntimeError) as ctx:
                tm._transcribe_with_qwen3_mlx_audio(
                    temp_path="/tmp/a.wav",
                    model_name="mlx-community/Qwen3-ASR-1.7B-4bit",
                    language="ja",
                )
        self.assertIn("pip install -U mlx-audio", str(ctx.exception))

    def test_retries_with_prepared_wav_when_audio_decode_error_occurs(self):
        fake_utils = types.ModuleType("mlx_audio.stt.utils")
        fake_generate = types.ModuleType("mlx_audio.stt.generate")
        fake_utils.load_model = lambda _name: object()

        class FakeResult:
            text = "ok"
            language = "ja"
            time_stamps = []

        calls: list[str] = []

        def _fake_generate(**kwargs):
            calls.append(str(kwargs.get("audio_path", "")))
            if len(calls) == 1:
                raise AttributeError("'NoneType' object has no attribute 'ndim'")
            return FakeResult()

        fake_generate.generate_transcription = _fake_generate

        with patch.dict(
            sys.modules,
            {
                "mlx_audio": types.ModuleType("mlx_audio"),
                "mlx_audio.stt": types.ModuleType("mlx_audio.stt"),
                "mlx_audio.stt.utils": fake_utils,
                "mlx_audio.stt.generate": fake_generate,
            },
            clear=False,
        ):
            with patch.object(tm, "_prepare_qwen3_audio", return_value="/tmp/prepared.wav") as prepare_audio:
                text, language, _, _ = tm._transcribe_with_qwen3_mlx_audio(
                    temp_path="/tmp/input.mp3",
                    model_name="mlx-community/Qwen3-ASR-1.7B-4bit",
                    language="ja",
                )

        self.assertEqual(text, "ok")
        self.assertEqual(language, "ja")
        prepare_audio.assert_called_once_with("/tmp/input.mp3")
        self.assertEqual(calls, ["/tmp/input.mp3", "/tmp/prepared.wav"])


class QwenRoutingTests(unittest.TestCase):
    def test_qwen_failure_does_not_fallback_to_cpu_whisper(self):
        with tempfile.NamedTemporaryFile(suffix=".wav") as fp:
            data = fp.read() if False else b"data"
            with patch.object(tm, "_is_qwen3_asr_model", return_value=True):
                with patch.object(tm, "_transcribe_with_qwen3_mlx_audio", side_effect=RuntimeError("boom")):
                    with patch.object(tm, "_transcribe_with_cpu_whisper") as cpu_path:
                        with self.assertRaises(RuntimeError):
                            tm.transcribe_audio(
                                audio_bytes=data,
                                model_name="mlx-community/Qwen3-ASR-1.7B-4bit",
                                language="ja",
                                source_filename="a.wav",
                            )
                        cpu_path.assert_not_called()


if __name__ == "__main__":
    unittest.main()
