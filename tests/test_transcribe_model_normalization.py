import sys
import types
import unittest


fake_faster_whisper = types.ModuleType("faster_whisper")


class DummyWhisperModel:
    pass


fake_faster_whisper.WhisperModel = DummyWhisperModel
sys.modules.setdefault("faster_whisper", fake_faster_whisper)

from app import transcribe


class ModelNormalizationTests(unittest.TestCase):
    def test_normalize_model_name_maps_sensevoice_repo_id_to_small(self) -> None:
        self.assertEqual(
            transcribe._normalize_model_name("FunAudioLLM/SenseVoiceSmall"),
            "small",
        )


if __name__ == "__main__":
    unittest.main()
