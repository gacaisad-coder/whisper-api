# Whisper API (OpenAI-Compatible)

此專案提供一個與 OpenAI/Groq 音訊轉錄格式接近的 FastAPI 服務，端點為：

- `POST /v1/audio/transcriptions`

## 安裝

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 打包/部署前必要 include 檢查

此專案的必要 runtime 相依已包含在 `requirements.txt`（`faster-whisper`、`openvino`、`openvino_tokenizers`、`python-multipart` 等）。建議在打包前先做一次 import smoke test：

```bash
python -c "import fastapi, faster_whisper, openvino, openvino_tokenizers; print('imports_ok')"
```

## 執行

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## API 呼叫方式（SenseVoiceSmall）

服務端預設模型已改為 `FunAudioLLM/SenseVoiceSmall`，因此可以不帶 `model` 參數直接呼叫：

> 在 macOS ARM64（`transcribe_mac.py`）上，`FunAudioLLM/SenseVoiceSmall` 會優先走 FunASR 推理（MPS/CPU）。

```bash
curl http://127.0.0.1:8000/v1/audio/transcriptions \
  -X POST \
  -F "file=@/path/to/audio.wav" \
  -F "response_format=json"
```

若你想明確指定模型（或覆寫成其他模型），可帶 `model`：

```bash
curl http://127.0.0.1:8000/v1/audio/transcriptions \
  -X POST \
  -F "file=@/path/to/audio.wav" \
  -F "model=FunAudioLLM/SenseVoiceSmall" \
  -F "response_format=json"
```

若你是打包成 exe，現在可直接用：

```bash
./whisper-api --check-gpu
```

此模式只會做 GPU 探測並立即結束，不會啟動 API 服務。回傳範例：

```json
{"gpu_supported": false, "reason": "openvino_gpu_missing:['CPU']"}
```

## Windows 打包（Workflow 優先）

如果你在 Windows 上遇到「Python 可以啟動，但 exe 的 `--check-gpu` 回傳 `openvino_gpu_missing:[]`」，通常是打包時少帶了 OpenVINO/faster-whisper 的 runtime 檔。

建議優先使用 GitHub Actions workflow：`.github/workflows/build-windows-exe.yml`。
目前 workflow 的 PyInstaller 已包含 `openvino`、`openvino_tokenizers`、`openvino-genai` 相關打包設定，避免執行檔缺少 tokenizer extension entrypoint。

Linux amd64 版本可使用 workflow：`.github/workflows/build-linux-amd64.yml`（手動觸發或推送 `v*` tag）。artifact 會包含可執行檔與 `whisper-api-server-linux-amd64.tar.gz`。

macOS ARM64 版本可使用 workflow：`.github/workflows/build-macos-arm64.yml`（手動觸發或推送 `v*` tag）。artifact 會包含可執行檔與 `whisper-api-server-macos-arm64.tar.gz`。

1. 到 GitHub Actions 手動觸發 **Build Windows EXE**（`workflow_dispatch`）。
2. 下載 artifact `whisper-api-server-windows-exe`。
3. 在 Windows 執行：

```powershell
.\whisper-api-server.exe --check-gpu
```

若你要在本機手動打包，也可用：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\build_windows_exe.ps1
.\dist\whisper-api-server.exe --check-gpu
```

建議判斷：
- 若輸出包含 `openvino_gpu_available`：代表 exe 可見 GPU。
- 若仍是 `openvino_gpu_missing:[]`：優先檢查 Intel 顯示驅動與執行環境是否真的可見 GPU（例如遠端桌面/虛擬機可能隱藏裝置）。

也可先在同機器 Python 環境驗證：

```powershell
python -c "from openvino import Core; print(Core().available_devices)"
```

## 裝置選擇與回退策略

`app/transcribe.py` 會讀取 `WHISPER_DEVICE`：

- `auto`（預設）：偵測 OpenVINO + GPU，若不可用自動退回 CPU。
- `intel_gpu`：在 Linux/Windows 上強制嘗試 Intel GPU 路徑；失敗時退回 CPU（穩定優先）。
- `apple_gpu`：在 macOS ARM64 上強制嘗試 Apple 整合 GPU（MLX）；失敗時退回 CPU。
- `cpu`：直接走 CPU。

另外，模型建立時會**顯式**傳入 `device` 與 `compute_type`，不依賴預設值。

### 模型名稱相容性（Groq/OpenAI client）

此服務會優先嘗試用 `openvino-genai` 載入 OpenVINO repo 型號（例如 `OpenVINO/whisper-large-v3-int8-ov`，或帶 `.../revision/main` 的變體）；實作會先用 `huggingface_hub.snapshot_download(...)` 把模型下載到本機，再交給 `WhisperPipeline`，符合 OpenVINO 模型卡建議流程。若該路徑失敗，才會回退到 `faster-whisper` 的別名型號（例如 `large-v3`），以避免 `model.bin` 缺失錯誤。

> 參考：`OpenVINO/whisper-large-v3-int8-ov` 模型卡的 GenAI 範例是先 `snapshot_download`，再 `WhisperPipeline(model_path, device)`。

可用環境變數：
- `WHISPER_OPENVINO_CACHE_DIR`：自訂 OpenVINO 模型下載目錄（預設 `~/.cache/whisper-api`）。
- `HF_TOKEN`（或 `HUGGINGFACE_HUB_TOKEN`）：提高 Hugging Face 下載速率上限，減少 unauthenticated warning。

## OpenVINO 相容矩陣與環境限制

| 元件 | 版本建議 | 說明 |
|---|---|---|
| Python | 3.10–3.12 (64-bit) | 建議使用 64-bit 環境 |
| faster-whisper | `>=1.1.0,<2.0.0` | Whisper 推理 |
| openvino | `>=2024.3.0,<2027.0.0`（非 macOS ARM64） | OpenVINO Runtime 偵測與 Intel GPU 可見性判斷（Linux/Windows） |
| openvino_tokenizers | `>=2024.3.0,<2027.0.0`（非 macOS ARM64） | OpenVINO tokenizer extension；缺少或版本不對齊時可能出現 `Cannot add extension` |
| openvino-genai | `>=2024.3.0,<2027.0.0`（非 macOS ARM64） | OpenVINO WhisperPipeline 推理 |
| Intel GPU Driver | 最新穩定版 | 需可被 OpenVINO `Core().available_devices` 探測到 `GPU` |
| OS/平台 | Linux / Windows x86_64、macOS ARM64 | 非支援平台或缺驅動時通常會退回 CPU |

> 若 OpenVINO 可 import 但看不到 `GPU`，服務會回報原因並改用 CPU。

若你在 log 看到類似錯誤：

`Cannot add extension. Cannot find entry point to the extension library`

通常是 `openvino` / `openvino_tokenizers` 版本未對齊，建議重裝（在同一個 venv）：

```bash
pip install -U openvino openvino_tokenizers
```

若要嘗試 nightly：

```bash
pip install --pre -U openvino openvino_tokenizers --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly
```

## Debug 可觀測性

API 新增可選參數 `include_debug`（`true/false`，預設 `false`）。

- `include_debug=true` 時，回應會附上：
  - `requested`：請求裝置策略
  - `resolved`：實際使用裝置
  - `backend`：實際後端
  - `reason`：探測或回退原因

範例：

```bash
curl http://127.0.0.1:8000/v1/audio/transcriptions \
  -X POST \
  -F "file=@/path/to/audio.wav" \
  -F "model=small" \
  -F "response_format=json" \
  -F "include_debug=true"
```

回應片段：

```json
{
  "text": "...",
  "debug": {
    "requested": "auto",
    "resolved": "cpu",
    "backend": "ctranslate2",
    "reason": "openvino_gpu_missing:['CPU']"
  }
}
```

另外，服務端現在會輸出更詳細的執行 log（預設 `INFO`），方便確認實際是否走 GPU：

- `whisper_engine_resolved`：請求裝置、最終裝置、後端與原因。
- `whisper_model_init` / `whisper_cpu_model_init`：模型初始化是否命中 cache，及 `device`、`compute_type`。
- `whisper_gpu_init_try` / `whisper_gpu_init_ok` / `whisper_gpu_init_failed`：GPU 初始化嘗試細節。
- `whisper_gpu_fallback`、`whisper_gpu_transcribe_failed fallback_to_cpu`：GPU 失敗後退回 CPU 的理由。

可用 `LOG_LEVEL` 調整等級（例如 `DEBUG`）：

```bash
LOG_LEVEL=DEBUG uvicorn app.main:app --host 0.0.0.0 --port 8000
```

CPU 路徑預設會使用約 80% 邏輯核心（`cpu_threads = floor(os.cpu_count() * 0.8)`），並在 log 輸出 `whisper_cpu_threads_config`。若要覆寫，可設定環境變數，或在啟動 server 時直接帶參數：

```bash
WHISPER_CPU_USAGE_RATIO=0.5 uvicorn app.main:app --host 0.0.0.0 --port 8000
python -m app.main --host 0.0.0.0 --port 8000 --cpu-usage-ratio 0.5
```

若要設定預設轉錄 `temperature`（當 API request 未帶 `temperature` 時生效），可用環境變數或啟動參數：

```bash
WHISPER_TEMPERATURE=0.2 uvicorn app.main:app --host 0.0.0.0 --port 8000
python -m app.main --host 0.0.0.0 --port 8000 --temperature 0.2
```

## GPU 支援確認參數（新增）

API 新增 `require_gpu`（`true/false`，預設 `false`）：

- `require_gpu=false`：維持原本策略，可回退 CPU。
- `require_gpu=true`：若無法解析為 Intel GPU，會直接回 `400 gpu_not_available`，可用於 CI 或部署驗收。

建議驗證指令：

```bash
WHISPER_DEVICE=intel_gpu curl http://127.0.0.1:8000/v1/audio/transcriptions \
  -X POST \
  -F "file=@/path/to/audio.wav" \
  -F "model=small" \
  -F "response_format=json" \
  -F "include_debug=true" \
  -F "require_gpu=true"
```

判斷方式：
- 成功：`debug.resolved` 為 `intel_gpu`（且 `backend` 為 `openvino`）。
- 失敗：回傳 `400`，`error.code = "gpu_not_available"`，訊息包含原因（例如 `openvino_gpu_missing`）。

## 錯誤格式

```json
{
  "error": {
    "message": "...",
    "type": "invalid_request_error",
    "code": "..."
  }
}
```
