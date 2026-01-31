# Qwen3-ASR OpenAI-Compatible API 🎙️

This project provides a production-ready FastAPI layer for the **Qwen3-ASR** model family. It transforms Alibaba's research code into a high-performance, containerized service with an OpenAI-compatible interface. 

## 🚀 Key Enhancements

### 1. Professional Subtitle Engine
Unlike raw ASR outputs, this API implements a specialized subtitle processor:
*   **Smart CJK Joining:** Automatically detects Chinese, Japanese, and Korean characters to join tokens without spaces, while maintaining spaces for Latin-based languages.
*   **Pause-Based Grouping:** Uses a configurable silence threshold (`max_gap_sec`) to decide when to end a subtitle line, creating natural-sounding breaks.
*   **SRT Generation:** Directly generates `.srt` formatted strings ready for use in video editors like Premiere Pro or DaVinci Resolve.

### 2. Unlimited Audio Handling
The service includes a robust sliding-window logic. Long audio files (e.g., 20-minute podcasts or hour-long videos) are automatically split into optimal chunks, processed in parallel batches on the GPU, and seamlessly merged back together.

### 3. Word-Level Timestamps
By integrating the **Qwen3-ForcedAligner-0.6B** model, the service provides high-precision timestamps for every word/character, allowing for perfect audio-to-text synchronization.

## ✨ Features

*   **OpenAI Compatibility:** Fully compatible with the OpenAI Transcription API spec. Point your existing tools to this API by changing the `base_url`.
*   **Dynamic Model Management:** Load and switch between the `0.6B` and `1.7B` models via API without restarting the container.
*   **Transformers Backend:** Optimized for feature completeness and Forced Aligner stability.
*   **Multi-Stage Build:** Includes a production-hardened Dockerfile with `flash-attn` support for 2-3x faster inference.

## 📂 Project Structure

To maintain compatibility with the core engine, the repository should be structured as follows:

```text
Qwen3-ASR-API/
├── qwen_asr/              # Core logic from official QwenLM repo
├── src/
│   └── stt_service/
│       ├── main.py        # API Entry & Logging Initialization
│       ├── core/
│       │   ├── config.py  # Global Settings & .env Loader
│       │   └── logging_config.py # Professional Logging Setup
│       ├── models/
│       │   └── schemas.py # Pydantic Request/Response Models
│       └── services/
│           ├── model_manager.py  # Dynamic Loader & Attribute Overrider
│           └── subtitle_utils.py # ComfyUI Linguistic & SRT Logic
├── models/                # Host folder for cached weights
├── .env                   # Configuration file
├── Dockerfile             # Multi-stage GPU-optimized build
└── docker-compose.yml     # Orchestration
```

---

## 🚀 Installation & Setup

### 1. Prepare the Repository
Fork the official [QwenLM/Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) repo and create a production branch. **Keep only the `qwen_asr` folder** and the `LICENSE` to keep the repo slim.

### 2. Configure Environment (`.env`)
Create a `.env` file in the root directory to customize your host and container settings:

```ini
# --- HOST SETTINGS ---
API_PORT=8000
HOST_MODELS_PATH=./models

# --- STARTUP DEFAULTS ---
DEFAULT_MODEL=Qwen/Qwen3-ASR-0.6B
DEFAULT_DEVICE=cuda:0
DEFAULT_DTYPE=bf16
LOG_LEVEL=INFO

# --- CACHE (Must match volume mapping) ---
HF_HOME=/app/models
```

### 3. Launch with Docker
```bash
docker compose up --build -d
```

---

## 📋 API Documentation

### 1. Service Health & Status
Check if the API is responsive and see which model is currently taking up VRAM.

*   **URL:** `/health`
*   **Method:** `GET`
*   **Description:** Returns the initialization state and hardware assignment.

**Example Response:**
```json
{
    "status": "ok",
    "model_loaded": "Qwen/Qwen3-ASR-0.6B",
    "device": "cuda:0"
}
```

---

### 2. Dynamic Model Management
Load or switch models, devices, and engine settings without restarting the container. The service starts in an **IDLE** state (no model loaded) to save resources.

*   **URL:** `/load_model`
*   **Method:** `POST`
*   **Content-Type:** `application/json`

**Request Body Parameters:**

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `model_id` | string | `Qwen/Qwen3-ASR-0.6B` | HF Repo ID (0.6B or 1.7B). |
| `device` | string | `cuda:0` | Target GPU (e.g., `cuda:0`, `cuda:1`) or `cpu`. |
| `dtype` | string | `bf16` | Precision: `bf16`, `fp16`, or `fp32`. |
| `use_aligner` | boolean | `true` | Load the Forced Aligner model for subtitles. |
| `attn_implementation` | string | `sdpa` | `sdpa` (fastest) or `eager`. |
| `max_inference_batch_size` | int | `32` | Default batching size for this model instance. |
| `max_new_tokens` | int | `512` | Default token limit for this model instance. |

**cURL Example:**
```bash
curl -X POST http://localhost:8000/load_model \
     -H "Content-Type: application/json" \
     -d '{"model_id": "Qwen/Qwen3-ASR-1.7B", "device": "cuda:0", "dtype": "bf16"}'
```

---

### 3. Audio Transcription & Subtitles
The main endpoint for speech-to-text. It follows the OpenAI specification but adds professional subtitle controls.

*   **URL:** `/v1/audio/transcriptions`
*   **Method:** `POST`
*   **Content-Type:** `multipart/form-data`

**Request Parameters:**

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `file` | file | **Required** | The audio file (mp3, wav, flac, m4a, etc). |
| `language` | string | `auto` | `auto` detection or force a lang (e.g. `Chinese`). |
| `prompt` | string | "" | Contextual "hints" to improve brand/name accuracy. |
| `max_new_tokens` | int | `512` | Max characters to generate per audio chunk. |
| `max_inference_batch_size` | int | `32` | GPU batch size (lower if you get OOM). |
| `return_timestamps` | boolean | `true` | Enable Aligner logic for word-level timestamps. |
| `max_gap_sec` | float | `0.6` | Silence gap threshold for starting a new subtitle line. |
| `max_chars` | int | `40` | Max character limit for a single subtitle line. |
| `split_mode` | string | `..._length` | Strategy: `punctuation`, `pause`, or `length`. |

**Postman Instructions:**
1.  Set method to **POST**.
2.  Go to the **Body** tab and select **form-data**.
3.  Add a key named `file`, change the type dropdown from "Text" to "**File**", and upload your audio.
4.  Add other keys (like `max_gap_sec`) as "Text" fields.

**cURL Example:**
```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
     -F "file=@/path/to/podcast.mp3" \
     -F "language=English" \
     -F "max_gap_sec=0.8" \
     -F "max_chars=60"
```

---

## 🎥 Output Format

The API returns a structured JSON response designed for both UI display and automated video processing.

**Response Structure:**
*   `text`: The full, continuous transcript string.
*   `srt`: A valid **SubRip (.srt)** formatted string, ready to save to a file.
*   `language`: The canonical name of the detected language.
*   `duration`: Total audio duration in seconds.
*   `segments`: An array of JSON objects containing the `start`, `end`, and `text` for every grouped sentence.

---

## ⚡ Performance Monitoring
Monitor your terminal or Docker logs (`docker logs -f qwen3_asr_api`) for real-time performance metrics:

*   **Audio Decoded:** Length of the input audio.
*   **RTF (Real-Time Factor):** How much faster than real-time the GPU is processing. A value of `10.0x` means 100 seconds of audio was transcribed in 10 seconds.
*   **Model Switches:** Logged when the engine moves between devices or model sizes.

---

## 🛠️ Technical Details

*   **Linguistic Processing:** Implements a Unicode-aware `join_tokens` function that prevents unnatural spaces in CJK text while preserving them for Latin text.
*   **Performance Tracking:** Logs include the **Real-Time Factor (RTF)**, calculated as `Audio Duration / Processing Time`.
*   **Attention Backend:** Uses PyTorch native **SDPA**, providing high-speed inference without the long compilation times of Flash-Attention.
