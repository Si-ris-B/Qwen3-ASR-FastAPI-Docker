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

## 🏁 Getting Started

### 1. Configure Environment
Create a `.env` file in the root directory:
```ini
DEFAULT_MODEL=Qwen/Qwen3-ASR-0.6B
DEFAULT_ALIGNER=Qwen/Qwen3-ForcedAligner-0.6B
LOG_LEVEL=INFO
HF_HOME=/app/models
```

### 2. Launch Service
Ensure the `qwen_asr` folder is present in your root, then run:
```bash
docker compose up --build -d
```

## 📖 API Reference

### Generate Transcription
Converts audio to text or subtitles using the OpenAI-compatible endpoint.

*   **URL:** `/v1/audio/transcriptions`
*   **Method:** `POST`
*   **Content-Type:** `multipart/form-data`

**Parameters:**

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `file` | file | - | The audio file (mp3, wav, flac, m4a). |
| `language` | string | `auto` | Specific language (e.g., `Chinese`, `English`) or `auto`. |
| `prompt` | string | "" | Optional "hints" (names, technical terms) to improve accuracy. |
| `max_gap_sec` | float | `0.6` | Max silence duration before splitting into a new subtitle line. |
| `max_chars` | int | `40` | Maximum characters allowed per subtitle line. |
| `split_mode` | string | `..._length` | Strategy: `punctuation`, `pause`, `length`, or combination. |

**Example Response:**
```json
{
  "text": "Hello, welcome to the Qwen3 ASR service.",
  "srt": "1\n00:00:00,000 --> 00:00:02,450\nHello, welcome to the Qwen3 ASR service.",
  "language": "English",
  "duration": 2.45,
  "segments": [
    { "index": 1, "start": 0.0, "end": 2.45, "text": "Hello, welcome..." }
  ]
}
```

### Dynamic Model Loading
Load a different model size or precision into VRAM.

*   **URL:** `/load_model`
*   **Method:** `POST`
*   **Body:**
```json
{
  "model_id": "Qwen/Qwen3-ASR-1.7B",
  "precision": "bf16",
  "use_aligner": true
}
```

### Service Health
Check the currently loaded model and system status.

*   **URL:** `/health`
*   **Method:** `GET`

## 🛠️ Technical Details

*   **Sample Rate:** 16000Hz (internal resampling via librosa).
*   **Device Support:** Optimized for NVIDIA GPUs (Compute Capability 8.0+ recommended for `bf16`).
*   **Attention:** Uses Scaled Dot Product Attention (SDPA) as standard, with optional `flash-attn` build.
*   **Storage:** Models are cached in the `./models` volume to prevent re-downloads.
