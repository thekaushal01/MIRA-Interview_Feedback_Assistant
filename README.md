# Mira – Interview Feedback Assistant

Mira is a lightweight AI-based interview feedback assistant that analyzes short interview videos for communication quality. It combines speech transcription and audio-feature heuristics to evaluate communication clarity and confidence.

## What Mira Does

- Accepts a short interview video (up to 2 minutes)                             
- Extracts audio from the video
- Preprocesses audio (silence trimming, normalization, pre-emphasis)
- Transcribes speech using Whisper (`tiny` model on CPU)
- Calculates words-per-minute speaking rate
- Detects common filler words
- Measures pitch variation as a confidence indicator
- Produces a confidence score out of 100 with a compact feedback summary
- Shows a two-column UI with upload + video preview on the left and feedback on the right

## Tech Stack

- Python
- Gradio (web UI)
- OpenAI Whisper (speech-to-text)
- MoviePy (audio extraction)
- Librosa + NumPy (audio feature analysis)
- PyTorch (Whisper runtime)

## Repository Structure

```text
.
├── app.py              # Main Gradio application and analysis pipeline
├── requirements.txt    # Python dependencies
└── README.md
```

## How It Works

1. **Video intake**
   - User uploads an interview video through Gradio.
   - Video length is validated (max 120 seconds).

2. **Audio extraction**
   - Audio track is extracted into a temporary `.wav` file.

3. **Transcription**
   - Whisper `tiny` model transcribes preprocessed audio into text.

4. **Communication metrics**
   - **Speech rate** = words per minute (WPM)
   - **Filler words** detected from: `um`, `uh`, `like`, `you know`, `basically`, `actually`
   - **Pitch variation** computed from pitch values using Librosa
   - **Text clarity proxies** including repetition ratio and sentence-length patterns

5. **Confidence scoring**
   - Starts at `100` and applies weighted penalties:
   - Pace bands (slow/fast speech)
   - Filler count and filler density penalties
   - Pitch variation range penalties (very low or unusually high)
   - Short-response and short-sentence penalties
   - Repetition-ratio penalties
   - Final score is clamped between `0` and `100`

6. **Cleanup**
   - Temporary extracted audio file is deleted after analysis.

## Setup

### Prerequisites

- Python 3.10+ recommended
- `pip`
- FFmpeg binary support via `imageio-ffmpeg` (installed through `requirements.txt`)

### Installation

```bash
python -m venv venv
```

**Windows (PowerShell):**

```powershell
venv\Scripts\Activate.ps1
```

**macOS/Linux:**

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### Run the App

```bash
python app.py
```

Open the local Gradio URL shown in your terminal (typically `http://127.0.0.1:7860`).

## Usage Notes

- Best input: clear interview recordings under 2 minutes
- Very noisy audio may reduce transcription quality
- First run can be slower while model/runtime assets initialize
- Current model is optimized for lightweight local usage, not maximum transcription accuracy
- UI is optimized for a compact single-screen workflow with upload, preview, and feedback side-by-side

## Limitations

- Confidence score is heuristic-based and not a clinical/psychometric metric
- Filler-word list is fixed and English-centric
- Uses only audio-driven cues, not facial expression/body-language analysis
- Whisper `tiny` trades some accuracy for speed

## Potential Improvements

- Add configurable filler-word dictionaries
- Support language selection and multilingual analysis
- Add richer pacing metrics (pause length, sentence cadence)
- Provide trend tracking across multiple practice sessions
- Offer downloadable feedback reports

## Troubleshooting

### App fails to start

- Ensure virtual environment is activated.
- Reinstall dependencies:

```bash
pip install -r requirements.txt --upgrade
```

### Slow processing

- Use shorter videos.
- Close other heavy apps to free CPU/RAM.

### No audio detected or analysis looks off

- Validate that input video contains a clear speech track.
- Retry with a higher-quality recording.






