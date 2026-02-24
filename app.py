import gradio as gr
import whisper
import moviepy.editor as mp
import re
import os
import tempfile
import librosa
import numpy as np
import imageio_ffmpeg
from moviepy.config import change_settings


def configure_ffmpeg():
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    ffmpeg_dir = os.path.dirname(ffmpeg_exe)

    current_path = os.environ.get("PATH", "")
    path_parts = current_path.split(os.pathsep) if current_path else []
    if ffmpeg_dir not in path_parts:
        os.environ["PATH"] = ffmpeg_dir + os.pathsep + current_path

    os.environ["IMAGEIO_FFMPEG_EXE"] = ffmpeg_exe
    os.environ["FFMPEG_BINARY"] = ffmpeg_exe
    change_settings({"FFMPEG_BINARY": ffmpeg_exe})


configure_ffmpeg()

# Load lightweight Whisper model
model = whisper.load_model("tiny", device="cpu")


def resolve_video_path(video_input):
    if isinstance(video_input, str):
        return video_input

    if hasattr(video_input, "name") and isinstance(video_input.name, str):
        return video_input.name

    if isinstance(video_input, (tuple, list)):
        for item in video_input:
            if isinstance(item, str):
                return item
            if hasattr(item, "name") and isinstance(item.name, str):
                return item.name

    if isinstance(video_input, dict):
        possible_keys = ["path", "video", "name"]
        for key in possible_keys:
            value = video_input.get(key)
            if isinstance(value, str):
                return value

    return None


# Extract Audio
def extract_audio(video_path):
    if not video_path or not os.path.exists(video_path):
        return None, None, "Could not read uploaded video file. Please upload again."

    clip = mp.VideoFileClip(video_path)

    # Limit video duration (2 minutes max)
    if clip.duration > 120:
        clip.close()
        return None, None, "Please upload a video under 2 minutes."

    if clip.audio is None:
        duration = clip.duration
        clip.close()
        return None, duration, "No audio track found in the video. Please upload a video with clear speech audio."

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        audio_path = temp_file.name

    clip.audio.write_audiofile(audio_path, verbose=False, logger=None)
    duration = clip.duration
    clip.close()
    return audio_path, duration, None


# Speech Rate Analysis
def speech_rate(text, duration):
    words = len(text.split())
    if duration == 0:
        return 0
    return words / (duration / 60)


# Pitch Variation (Confidence Indicator)
def pitch_variation(audio_path):
    y, sr = librosa.load(audio_path)
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_values = pitches[pitches > 0]

    if len(pitch_values) == 0:
        return 0

    return np.std(pitch_values)


def tokenize_words(text):
    return re.findall(r"\b[a-zA-Z']+\b", text.lower())


def repeated_word_ratio(words):
    if not words:
        return 0
    unique_words = set(words)
    return 1 - (len(unique_words) / len(words))


def sentence_count(text):
    sentences = [chunk.strip() for chunk in re.split(r"[.!?]+", text) if chunk.strip()]
    return len(sentences)


def confidence_score_breakdown(text, duration, words_per_minute, filler_count, pitch_std):
    words = tokenize_words(text)
    total_words = len(words)
    filler_density = (filler_count / total_words) if total_words else 0
    repetition_ratio = repeated_word_ratio(words)
    total_sentences = sentence_count(text)
    avg_sentence_length = (total_words / total_sentences) if total_sentences else 0

    score = 100

    # Delivery pace
    if words_per_minute < 90:
        score -= 12
    elif words_per_minute < 110:
        score -= 6
    elif words_per_minute > 190:
        score -= 12
    elif words_per_minute > 170:
        score -= 6

    # Filler intensity
    score -= min(20, filler_count * 3)
    if filler_density > 0.06:
        score -= 10
    elif filler_density > 0.03:
        score -= 5

    # Voice confidence
    if pitch_std < 18:
        score -= 12
    elif pitch_std < 30:
        score -= 6
    elif pitch_std > 700:
        score -= 8

    # Text clarity proxies
    if total_words < 45:
        score -= 12
    elif total_words < 70:
        score -= 6

    if avg_sentence_length < 6:
        score -= 8

    if repetition_ratio > 0.55:
        score -= 8
    elif repetition_ratio > 0.45:
        score -= 4

    return max(0, min(100, int(round(score))))


# Main Analysis Function
def analyze_video(video):
    video_path = resolve_video_path(video)
    if not video_path:
        return "Invalid video input received. Please upload the video again and retry."

    audio_path, duration, error = extract_audio(video_path)

    if error:
        return error

    try:
        waveform, _ = librosa.load(audio_path, sr=16000, mono=True)
        result = model.transcribe(waveform)
        text = result["text"]

        # Filler words
        fillers = ["um", "uh", "like", "you know", "basically", "actually"]
        filler_count = sum(len(re.findall(rf"\b{word}\b", text.lower())) for word in fillers)

        words_per_minute = speech_rate(text, duration)
        pitch_std = pitch_variation(audio_path)

        confidence_score = confidence_score_breakdown(
            text=text,
            duration=duration,
            words_per_minute=words_per_minute,
            filler_count=filler_count,
            pitch_std=pitch_std,
        )

        return f"""
🎙 Transcript:
{text}

☰ Analysis:
• Duration: {round(duration,2)} sec
• Words per minute: {round(words_per_minute,2)}
• Filler words: {filler_count}
• Pitch variation: {round(pitch_std,2)}

𖣠 Confidence Score: {confidence_score}/100
"""
    except FileNotFoundError as exc:
        return f"Processing failed: required system binary not found ({exc}). Please restart the app and try again."
    except Exception as exc:
        return f"Processing failed: {exc}"
    finally:
        if audio_path and os.path.exists(audio_path):
            os.remove(audio_path)


# Gradio Interface
iface = gr.Interface(
    fn=analyze_video,
    inputs=gr.File(label="Upload Interview Video (Max 2 min)", file_types=["video"]),
    outputs=gr.Textbox(label="Mira's Feedback"),
    title="Mira – AI Interview Feedback Assistant",
    description="Upload a short interview video to receive AI-based communication feedback."
)

if __name__ == "__main__":
    iface.launch()