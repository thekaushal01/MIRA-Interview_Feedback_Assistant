import gradio as gr
import whisper
import moviepy.editor as mp
import re
import os
import tempfile
import librosa
import numpy as np

# Load lightweight Whisper model
model = whisper.load_model("tiny", device="cpu")


# Extract Audio
def extract_audio(video_path):
    clip = mp.VideoFileClip(video_path)

    # Limit video duration (2 minutes max)
    if clip.duration > 120:
        clip.close()
        return None, "Please upload a video under 2 minutes."

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        audio_path = temp_file.name

    clip.audio.write_audiofile(audio_path, verbose=False, logger=None)
    clip.close()
    return audio_path, None


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


# Main Analysis Function
def analyze_video(video):
    audio_path, error = extract_audio(video)

    if error:
        return error

    try:
        result = model.transcribe(audio_path)
        text = result["text"]

        # Filler words
        fillers = ["um", "uh", "like", "you know", "basically", "actually"]
        filler_count = sum(len(re.findall(rf"\b{word}\b", text.lower())) for word in fillers)

        clip = mp.VideoFileClip(video)
        duration = clip.duration
        clip.close()

        words_per_minute = speech_rate(text, duration)
        pitch_std = pitch_variation(audio_path)

        # Confidence scoring logic
        confidence_score = 100
        confidence_score -= filler_count * 2

        if words_per_minute < 80:
            confidence_score -= 10
        elif words_per_minute > 170:
            confidence_score -= 10

        if pitch_std < 20:
            confidence_score -= 10

        confidence_score = max(0, min(100, confidence_score))

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
    finally:
        if audio_path and os.path.exists(audio_path):
            os.remove(audio_path)


# Gradio Interface
iface = gr.Interface(
    fn=analyze_video,
    inputs=gr.Video(label="Upload Interview Video (Max 2 min)"),
    outputs=gr.Textbox(label="Mira's Feedback"),
    title="Mira – AI Interview Feedback Assistant",
    description="Upload a short interview video to receive AI-based communication feedback."
)

if __name__ == "__main__":
    iface.launch()