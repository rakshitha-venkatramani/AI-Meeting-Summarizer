# AI-Meeting-Summarizer
This project takes a video file stored on Google Drive and extracts its audio track. The audio is then transcribed using Whisper.
Next, a concise summary is generated with BART. After that, actionable tasks are identified using the Open Pre-trained Transformer (OPT) model.
Finally, all the results are sent to the user via the Gmail API.

# Features
Downloads a video from Google Drive.
Extracts audio using MoviePy.
Transcribes audio with OpenAI Whisper.
Summarizes the transcript using BART (facebook/bart-large-cnn).
Extracts tasks using OPT (facebook/opt-350m).
Sends the summary, transcript, and tasks to specified recipients via Gmail.

# Prerequisites
FFmpeg: Required for audio extraction and transcription.
Google Cloud Project: OAuth 2.0 credentials for Gmail API 
