import requests
from moviepy.editor import VideoFileClip
import whisper
import torch
from transformers import pipeline, BartForConditionalGeneration, BartTokenizer, OPTForCausalLM, GPT2Tokenizer
import os
import subprocess
import datetime
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import base64
from email.mime.text import MIMEText

# Base directory
BASE_DIR = r"YOUR_WORKING_DIR"
SCOPES = ["https://www.googleapis.com/auth/gmail.send"]

# Download function for Google Drive
def download_gdrive_file(url, output_name):
    file_id = url.split('/d/')[1].split('/')[0]
    download_url = f'https://drive.google.com/uc?export=download&id={file_id}'
    response = requests.get(download_url, stream=True)
    if response.status_code == 200:
        with open(output_name, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print(f"File downloaded as {output_name}")
    else:
        print(f"Failed to download. Status code: {response.status_code}")

# Check if FFmpeg is available
def check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except FileNotFoundError:
        print("Error: FFmpeg is not installed or not in PATH. Install FFmpeg and add it to your system PATH.")
        return False
    except Exception as e:
        print(f"Error checking FFmpeg: {str(e)}")
        return False

# Load or download Whisper model
def load_or_download_whisper_model(model_name="base", local_dir=os.path.join(BASE_DIR, "local_whisper")):
    local_path = os.path.join(local_dir, f"{model_name}.pt")
    os.makedirs(local_dir, exist_ok=True)
    if os.path.exists(local_path):
        print(f"Loading Whisper '{model_name}' model from {local_path}")
        model = whisper.load_model(model_name, download_root=None)
        state_dict = torch.load(local_path)
        model.load_state_dict(state_dict)
    else:
        print(f"Downloading Whisper '{model_name}' model and saving to {local_path}")
        model = whisper.load_model(model_name, download_root=local_dir)
        torch.save(model.state_dict(), local_path)
        print(f"Model saved to {local_path}")
    return model

# Load or download BART model and tokenizer
def load_or_download_bart_model(model_name="facebook/bart-large-cnn", local_dir=os.path.join(BASE_DIR, "local_bart_model")):
    model_file = os.path.join(local_dir, "model.safetensors")
    config_file = os.path.join(local_dir, "config.json")
    tokenizer_files = [os.path.join(local_dir, f) for f in ["vocab.json", "merges.txt"]]
    
    if os.path.exists(model_file) and os.path.exists(config_file) and all(os.path.exists(f) for f in tokenizer_files):
        print(f"Loading BART model from {local_dir}")
        model = BartForConditionalGeneration.from_pretrained(local_dir)
        tokenizer = BartTokenizer.from_pretrained(local_dir)
    else:
        print(f"Local BART model files missing in {local_dir}. Downloading '{model_name}' instead.")
        model = BartForConditionalGeneration.from_pretrained(model_name)
        tokenizer = BartTokenizer.from_pretrained(model_name)
        model.save_pretrained(local_dir)
        tokenizer.save_pretrained(local_dir)
        print(f"BART model and tokenizer saved to {local_dir}")
    return model, tokenizer

# Load or download OPT model and tokenizer
def load_or_download_opt_model(model_name="facebook/opt-350m", local_dir=os.path.join(BASE_DIR, "local_opt_model")):
    model_file = os.path.join(local_dir, "pytorch_model.bin")
    config_file = os.path.join(local_dir, "config.json")
    tokenizer_files = [os.path.join(local_dir, "tokenizer.json")]
    
    if os.path.exists(model_file) and os.path.exists(config_file) and all(os.path.exists(f) for f in tokenizer_files):
        print(f"Loading OPT model from {local_dir}")
        model = OPTForCausalLM.from_pretrained(local_dir)
        tokenizer = GPT2Tokenizer.from_pretrained(local_dir)
    else:
        print(f"Local OPT model files missing in {local_dir}. Downloading '{model_name}' instead.")
        model = OPTForCausalLM.from_pretrained(model_name)
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        model.save_pretrained(local_dir)
        tokenizer.save_pretrained(local_dir)
        print(f"OPT model and tokenizer saved to {local_dir}")
    return model, tokenizer

# Dynamic summarizer
def summarize_long_text_dynamic(text, summarizer, max_input_tokens=1000, style="default"):
    tokenizer = summarizer.tokenizer
    tokens = tokenizer(text)["input_ids"]
    input_token_count = len(tokens)

    max_length = max(50, min(500, int(input_token_count * 0.15)))
    min_length = max(20, min(200, int(input_token_count * 0.05)))

    style_configs = {
        "concise": {"max_length_factor": 0.05, "min_length_factor": 0.02, "prefix": "Short summary: "},
        "detailed": {"max_length_factor": 0.25, "min_length_factor": 0.10, "prefix": "Detailed summary: "},
        "formal": {"max_length_factor": 0.15, "min_length_factor": 0.05, "prefix": "Formal summary: "},
        "casual": {"max_length_factor": 0.15, "min_length_factor": 0.05, "prefix": "Yo, here’s the rundown: "},
        "bullet": {"max_length_factor": 0.15, "min_length_factor": 0.05, "prefix": "Key points:\n- "},
        "default": {"max_length_factor": 0.15, "min_length_factor": 0.05, "prefix": ""}
    }

    config = style_configs.get(style, style_configs["default"])
    max_length = max(50, min(500, int(input_token_count * config["max_length_factor"])))
    min_length = max(20, min(200, int(input_token_count * config["min_length_factor"])))

    print(f"Style: {style}, Input tokens: {input_token_count}, Summary max_length: {max_length}, min_length: {min_length}")

    if input_token_count <= max_input_tokens:
        summary = summarizer(text, min_length=min_length, max_length=max_length)[0]["summary_text"]
        if style == "bullet":
            summary = config["prefix"] + "\n- ".join(summary.split(". "))
        else:
            summary = config["prefix"] + summary
        return summary

    chunks = []
    for i in range(0, len(tokens), max_input_tokens):
        chunk_tokens = tokens[i:i + max_input_tokens]
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        chunks.append(chunk_text)

    chunk_max_length = max(30, min(150, int(max_input_tokens * 0.10)))
    chunk_min_length = max(10, min(50, int(max_input_tokens * 0.03)))

    summaries = [summarizer(chunk, min_length=chunk_min_length, max_length=chunk_max_length)[0]["summary_text"] for chunk in chunks]
    combined_text = " ".join(summaries)

    combined_tokens = tokenizer(combined_text)["input_ids"]
    if len(combined_tokens) > max_input_tokens:
        return summarize_long_text_dynamic(combined_text, summarizer, max_input_tokens, style)

    summary = summarizer(combined_text, min_length=min_length, max_length=max_length)[0]["summary_text"]
    if style == "bullet":
        summary = config["prefix"] + "\n- ".join(summary.split(". "))
    else:
        summary = config["prefix"] + summary
    return summary

# Task extraction
def extract_tasks_generative(text, generator, max_input_tokens=974, max_new_tokens=200):
    tokenizer = generator.tokenizer
    tokens = tokenizer(text)["input_ids"]
    input_token_count = len(tokens)

    base_prompt = """Analyze the following conversation and list all tasks assigned in the format 'Person: Task'. If no tasks are assigned, return 'No tasks assigned'. Use context to identify assignees and actions. Examples:
- 'Jenkins, you got the report ready?' → 'Jenkins: Prepare the report'
- 'Harris, you mind taking that?' → 'Harris: Coordinate with the state office'
- 'We oughta let the public know' → 'Team: Post an update' (if no specific person is assigned)
- 'I’ll do it' → 'Lopez: Grab the cost estimates' (use prior context for 'I')

Conversation:\n\n"""

    if input_token_count <= max_input_tokens:
        prompt = base_prompt + text
        tasks = generator(prompt, max_new_tokens=max_new_tokens, truncation=True, temperature=0.7, top_k=50, do_sample=True, pad_token_id=tokenizer.eos_token_id)[0]["generated_text"][len(prompt):].strip()
        return tasks

    chunks = []
    for i in range(0, len(tokens), max_input_tokens):
        chunk_tokens = tokens[i:i + max_input_tokens]
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        chunks.append(chunk_text)

    tasks_list = []
    for chunk in chunks:
        prompt = base_prompt + chunk
        task_output = generator(prompt, max_new_tokens=max_new_tokens, truncation=True, temperature=0.7, top_k=50, do_sample=True, pad_token_id=tokenizer.eos_token_id)[0]["generated_text"][len(prompt):].strip()
        if task_output and "No tasks assigned" not in task_output.lower():
            tasks_list.append(task_output)

    if not tasks_list:
        return "No tasks assigned"
    combined_tasks = "\n".join(tasks_list)
    task_lines = []
    for line in combined_tasks.split("\n"):
        line = line.strip()
        if ":" in line and not any(x in line.lower() for x in ["email", "from", "sent", "subject"]):
            task_lines.append(line)
        elif line and "assigned" not in line.lower():
            if "will" in line or "can" in line:
                parts = line.split(" ", 1)
                if len(parts) > 1:
                    task_lines.append(f"{parts[0]}: {parts[1]}")
    unique_tasks = list(dict.fromkeys(task_lines))
    return "\n".join(unique_tasks) if unique_tasks else "No tasks assigned"

# Send summary via Gmail
def send_summary_email(summary, transcript, tasks, recipients, subject="Meeting Summary"):
    creds = None
    token_path = os.path.join(BASE_DIR, "token.json")
    creds_path = os.path.join(BASE_DIR, "long-justice-452914-c1-4de2f1e04728.json")
    
    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(creds_path, SCOPES)
            creds = flow.run_local_server(port=0)
            with open(token_path, "w") as token:
                token.write(creds.to_json())

    try:
        service = build("gmail", "v1", credentials=creds)
        
        # Email body
        body = f"Summary:\n{summary}\n\nFull Transcript:\n{transcript}\n\nTasks:\n{tasks}"
        message = MIMEText(body)
        message['to'] = ", ".join(recipients)
        message['subject'] = subject
        
        # Encode the message
        raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode()
        email = {'raw': raw_message}

        # Send the email
        sent_message = service.users().messages().send(userId="me", body=email).execute()
        print(f"Email sent successfully! Message ID: {sent_message['id']}")
        return sent_message
    except HttpError as error:
        print(f"Error sending email: {error}")
        return None

# Main pipeline
def main():
    if not check_ffmpeg():
        return

    # Download video
    gdrive_link = 'ADD_YOUR_VIDEO_GDRIVE_LINK'
    mp4_file = os.path.join(BASE_DIR, "downloaded_file.mp4")
    if not os.path.exists(mp4_file):
        download_gdrive_file(gdrive_link, mp4_file)
    else:
        print(f"Using existing {mp4_file}")

    # Extract audio
    mp3_file = os.path.join(BASE_DIR, "audio.mp3")
    try:
        video_clip = VideoFileClip(mp4_file)
        audio_clip = video_clip.audio
        audio_clip.write_audiofile(mp3_file)
        audio_clip.close()
        video_clip.close()
        print("Audio extraction successful!")
    except Exception as e:
        print(f"Error during audio extraction: {str(e)}")
        return

    # Load Whisper model
    try:
        model_whisper = load_or_download_whisper_model("base")
    except Exception as e:
        print(f"Error loading Whisper model: {str(e)}")
        return

    # Transcribe audio
    try:
        result = model_whisper.transcribe(mp3_file)
        meeting_transcript = result["text"]
        print("Transcript:", meeting_transcript)
    except Exception as e:
        print(f"Error during transcription: {str(e)}")
        return

    # Load BART model and summarizer
    try:
        summarizer_model, summarizer_tokenizer = load_or_download_bart_model()
        summarizer = pipeline("summarization", model=summarizer_model, tokenizer=summarizer_tokenizer)
    except Exception as e:
        print(f"Error loading BART model: {str(e)}")
        return

    # Load OPT model and task generator
    try:
        task_model, task_tokenizer = load_or_download_opt_model()
        task_generator = pipeline("text-generation", model=task_model, tokenizer=task_tokenizer)
    except Exception as e:
        print(f"Error loading OPT model: {str(e)}")
        return

    # Generate summary and tasks
    summary = None
    tasks = None
    try:
        summary = summarize_long_text_dynamic(meeting_transcript, summarizer, max_input_tokens=1000, style="default")
        print("Summary:", summary)
    except Exception as e:
        print(f"Error during summarization: {str(e)}")

    try:
        tasks = extract_tasks_generative(meeting_transcript, task_generator)
        print("Tasks:", tasks)
    except Exception as e:
        print(f"Error during task extraction: {str(e)}")

    # Send summary via email
    if summary and tasks:
        try:
            recipients = ["example@gmail.com"]  # Replace with actual emails
            # Optional: Prompt for recipients
            # recipients_input = input("Enter recipient email addresses (comma-separated): ")
            # recipients = [r.strip() for r in recipients_input.split(",")]
            
            send_summary_email(summary, meeting_transcript, tasks, recipients, subject="Meeting Summary - " + datetime.datetime.now().strftime("%Y-%m-%d"))
        except Exception as e:
            print(f"Error sending email: {str(e)}")

if __name__ == "__main__":
    main()