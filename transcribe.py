import os
import sys
from pathlib import Path
from faster_whisper import WhisperModel

def transcribe_video(video_path, output_path=None, model_size="base", device="cuda", compute_type="auto"):
    video_path = Path(video_path)
    
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)
    
    if output_path is None:
        output_path = video_path.with_suffix('.txt')
    else:
        output_path = Path(output_path)
    
    print(f"Loading model: {model_size}")
    print(f"Using device: {device}")
    
    # model init
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    
    print(f"Transcribing: {video_path}")
    print("This may take a while depending on the video length...")
    
    # transcribe video
    segments, info = model.transcribe(str(video_path), beam_size=5)
    
    print(f"Detected language: {info.language} (probability: {info.language_probability:.2f})")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for segment in segments:
            f.write(f"[{format_timestamp(segment.start)} --> {format_timestamp(segment.end)}] {segment.text}\n")
    
    print(f"Transcription saved to: {output_path}")

# timestamp pretty print
def format_timestamp(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

def main():
    if len(sys.argv) < 2:
        print("Usage: python transcribe.py <video_file> [output_file] [model_size] [device]")
        print("\nmodel_size: tiny, base (default), small, medium, large-v2, large-v3")
        # small -> good performance for size
        print("device: cuda (default), cpu")
        # examples
        # python transcribe.py /data/video.mp4
        # python transcribe.py /data/video.mp4 /data/output.txt
        # python transcribe.py /data/video.mp4 /data/output.txt medium
        # python transcribe.py /data/video.mp4 /data/output.txt medium cpu
        sys.exit(1)
    
    video_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    model_size = sys.argv[3] if len(sys.argv) > 3 else "base"
    device = sys.argv[4] if len(sys.argv) > 4 else "cuda"
    
    compute_type = "auto" if device == "cuda" else "int8"
    
    transcribe_video(video_path, output_path, model_size, device, compute_type)

if __name__ == "__main__":
    main()