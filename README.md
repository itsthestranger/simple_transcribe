# simple_transcribe

Lightweight python application(s) to extract speech from video files and produce text transcripts with the faster-whisper library. Includes a CLI script (`transcribe.py`) and an optional Dash-based UI (`dash/`) with Docker support.

## Quick start

- CLI (local):

  ```bash
  python3 transcribe.py <video-file> > transcript.txt
  ```

- Dash UI (Docker):
 - Dash UI (Docker):

  ```bash
  cd dash
  docker-compose build
  docker-compose up -d
  ```

## Recommended run (Docker)

Build the image:

```bash
sudo docker build -t whisper-transcribe .
```

Run (mount `videos/` and enable GPUs):

```bash
sudo docker run --gpus all -v /path/to/repo/videos:/data whisper-transcribe /data/<input>.mp4 /data/output.txt
```

See `run.md` for exact example commands.

See `requirements.txt` and `dash/requirements.txt` for Python dependencies.

## Layout

- `transcribe.py` — CLI for transcribing videos
- `dash/` — Dash UI, Dockerfile, and `docker-compose.yml`
- `videos/` — sample inputs and outputs

