# simple_transcribe

Lightweight python application(s) to extract speech from video files and produce text transcripts with the faster-whisper library. Includes a CLI script (`transcribe.py`) and an optional Dash-based UI (`dash/`) with Docker support.

## Quick start

### CLI:

  Build the image:
  
  ```bash
  sudo docker build -t transcribe .
  ```
  
  Run (mount `<videos>/` and enable GPUs):
  
  ```bash
  sudo docker run --gpus all -v /path/to/<videos>:/data transcribe /data/<input>.mp4 /data/<output>.txt
  ```


### Dash UI:

  ```bash
  cd dash
  docker-compose build
  docker-compose up -d
  ```


The application(s) can be run locally, but Docker is recommended.
See `requirements.txt` and `dash/requirements.txt` for Python dependencies.

## Layout

- `transcribe.py` — CLI for transcribing videos
- `dash/` — Dash UI (`transcribe_dash.py`), Dockerfile, and `docker-compose.yml`

