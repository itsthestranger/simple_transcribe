# simple_transcribe

Lightweight python application(s) to extract speech from video files and produce text transcripts with the faster-whisper library. Includes a CLI script (`transcribe.py`) and an optional Dash-based UI (`dash/`) with Docker support.

## Quick start

- CLI (local):

  ```bash
  python3 transcribe.py <video_file> [output_file] [model_size] [device]
  ```

### recommended run (Docker)

Build the image:

```bash
sudo docker build -t whisper-transcribe .
```

Run (mount `videos/` and enable GPUs):

```bash
sudo docker run --gpus all -v /path/to/repo/videos:/data whisper-transcribe /data/<input>.mp4 /data/output.txt
```


- Dash UI (Docker):

  ```bash
  cd dash
  docker-compose build
  docker-compose up -d
  ```


See `requirements.txt` and `dash/requirements.txt` for Python dependencies.

## Layout

- `transcribe.py` — CLI for transcribing videos
- `dash/` — Dash UI (`transcribe_dash.py`), Dockerfile, and `docker-compose.yml`

