import os
import base64
import tempfile
from pathlib import Path
from dash import Dash, html, dcc, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
from faster_whisper import WhisperModel

# Initialize the Dash app with Bootstrap theme
dash_app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = dash_app.server

# Global variable to store the latest transcription
latest_transcription = ""

def format_timestamp(seconds):
    """Format seconds into HH:MM:SS format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

def transcribe_audio(file_content, filename, model_size, device="cuda"):
    """Transcribe audio/video file and return the transcription"""
    global latest_transcription
    
    # Create a temporary file to store the uploaded content
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(filename).suffix) as tmp_file:
        tmp_file.write(file_content)
        tmp_path = tmp_file.name
    
    try:
        # Initialize model
        compute_type = "auto" if device == "cuda" else "int8"
        model = WhisperModel(model_size, device=device, compute_type=compute_type)
        
        # Transcribe
        segments, info = model.transcribe(tmp_path, beam_size=5)
        
        # Format transcription
        transcription_lines = [
            f"Detected language: {info.language} (probability: {info.language_probability:.2f})\n",
            "=" * 80 + "\n\n"
        ]
        
        for segment in segments:
            line = f"[{format_timestamp(segment.start)} --> {format_timestamp(segment.end)}] {segment.text}\n"
            transcription_lines.append(line)
        
        latest_transcription = "".join(transcription_lines)
        return latest_transcription, None
        
    except Exception as e:
        return "", str(e)
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

# App layout
dash_app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("🎙️ Audio/Video Transcription Tool", 
                   className="text-center mb-4 mt-4",
                   style={"color": "#2c3e50"}),
            html.Hr()
        ])
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H4("Upload & Configure", className="mb-0")),
                dbc.CardBody([
                    # File upload
                    html.Label("Select Audio/Video File:", className="fw-bold mb-2"),
                    dcc.Upload(
                        id='upload-file',
                        children=dbc.Button(
                            [html.I(className="bi bi-cloud-upload me-2"), "Choose File"],
                            color="primary",
                            className="w-100"
                        ),
                        multiple=False,
                        className="mb-3"
                    ),
                    html.Div(id='file-info', className="mb-3 text-muted"),
                    
                    # Model selection
                    html.Label("Select Model Size:", className="fw-bold mb-2"),
                    dcc.Dropdown(
                        id='model-dropdown',
                        options=[
                            {'label': 'Tiny (Fastest, Less Accurate)', 'value': 'tiny'},
                            {'label': 'Base (Default)', 'value': 'base'},
                            {'label': 'Small (Balanced)', 'value': 'small'},
                            {'label': 'Medium (More Accurate)', 'value': 'medium'},
                            {'label': 'Large-v2 (Most Accurate, Slower)', 'value': 'large-v2'},
                            {'label': 'Large-v3 (Latest, Most Accurate)', 'value': 'large-v3'},
                        ],
                        value='base',
                        className="mb-3"
                    ),
                    
                    # Device selection
                    html.Label("Processing Device:", className="fw-bold mb-2"),
                    dcc.Dropdown(
                        id='device-dropdown',
                        options=[
                            {'label': 'CUDA (GPU - Faster)', 'value': 'cuda'},
                            {'label': 'CPU (Slower)', 'value': 'cpu'},
                        ],
                        value='cuda',
                        className="mb-3"
                    ),
                    
                    # Transcribe button
                    dbc.Button(
                        [html.I(className="bi bi-play-circle me-2"), "Start Transcription"],
                        id='transcribe-btn',
                        color="success",
                        size="lg",
                        className="w-100",
                        disabled=True
                    ),
                    
                    # Progress/Status
                    html.Div(id='status-message', className="mt-3"),
                    dcc.Loading(
                        id="loading",
                        type="default",
                        children=html.Div(id="loading-output")
                    ),
                ])
            ], className="shadow-sm mb-4")
        ], width=12, lg=4),
        
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(
                    dbc.Row([
                        dbc.Col(html.H4("Transcription Result", className="mb-0")),
                        dbc.Col([
                            dbc.ButtonGroup([
                                dbc.Button(
                                    [html.I(className="bi bi-clipboard me-1"), "Copy"],
                                    id='copy-btn',
                                    color="info",
                                    size="sm",
                                    className="me-2"
                                ),
                                dbc.Button(
                                    [html.I(className="bi bi-download me-1"), "Download"],
                                    id='download-btn',
                                    color="primary",
                                    size="sm"
                                ),
                            ], className="float-end")
                        ], width="auto")
                    ])
                ),
                dbc.CardBody([
                    dcc.Textarea(
                        id='transcription-output',
                        placeholder='Transcription will appear here...',
                        style={
                            'width': '100%',
                            'height': '500px',
                            'fontFamily': 'monospace',
                            'fontSize': '14px'
                        },
                        className="form-control"
                    ),
                    dcc.Download(id="download-text"),
                    # Hidden div for copy functionality
                    html.Div(id='copy-status', className="mt-2")
                ])
            ], className="shadow-sm")
        ], width=12, lg=8)
    ])
], fluid=True, className="py-4", style={"backgroundColor": "#f8f9fa", "minHeight": "100vh"})

# Callback to display uploaded file info and enable transcribe button
@dash_app.callback(
    [Output('file-info', 'children'),
     Output('transcribe-btn', 'disabled')],
    Input('upload-file', 'contents'),
    State('upload-file', 'filename')
)
def update_file_info(contents, filename):
    if contents is None:
        return "", True
    
    file_info = dbc.Alert([
        html.I(className="bi bi-file-earmark-text me-2"),
        html.Strong("Selected file: "),
        html.Span(filename)
    ], color="info", className="py-2")
    
    return file_info, False

# Callback to perform transcription
@dash_app.callback(
    [Output('transcription-output', 'value'),
     Output('status-message', 'children'),
     Output('loading-output', 'children')],
    Input('transcribe-btn', 'n_clicks'),
    [State('upload-file', 'contents'),
     State('upload-file', 'filename'),
     State('model-dropdown', 'value'),
     State('device-dropdown', 'value')],
    prevent_initial_call=True
)
def perform_transcription(n_clicks, contents, filename, model_size, device):
    if contents is None:
        return "", dbc.Alert("Please upload a file first!", color="warning"), ""
    
    # Decode the file contents
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    
    try:
        # Show processing message
        processing_msg = dbc.Alert(
            [html.I(className="bi bi-hourglass-split me-2"), 
             f"Transcribing with {model_size} model on {device.upper()}... This may take a while."],
            color="info"
        )
        
        # Perform transcription
        transcription, error = transcribe_audio(decoded, filename, model_size, device)
        
        if error:
            return "", dbc.Alert(f"Error: {error}", color="danger"), ""
        
        success_msg = dbc.Alert(
            [html.I(className="bi bi-check-circle me-2"), "Transcription completed successfully!"],
            color="success"
        )
        
        return transcription, success_msg, ""
        
    except Exception as e:
        error_msg = dbc.Alert(f"Error: {str(e)}", color="danger")
        return "", error_msg, ""

# Callback for copy to clipboard
@dash_app.callback(
    Output('copy-status', 'children'),
    Input('copy-btn', 'n_clicks'),
    State('transcription-output', 'value'),
    prevent_initial_call=True
)
def copy_to_clipboard(n_clicks, transcription):
    if not transcription:
        return dbc.Alert("Nothing to copy!", color="warning", dismissable=True, duration=2000)
    
    # Note: Actual clipboard copy happens via clientside callback
    return dbc.Alert("Copied to clipboard!", color="success", dismissable=True, duration=2000)

# Clientside callback for actual clipboard copy (runs in browser)
dash_app.clientside_callback(
    """
    function(n_clicks, text) {
        if (n_clicks && text) {
            navigator.clipboard.writeText(text).then(function() {
                console.log('Text copied to clipboard');
            }).catch(function(err) {
                console.error('Failed to copy text: ', err);
            });
        }
        return '';
    }
    """,
    Output('copy-btn', 'n_clicks'),
    Input('copy-btn', 'n_clicks'),
    State('transcription-output', 'value'),
    prevent_initial_call=True
)

# Callback for download
@dash_app.callback(
    Output('download-text', 'data'),
    Input('download-btn', 'n_clicks'),
    State('transcription-output', 'value'),
    State('upload-file', 'filename'),
    prevent_initial_call=True
)
def download_transcription(n_clicks, transcription, original_filename):
    if not transcription:
        return None
    
    # Generate filename
    if original_filename:
        download_filename = Path(original_filename).stem + "_transcription.txt"
    else:
        download_filename = "transcription.txt"
    
    return dict(content=transcription, filename=download_filename)

if __name__ == '__main__':
    dash_app.run_server(debug=True, host='0.0.0.0', port=8050)