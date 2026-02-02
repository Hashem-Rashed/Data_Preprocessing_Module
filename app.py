"""
Dash web application for the Modern Data Analysis Pipeline
"""
import dash
from dash import dcc, html, Input, Output, State, callback_context, dash_table
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pipeline_manager import PipelineManager
import base64
import io
import os
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# Import dashboard components
from dashboard import create_dashboard
# Import enhanced dashboard
from enhanced_dashboard import create_enhanced_dashboard

# Import settings from config
try:
    from config import settings
    PREVIEW_ROWS = getattr(settings, 'PREVIEW_ROWS', 10)
    OUTLIER_FACTOR = getattr(settings, 'OUTLIER_FACTOR', 1.5)
    DROP_THRESHOLD = getattr(settings, 'DROP_THRESHOLD', 0.5)
    IMPUTE_STRATEGY = getattr(settings, 'IMPUTE_STRATEGY', 'median')
    OUTPUT_PATH = getattr(settings, 'OUTPUT_PATH', 'data/cleaned_data.csv')
    REPORT_PATH = getattr(settings, 'REPORT_PATH', 'reports/quality_report.txt')
except:
    # Default values if config import fails
    PREVIEW_ROWS = 10
    OUTLIER_FACTOR = 1.5
    DROP_THRESHOLD = 0.5
    IMPUTE_STRATEGY = 'median'
    OUTPUT_PATH = 'data/cleaned_data.csv'
    REPORT_PATH = 'reports/quality_report.txt'

# Initialize the app
app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    title='Modern Data Analysis Pipeline',
    assets_folder='assets'  # This automatically loads CSS files from assets folder
)

# Initialize pipeline manager
manager = PipelineManager()

# Global variable to store data
global_data = None
cleaned_data = None
reports = {}

# App layout - USING CSS CLASSES INSTEAD OF INLINE STYLES
app.layout = html.Div([
    # Font Awesome for icons
    html.Link(
        rel='stylesheet',
        href='https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css'
    ),
    
    # Main container with CSS class
    html.Div([
        # Header - USING CSS CLASSES
        html.Div([
            html.H1("📊 Modern Data Analysis Pipeline", className="dashboard-title"),
            html.P("Automated data cleaning and visualization dashboard", 
                   className="dashboard-subtitle"),
        ], className="header"),
        
        # Tabs for different sections
        dcc.Tabs([
            # Tab 1: Data Upload & Processing
            dcc.Tab(label='📁 Data Processing', children=[
                html.Div([
                    # Left column for upload/controls
                    html.Div([
                        html.H3("Upload Data", className="section-title"),
                        
                        # File Upload with CSS class
                        dcc.Upload(
                            id='upload-data',
                            children=html.Div([
                                html.I(className="fas fa-cloud-upload-alt", style={'marginRight': '10px'}),
                                'Drag and Drop or ',
                                html.A('Select Files')
                            ]),
                            className="upload-area"
                        ),
                        
                        # Large file processing button
                        html.Button("Process Large File (Chunked)", 
                                   id="process-large-file-btn",
                                   className="dashboard-button secondary",
                                   style={'display': 'none'}),
                        
                        # Continue with large file button
                        html.Button("Continue Anyway with Large File", 
                                   id="continue-large-file-btn",
                                   className="dashboard-button warning",
                                   style={'display': 'none'}),
                        
                        # Or use sample data
                        html.Button("Use Sample Data", 
                                   id="sample-data-btn",
                                   className="dashboard-button"),
                        
                        # Current file info with CSS class
                        html.Div(id='file-info', className="info-card"),
                        
                        html.Hr(className="divider"),
                        
                        # Processing Options with enhanced styling
                        html.Div([
                            html.H4("⚙️ Processing Options", className="section-title"),
                            
                            # Outlier Factor
                            html.Div([
                                html.Div([
                                    html.Span("📊", className="option-icon"),
                                    html.Label("Outlier Factor (IQR)", className="form-label"),
                                    html.Div([
                                        dcc.Slider(
                                            id='outlier-factor',
                                            min=1,
                                            max=3,
                                            step=0.1,
                                            value=OUTLIER_FACTOR,
                                            marks={i: str(i) for i in [1, 1.5, 2, 2.5, 3]},
                                            className="processing-slider"
                                        ),
                                        html.Div([
                                            html.Span("Current: ", className="value-label"),
                                            html.Span(id='outlier-factor-value', children=str(OUTLIER_FACTOR), className="current-value")
                                        ], className="slider-value-display")
                                    ], className="rc-slider-container")
                                ], className="form-group"),
                                
                                # Missing Values Threshold
                                html.Div([
                                    html.Div([
                                        html.Span("📉", className="option-icon"),
                                        html.Label("Missing Values Threshold (%)", className="form-label"),
                                        html.Div([
                                            dcc.Slider(
                                                id='drop-threshold',
                                                min=0,
                                                max=100,
                                                step=5,
                                                value=DROP_THRESHOLD * 100,
                                                marks={i: f'{i}%' for i in [0, 25, 50, 75, 100]},
                                                className="processing-slider"
                                            ),
                                            html.Div([
                                                html.Span("Current: ", className="value-label"),
                                                html.Span(id='drop-threshold-value', children=f"{DROP_THRESHOLD * 100}%", className="current-value")
                                            ], className="slider-value-display")
                                        ], className="rc-slider-container")
                                    ], className="form-group"),
                                ], className="form-row"),
                                
                                # Imputation Strategy and Preview Rows
                                html.Div([
                                    html.Div([
                                        html.Div([
                                            html.Span("🔄", className="option-icon"),
                                            html.Label("Imputation Strategy", className="form-label"),
                                            dcc.Dropdown(
                                                id='impute-strategy',
                                                options=[
                                                    {'label': '📊 Median', 'value': 'median'},
                                                    {'label': '📈 Mean', 'value': 'mean'},
                                                    {'label': '🎯 Most Frequent', 'value': 'most_frequent'}
                                                ],
                                                value=IMPUTE_STRATEGY,
                                                className="filter-dropdown"
                                            )
                                        ], className="form-group"),
                                        
                                        html.Div([
                                            html.Div([
                                                html.Span("👁️", className="option-icon"),
                                                html.Label("Preview Rows", className="form-label"),
                                                html.Div([
                                                    dcc.Slider(
                                                        id='preview-rows',
                                                        min=5,
                                                        max=50,
                                                        step=5,
                                                        value=PREVIEW_ROWS,
                                                        marks={i: str(i) for i in [5, 10, 20, 30, 50]},
                                                        className="processing-slider"
                                                    ),
                                                    html.Div([
                                                        html.Span("Current: ", className="value-label"),
                                                        html.Span(id='preview-rows-value', children=str(PREVIEW_ROWS), className="current-value")
                                                    ], className="slider-value-display")
                                                ], className="rc-slider-container")
                                            ], className="form-group"),
                                        ]),
                                    ], className="form-row"),
                                ]),
                                
                                # Preset Options
                                html.Div([
                                    html.Label("Quick Presets", className="form-label"),
                                    html.Div([
                                        html.Button("Balanced", className="preset-button", id="preset-balanced"),
                                        html.Button("Aggressive", className="preset-button", id="preset-aggressive"),
                                        html.Button("Conservative", className="preset-button", id="preset-conservative"),
                                        html.Button("Custom", className="preset-button active", id="preset-custom")
                                    ], className="preset-options"),
                                ], style={'marginTop': '20px'}),
                                
                                # Reset Button
                                html.Button("↺ Reset to Defaults", className="reset-button", id="reset-options"),
                                
                            ], className="processing-options"),
                        ], className="processing-options-wrapper"),
                        
                        html.Hr(className="divider"),
                        
                        # Process Buttons
                        html.Div([
                            html.Button("Run Full Pipeline", 
                                       id="run-pipeline-btn",
                                       className="dashboard-button primary"),
                            
                            html.Button("Step-by-Step Processing", 
                                       id="step-by-step-btn",
                                       className="dashboard-button secondary"),
                        ], className="button-group"),
                        
                        # Progress Indicator
                        dcc.Loading(
                            id="loading",
                            type="circle",
                            children=html.Div(id="loading-output", className="loading-output")
                        ),
                        
                    ], className="control-panel left-panel"),
                    
                    # Results Column
                    html.Div([
                        html.H3("Data Preview & Results", className="section-title"),
                        
                        # Tab for different previews
                        dcc.Tabs([
                            dcc.Tab(label='Raw Data', children=[
                                html.Div(id='raw-data-preview', className="preview-container")
                            ]),
                            
                            dcc.Tab(label='Cleaned Data', children=[
                                html.Div(id='cleaned-data-preview', className="preview-container")
                            ]),
                            
                            dcc.Tab(label='Data Quality Report', children=[
                                html.Div(id='quality-report', className="preview-container")
                            ])
                        ], className="preview-tabs"),
                        
                        # Statistics Cards with CSS class
                        html.Div(id='statistics-cards', className="metrics-container"),
                        
                    ], className="control-panel right-panel")
                ], className="processing-container")
            ]),
            
            # Tab 2: Dashboard
            dcc.Tab(label='📈 Dashboard', children=[
                html.Div(id='dashboard-content', className="dashboard-content")
            ]),
            
            # Tab 3: Enhanced Dashboard
            dcc.Tab(label='🎯 Enhanced Dashboard', children=[
                html.Div(id='enhanced-dashboard-content', className="dashboard-content")
            ]),
            
            # Tab 4: Export
            dcc.Tab(label='💾 Export', children=[
                html.Div([
                    html.Div([
                        html.H3("Export Results", className="section-title"),
                        
                        html.Div([
                            html.H5("Download Cleaned Data"),
                            html.P("Export the processed dataset to CSV format"),
                            html.Button("Download CSV", 
                                       id="download-csv-btn",
                                       className="dashboard-button success")
                        ], className="export-card"),
                        
                        html.Div([
                            html.H5("Quality Report"),
                            html.P("Download the detailed data quality report"),
                            html.Button("Download Report", 
                                       id="download-report-btn",
                                       className="dashboard-button info")
                        ], className="export-card"),
                        
                        # Download components
                        dcc.Download(id="download-data"),
                        dcc.Download(id="download-report"),
                        
                        html.Hr(className="divider"),
                        
                        # Configuration Summary
                        html.H4("Current Configuration", className="section-title"),
                        html.Div(id='config-summary', className="config-card")
                        
                    ], className="export-left-panel"),
                    
                    html.Div([
                        html.H3("Pipeline Summary", className="section-title"),
                        html.Div(id='pipeline-summary', className="summary-card")
                    ], className="export-right-panel")
                ], className="export-container")
            ])
        ], className="main-tabs"),
        
        # Store for data
        dcc.Store(id='stored-data'),
        dcc.Store(id='stored-cleaned-data'),
        dcc.Store(id='stored-reports'),
        
        # Store for enhanced dashboard
        dcc.Store(id='stored-enhanced-summary'),
        
        # Interval for updates
        dcc.Interval(id='interval-component', interval=1000, n_intervals=0),
        
        # Enhanced dashboard download components
        dcc.Download(id="enhanced-csv-download"),
        dcc.Download(id="enhanced-report-download"),
        
    ], id="main-container")
])

# Callbacks
@app.callback(
    [Output('file-info', 'children'),
     Output('stored-data', 'data'),
     Output('process-large-file-btn', 'style'),
     Output('continue-large-file-btn', 'style')],
    [Input('upload-data', 'contents'),
     Input('sample-data-btn', 'n_clicks'),
     Input('continue-large-file-btn', 'n_clicks')],
    [State('upload-data', 'filename')]
)
def load_data(contents, sample_clicks, continue_clicks, filename):
    ctx = callback_context
    if not ctx.triggered:
        return html.P("No data loaded yet", className="info-text"), None, {'display': 'none'}, {'display': 'none'}
    
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    global global_data
    
    # Default button styles
    large_file_btn_style = {'display': 'none'}
    continue_btn_style = {'display': 'none'}
    
    if trigger_id == 'upload-data' and contents:
        try:
            content_type, content_string = contents.split(',')
            decoded = base64.b64decode(content_string)
            
            # Check file size
            file_size = len(decoded) / (1024 * 1024)  # Size in MB
            
            # For large files (>50MB), show warning and options
            if file_size > 50:
                warning_msg = html.Div([
                    html.H5(f"⚠️ Large File Detected: {filename}", className="warning-title"),
                    html.P(f"Size: {file_size:.1f} MB", className="warning-text"),
                    html.P("This file is large. For better performance:", className="warning-text"),
                ], className="warning-card")
                
                # Show the large file processing buttons
                large_file_btn_style['display'] = 'block'
                continue_btn_style['display'] = 'block'
                
                return warning_msg, None, large_file_btn_style, continue_btn_style
            
            # For normal-sized files, process immediately
            if filename.endswith('.csv'):
                df = pd.read_csv(io.BytesIO(decoded))
            elif filename.endswith('.xlsx'):
                df = pd.read_excel(io.BytesIO(decoded))
            elif filename.endswith('.json'):
                df = pd.read_json(io.BytesIO(decoded))
            else:
                return html.Div([
                    html.H5(f"❌ Unsupported file format: {filename}", className="error-title"),
                    html.P("Supported formats: CSV, Excel (.xlsx), JSON", className="error-text")
                ], className="error-card"), None, large_file_btn_style, continue_btn_style
            
            global_data = df
            info = html.Div([
                html.H5(f"✅ File Loaded: {filename}", className="success-title"),
                html.P(f"Shape: {df.shape[0]:,} rows × {df.shape[1]:,} columns", className="info-text"),
                html.P(f"Size: {file_size:.1f} MB", className="info-text"),
                html.P(f"Memory usage: {df.memory_usage(deep=True).sum() / (1024**2):.2f} MB", className="info-text"),
                html.P(f"Columns preview: {', '.join(df.columns[:5])}..." if len(df.columns) > 5 else f"Columns: {', '.join(df.columns)}", 
                       className="info-text")
            ], className="success-card")
            return info, df.to_json(date_format='iso', orient='split'), large_file_btn_style, continue_btn_style
            
        except Exception as e:
            error_msg = html.Div([
                html.H5(f"❌ Error loading file: {filename}", className="error-title"),
                html.P(f"Error: {str(e)}", className="error-text"),
                html.P("Make sure the file format is correct and not corrupted.", className="error-text")
            ], className="error-card")
            return error_msg, None, large_file_btn_style, continue_btn_style
    
    elif trigger_id == 'sample-data-btn':
        df = manager._create_sample_data()
        global_data = df
        info = html.Div([
            html.H5("📊 Sample Data Generated", className="success-title"),
            html.P(f"Shape: {df.shape[0]:,} rows × {df.shape[1]:,} columns", className="info-text"),
            html.P(f"Memory: {df.memory_usage(deep=True).sum() / (1024**2):.2f} MB", className="info-text"),
            html.P("Note: This is generated sample data for demonstration.", className="info-text")
        ], className="success-card")
        return info, df.to_json(date_format='iso', orient='split'), large_file_btn_style, continue_btn_style
    
    elif trigger_id == 'continue-large-file-btn' and contents:
        # User chose to continue with large file despite warning
        try:
            content_type, content_string = contents.split(',')
            decoded = base64.b64decode(content_string)
            
            file_size = len(decoded) / (1024 * 1024)
            
            if filename.endswith('.csv'):
                df = pd.read_csv(io.BytesIO(decoded))
            elif filename.endswith('.xlsx'):
                df = pd.read_excel(io.BytesIO(decoded))
            elif filename.endswith('.json'):
                df = pd.read_json(io.BytesIO(decoded))
            else:
                return html.Div([
                    html.H5(f"❌ Unsupported file format: {filename}", className="error-title")
                ], className="error-card"), None, large_file_btn_style, continue_btn_style
            
            global_data = df
            info = html.Div([
                html.H5(f"✅ Large File Loaded: {filename}", className="success-title"),
                html.P(f"Shape: {df.shape[0]:,} rows × {df.shape[1]:,} columns", className="info-text"),
                html.P(f"Size: {file_size:.1f} MB", className="info-text"),
                html.P("Processing may be slower due to file size.", className="warning-text"),
                html.P(f"Columns preview: {', '.join(df.columns[:5])}..." if len(df.columns) > 5 else f"Columns: {', '.join(df.columns)}", 
                       className="info-text")
            ], className="success-card")
            return info, df.to_json(date_format='iso', orient='split'), large_file_btn_style, continue_btn_style
            
        except Exception as e:
            error_msg = html.Div([
                html.H5(f"❌ Error loading large file", className="error-title"),
                html.P(f"Error: {str(e)}", className="error-text"),
                html.P("Try using the 'Process Large File (Chunked)' option.", className="error-text")
            ], className="error-card")
            return error_msg, None, large_file_btn_style, continue_btn_style
    
    return html.P("No data loaded yet", className="info-text"), None, large_file_btn_style, continue_btn_style

@app.callback(
    Output('stored-data', 'data', allow_duplicate=True),
    [Input('process-large-file-btn', 'n_clicks')],
    [State('upload-data', 'filename'),
     State('upload-data', 'contents')],
    prevent_initial_call=True
)
def process_large_file_chunked(n_clicks, filename, contents):
    if n_clicks is None or not contents:
        return None
    
    try:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        
        # Save to temp file
        temp_path = f"temp_{filename}"
        with open(temp_path, 'wb') as f:
            f.write(decoded)
        
        # Load in chunks
        chunks = []
        chunk_size = 50000
        
        # Use pandas read_csv with chunks
        for i, chunk in enumerate(pd.read_csv(temp_path, chunksize=chunk_size)):
            chunks.append(chunk)
            # Limit to 500,000 rows max for performance
            if i >= 10:  # 10 chunks * 50,000 = 500,000 rows max
                break
        
        df = pd.concat(chunks, ignore_index=True)
        
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        if df is not None:
            return df.to_json(date_format='iso', orient='split')
        else:
            return None
            
    except Exception as e:
        print(f"Error processing large file: {e}")
        # Clean up temp file if it exists
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)
        return None

@app.callback(
    [Output('stored-cleaned-data', 'data'),
     Output('stored-reports', 'data'),
     Output('loading-output', 'children')],
    [Input('run-pipeline-btn', 'n_clicks')],
    [State('stored-data', 'data'),
     State('outlier-factor', 'value'),
     State('drop-threshold', 'value'),
     State('impute-strategy', 'value')]
)
def run_pipeline(n_clicks, data_json, outlier_factor, drop_threshold, impute_strategy):
    if n_clicks is None or data_json is None:
        return None, None, ""
    
    # Load data
    df = pd.read_json(io.StringIO(data_json), orient='split')
    
    # Run pipeline
    try:
        cleaned_df = manager.run_full_pipeline(df)
        
        # Generate reports
        schema_report = manager.check_data_schema(df)
        null_report = manager.missing_values_report(df)
        outlier_report = manager.outlier_statistics(df)
        
        reports = {
            'schema': schema_report,
            'missing_values': null_report,
            'outliers': outlier_report,
            'final': {
                'timestamp': datetime.now().isoformat(),
                'original_shape': df.shape,
                'cleaned_shape': cleaned_df.shape,
                'columns_removed': list(set(df.columns) - set(cleaned_df.columns)),
                'config': {
                    'outlier_factor': outlier_factor,
                    'drop_threshold': drop_threshold / 100,
                    'impute_strategy': impute_strategy
                }
            }
        }
        
        manager.reports = reports
        
        success_msg = html.Div([
            html.Div("✅ Pipeline completed!", className="success-title"),
            html.Div(f"Cleaned data: {cleaned_df.shape[0]:,} rows × {cleaned_df.shape[1]:,} columns", 
                    className="info-text")
        ], className="success-card")
        
        return cleaned_df.to_json(date_format='iso', orient='split'), reports, success_msg
        
    except Exception as e:
        error_msg = html.Div([
            html.Div(f"❌ Error: {str(e)}", className="error-title")
        ], className="error-card")
        return None, None, error_msg

@app.callback(
    Output('raw-data-preview', 'children'),
    [Input('stored-data', 'data'),
     Input('preview-rows', 'value')]
)
def preview_raw_data(data_json, preview_rows):
    if data_json is None:
        return html.Div("No data loaded yet.", className="info-text")
    
    df = pd.read_json(io.StringIO(data_json), orient='split')
    
    # Create table using dash_table
    table = dash_table.DataTable(
        data=df.head(preview_rows).to_dict('records'),
        columns=[{'name': col, 'id': col} for col in df.columns],
        page_size=preview_rows,
        style_table={'overflowX': 'auto'},
        style_cell={
            'minWidth': '100px', 'maxWidth': '300px',
            'overflow': 'hidden',
            'textOverflow': 'ellipsis',
            'backgroundColor': 'rgba(255, 255, 255, 0.05)',
            'color': 'white',
            'border': '1px solid rgba(255, 255, 255, 0.1)'
        },
        style_header={
            'backgroundColor': 'rgba(255, 255, 255, 0.1)',
            'fontWeight': 'bold',
            'color': 'white',
            'border': '1px solid rgba(255, 255, 255, 0.2)'
        }
    )
    
    return html.Div([
        html.P(f"Showing first {preview_rows} rows of {len(df):,} total", className="info-text"),
        table
    ], className="data-preview")

@app.callback(
    Output('cleaned-data-preview', 'children'),
    [Input('stored-cleaned-data', 'data'),
     Input('preview-rows', 'value')]
)
def preview_cleaned_data(data_json, preview_rows):
    if data_json is None:
        return html.Div("No cleaned data available. Run the pipeline first.", className="info-text")
    
    df = pd.read_json(io.StringIO(data_json), orient='split')
    
    # Create table using dash_table
    table = dash_table.DataTable(
        data=df.head(preview_rows).to_dict('records'),
        columns=[{'name': col, 'id': col} for col in df.columns],
        page_size=preview_rows,
        style_table={'overflowX': 'auto'},
        style_cell={
            'minWidth': '100px', 'maxWidth': '300px',
            'overflow': 'hidden',
            'textOverflow': 'ellipsis',
            'backgroundColor': 'rgba(255, 255, 255, 0.05)',
            'color': 'white',
            'border': '1px solid rgba(255, 255, 255, 0.1)'
        },
        style_header={
            'backgroundColor': 'rgba(255, 255, 255, 0.1)',
            'fontWeight': 'bold',
            'color': 'white',
            'border': '1px solid rgba(255, 255, 255, 0.2)'
        }
    )
    
    return html.Div([
        html.P(f"Showing first {preview_rows} rows of {len(df):,} total", className="info-text"),
        table
    ], className="data-preview")

@app.callback(
    Output('dashboard-content', 'children'),
    [Input('stored-cleaned-data', 'data')]
)
def update_dashboard(data_json):
    if data_json is None:
        return html.Div([
            html.H4("No data available for dashboard", className="section-title"),
            html.P("Please run the data pipeline first.", className="info-text")
        ], className="empty-state")
    
    df = pd.read_json(io.StringIO(data_json), orient='split')
    return create_dashboard(df)

@app.callback(
    Output('enhanced-dashboard-content', 'children'),
    [Input('stored-cleaned-data', 'data')]
)
def update_enhanced_dashboard(data_json):
    """Update enhanced dashboard with cleaned data"""
    if data_json is None:
        return html.Div([
            html.Div([
                html.H4("🎯 Enhanced Dashboard", className="section-title"),
                html.P("This dashboard provides a client-focused view of your processed data.", 
                      className="info-text"),
                html.Hr(className="divider"),
                html.P("📊 Please run the data processing pipeline first to see your data here.",
                      className="warning-text"),
                html.P("1. Go to 'Data Processing' tab", className="info-text"),
                html.P("2. Upload your data or use sample data", className="info-text"),
                html.P("3. Click 'Run Full Pipeline'", className="info-text"),
                html.P("4. Return here to see your enhanced dashboard!", className="info-text")
            ], className="empty-state")
        ])
    
    df = pd.read_json(io.StringIO(data_json), orient='split')
    return create_enhanced_dashboard(df)

@app.callback(
    Output('quality-report', 'children'),
    [Input('stored-reports', 'data')]
)
def show_quality_report(reports_data):
    if reports_data is None:
        return html.Div("No quality report available yet.", className="info-text")
    
    reports = reports_data
    
    cards = []
    
    # Final report card
    if 'final' in reports:
        final = reports['final']
        cards.append(html.Div([
            html.H5("Pipeline Summary", className="card-title"),
            html.P(f"Original: {final['original_shape'][0]:,} × {final['original_shape'][1]:,}", className="info-text"),
            html.P(f"Cleaned: {final['cleaned_shape'][0]:,} × {final['cleaned_shape'][1]:,}", className="info-text"),
            html.P(f"Columns Removed: {len(final['columns_removed'])}", className="info-text")
        ], className="report-card"))
    
    # Missing values card
    if 'missing_values' in reports:
        null_report = reports['missing_values']
        
        if isinstance(null_report, dict):
            total_nulls = null_report.get('total_nulls', 0)
            total_rows = null_report.get('total_rows', 1)
            total_columns = null_report.get('total_columns', 1)
            columns_to_drop = null_report.get('columns_to_drop', [])
            
            total_cells = total_rows * total_columns
            null_percentage = (total_nulls / total_cells * 100) if total_cells > 0 else 0
            
            cards.append(html.Div([
                html.H5("Missing Values", className="card-title"),
                html.P(f"Total Nulls: {total_nulls:,}", className="info-text"),
                html.P(f"Null Percentage: {null_percentage:.1f}%", className="info-text"),
                html.P(f"Columns to Drop: {len(columns_to_drop)}", className="info-text")
            ], className="report-card"))
        else:
            cards.append(html.Div([
                html.H5("Missing Values", className="card-title"),
                html.P(f"Report type: {type(null_report).__name__}", className="info-text")
            ], className="report-card"))
    
    # Outliers card
    if 'outliers' in reports:
        outlier_report = reports['outliers']
        
        if isinstance(outlier_report, str):
            if outlier_report == 'outliers':
                cards.append(html.Div([
                    html.H5("Outliers", className="card-title"),
                    html.P("No numerical columns found for outlier analysis", className="info-text")
                ], className="report-card"))
            else:
                cards.append(html.Div([
                    html.H5("Outliers", className="card-title"),
                    html.P(f"Note: {outlier_report}", className="info-text")
                ], className="report-card"))
        elif isinstance(outlier_report, dict):
            if 'error' in outlier_report:
                error_msg = outlier_report.get('error', 'Unknown error')
                cards.append(html.Div([
                    html.H5("Outliers", className="card-title"),
                    html.P(f"Error: {error_msg}", className="error-text")
                ], className="report-card"))
            else:
                total_outliers = 0
                columns_with_outliers = 0
                
                for key, value in outlier_report.items():
                    if key.startswith('_') or key == 'error':
                        continue
                    
                    if isinstance(value, dict):
                        outliers_data = value.get('outliers')
                        if isinstance(outliers_data, dict):
                            count = outliers_data.get('count', 0)
                            total_outliers += count
                            if count > 0:
                                columns_with_outliers += 1
                
                cards.append(html.Div([
                    html.H5("Outliers", className="card-title"),
                    html.P(f"Total Outliers: {total_outliers:,}", className="info-text"),
                    html.P(f"Columns with outliers: {columns_with_outliers}", className="info-text")
                ], className="report-card"))
        else:
            cards.append(html.Div([
                html.H5("Outliers", className="card-title"),
                html.P(f"Report type: {type(outlier_report).__name__}", className="info-text")
            ], className="report-card"))
    
    return html.Div(cards, className="reports-container")

@app.callback(
    Output('statistics-cards', 'children'),
    [Input('stored-data', 'data'),
     Input('stored-cleaned-data', 'data')]
)
def update_statistics(raw_data_json, cleaned_data_json):
    cards = []
    
    if raw_data_json:
        df_raw = pd.read_json(io.StringIO(raw_data_json), orient='split')
        cards.append(html.Div([
            html.H5("Raw Data", className="stat-label"),
            html.H3(f"{df_raw.shape[0]:,}", className="stat-number"),
            html.P("Rows", className="stat-description"),
            html.H3(f"{df_raw.shape[1]}", className="stat-number"),
            html.P("Columns", className="stat-description"),
        ], className="stat-card"))
    
    if cleaned_data_json:
        df_clean = pd.read_json(io.StringIO(cleaned_data_json), orient='split')
        cards.append(html.Div([
            html.H5("Cleaned Data", className="stat-label"),
            html.H3(f"{df_clean.shape[0]:,}", className="stat-number"),
            html.P("Rows", className="stat-description"),
            html.H3(f"{df_clean.shape[1]}", className="stat-number"),
            html.P("Columns", className="stat-description"),
        ], className="stat-card"))
    
    if raw_data_json:
        df_raw = pd.read_json(io.StringIO(raw_data_json), orient='split')
        null_count = df_raw.isnull().sum().sum()
        total_cells = df_raw.shape[0] * df_raw.shape[1]
        null_percentage = (null_count / total_cells * 100) if total_cells > 0 else 0
        
        cards.append(html.Div([
            html.H5("Missing Values", className="stat-label"),
            html.H3(f"{null_count:,}", className="stat-number"),
            html.P("Null Values", className="stat-description"),
            html.H3(f"{null_percentage:.1f}%", className="stat-number"),
            html.P("Percentage", className="stat-description"),
        ], className="stat-card"))
    
    if raw_data_json:
        df_raw = pd.read_json(io.StringIO(raw_data_json), orient='split')
        num_cols = len(df_raw.select_dtypes(include=['int64', 'float64']).columns)
        cat_cols = len(df_raw.select_dtypes(include=['object', 'category']).columns)
        cards.append(html.Div([
            html.H5("Data Types", className="stat-label"),
            html.H3(f"{num_cols}", className="stat-number"),
            html.P("Numerical", className="stat-description"),
            html.H3(f"{cat_cols}", className="stat-number"),
            html.P("Categorical", className="stat-description"),
        ], className="stat-card"))
    
    return html.Div(cards, className="statistics-grid")

@app.callback(
    Output('download-data', 'data'),
    [Input('download-csv-btn', 'n_clicks')],
    [State('stored-cleaned-data', 'data')]
)
def download_data(n_clicks, data_json):
    if n_clicks is None or data_json is None:
        return None
    
    df = pd.read_json(io.StringIO(data_json), orient='split')
    
    # Save to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"cleaned_data_{timestamp}.csv"
    
    return dcc.send_data_frame(df.to_csv, filename, index=False)

@app.callback(
    Output('download-report', 'data'),
    [Input('download-report-btn', 'n_clicks')],
    [State('stored-reports', 'data')]
)
def download_report(n_clicks, reports_data):
    if n_clicks is None or reports_data is None:
        return None
    
    import json
    
    # Convert reports to JSON
    report_json = json.dumps(reports_data, indent=2)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"quality_report_{timestamp}.json"
    
    return dict(content=report_json, filename=filename)

@app.callback(
    Output('config-summary', 'children'),
    [Input('outlier-factor', 'value'),
     Input('drop-threshold', 'value'),
     Input('impute-strategy', 'value')]
)
def update_config_summary(outlier_factor, drop_threshold, impute_strategy):
    return html.Div([
        html.P(f"Outlier Factor (IQR): {outlier_factor}", className="info-text"),
        html.P(f"Drop Threshold: {drop_threshold}%", className="info-text"),
        html.P(f"Imputation Strategy: {impute_strategy}", className="info-text"),
        html.P(f"Preview Rows: {PREVIEW_ROWS}", className="info-text")
    ], className="config-list")

@app.callback(
    Output('numerical-chart', 'figure'),
    [Input('num-col-selector', 'value'),
     Input('num-chart-type', 'value')],
    [State('stored-cleaned-data', 'data')]
)
def update_numerical_chart(col, chart_type, data_json):
    if not data_json or not col:
        return go.Figure()
    
    df = pd.read_json(io.StringIO(data_json), orient='split')
    
    if col not in df.columns:
        return go.Figure()
    
    if chart_type == 'histogram':
        fig = px.histogram(df, x=col, title=f"Distribution of {col}")
        fig.update_layout(bargap=0.1)
    elif chart_type == 'box':
        fig = px.box(df, y=col, title=f"Box Plot of {col}")
    elif chart_type == 'violin':
        fig = px.violin(df, y=col, title=f"Violin Plot of {col}")
    elif chart_type == 'ecdf':
        fig = px.ecdf(df, x=col, title=f"ECDF of {col}")
    else:
        fig = px.histogram(df, x=col, title=f"Distribution of {col}")
    
    return fig


@app.callback(
    Output('categorical-chart', 'figure'),
    [Input('cat-col-selector', 'value'),
     Input('cat-num-col-selector', 'value')],
    [State('stored-cleaned-data', 'data')]
)
def update_categorical_chart(cat_col, num_col, data_json):
    if not data_json or not cat_col:
        return go.Figure()
    
    df = pd.read_json(io.StringIO(data_json), orient='split')
    
    if cat_col not in df.columns:
        return go.Figure()
    
    if num_col and num_col in df.columns:
        # Create grouped box plot
        fig = px.box(df, x=cat_col, y=num_col, 
                    title=f"{num_col} by {cat_col}")
    else:
        # Create bar chart of value counts
        value_counts = df[cat_col].value_counts().reset_index()
        value_counts.columns = [cat_col, 'count']
        
        fig = px.bar(value_counts.head(20), x=cat_col, y='count',
                    title=f"Value Counts for {cat_col}")
    
    return fig


@app.callback(
    Output('value-counts-chart', 'figure'),
    [Input('cat-col-selector', 'value')],
    [State('stored-cleaned-data', 'data')]
)
def update_value_counts_chart(cat_col, data_json):
    if not data_json or not cat_col:
        return go.Figure()
    
    df = pd.read_json(io.StringIO(data_json), orient='split')
    
    if cat_col not in df.columns:
        return go.Figure()
    
    # Create pie chart of value counts
    value_counts = df[cat_col].value_counts().reset_index()
    value_counts.columns = [cat_col, 'count']
    
    fig = px.pie(value_counts.head(10), values='count', names=cat_col,
                title=f"Top 10 Categories in {cat_col}")
    
    return fig


@app.callback(
    Output('scatter-plot', 'figure'),
    [Input('x-col-selector', 'value'),
     Input('y-col-selector', 'value'),
     Input('color-col-selector', 'value')],
    [State('stored-cleaned-data', 'data')]
)
def update_scatter_plot(x_col, y_col, color_col, data_json):
    if not data_json or not x_col or not y_col:
        return go.Figure()
    
    df = pd.read_json(io.StringIO(data_json), orient='split')
    
    if x_col not in df.columns or y_col not in df.columns:
        return go.Figure()
    
    if color_col and color_col != 'none' and color_col in df.columns:
        fig = px.scatter(df, x=x_col, y=y_col, color=color_col,
                        title=f"{y_col} vs {x_col} (colored by {color_col})",
                        hover_data=df.columns.tolist())
    else:
        fig = px.scatter(df, x=x_col, y=y_col,
                        title=f"{y_col} vs {x_col}",
                        hover_data=df.columns.tolist())
    
    return fig


# Add the missing step-by-step callback
@app.callback(
    [Output('stored-cleaned-data', 'data', allow_duplicate=True),
     Output('stored-reports', 'data', allow_duplicate=True),
     Output('loading-output', 'children', allow_duplicate=True)],
    [Input('step-by-step-btn', 'n_clicks')],
    [State('stored-data', 'data'),
     State('outlier-factor', 'value'),
     State('drop-threshold', 'value'),
     State('impute-strategy', 'value')],
    prevent_initial_call=True
)
def step_by_step_processing(n_clicks, data_json, outlier_factor, drop_threshold, impute_strategy):
    if n_clicks is None or data_json is None:
        return None, None, ""
    
    # Load data
    df = pd.read_json(io.StringIO(data_json), orient='split')
    
    try:
        # Run each step sequentially
        from pipeline_manager import PipelineManager
        manager = PipelineManager()
        
        # Step 1: Enforce data types
        df = manager.enforce_data_types(df)
        
        # Step 2: Handle nulls
        df = manager.handle_nulls(df)
        
        # Step 3: Clip outliers
        df = manager.clip_outliers(df)
        
        # Generate reports
        schema_report = manager.check_data_schema(df)
        null_report = manager.missing_values_report(df)
        outlier_report = manager.outlier_statistics(df)
        
        reports = {
            'schema': schema_report,
            'missing_values': null_report,
            'outliers': outlier_report,
            'final': {
                'timestamp': datetime.now().isoformat(),
                'original_shape': df.shape,
                'cleaned_shape': df.shape,
                'columns_removed': [],
                'config': {
                    'outlier_factor': outlier_factor,
                    'drop_threshold': drop_threshold / 100,
                    'impute_strategy': impute_strategy
                }
            }
        }
        
        success_msg = html.Div([
            html.Div("✅ Step-by-step processing completed!", className="success-title"),
            html.Div(f"Cleaned data: {df.shape[0]:,} rows × {df.shape[1]:,} columns", className="info-text")
        ], className="success-card")
        
        return df.to_json(date_format='iso', orient='split'), reports, success_msg
        
    except Exception as e:
        error_msg = html.Div([
            html.Div(f"❌ Error: {str(e)}", className="error-title")
        ], className="error-card")
        return None, None, error_msg
    
@app.callback(
    [Output('dashboard-missing-values-chart', 'figure'),
     Output('dashboard-missing-summary', 'children')],
    [Input('drop-threshold', 'value'),
     Input('stored-data', 'data')],
    prevent_initial_call=True
)
def update_dashboard_with_threshold(threshold_percent, data_json):
    """Update dashboard based on missing values threshold"""
    if data_json is None:
        return go.Figure(), html.Div("No data available", className="info-text")
    
    df = pd.read_json(io.StringIO(data_json), orient='split')
    threshold = threshold_percent / 100  # Convert % to decimal
    
    # Calculate which columns would be dropped
    null_percentage = (df.isnull().sum() / len(df)) * 100
    columns_to_drop = null_percentage[null_percentage > threshold_percent].index.tolist()
    columns_to_keep = null_percentage[null_percentage <= threshold_percent].index.tolist()
    
    # Create visualization
    fig = go.Figure()
    
    # Add bars for all columns
    fig.add_trace(go.Bar(
        x=null_percentage.index,
        y=null_percentage.values,
        name='Missing %',
        marker_color=['red' if pct > threshold_percent else 'blue' 
                     for pct in null_percentage.values]
    ))
    
    # Add threshold line
    fig.add_hline(y=threshold_percent, 
                  line_dash="dash", 
                  line_color="red",
                  annotation_text=f"Threshold: {threshold_percent}%",
                  annotation_position="top right")
    
    fig.update_layout(
        title=f"Missing Values by Column (Threshold: {threshold_percent}%)",
        xaxis_title="Columns",
        yaxis_title="Missing Values (%)",
        height=400
    )
    
    # Create summary text
    summary = html.Div([
        html.H5(f"Threshold Analysis: {threshold_percent}%", className="card-title"),
        html.P(f"Columns that would be dropped: {len(columns_to_drop)}", className="info-text"),
        html.P(f"Columns that would be kept: {len(columns_to_keep)}", className="info-text"),
        html.P(f"Total null values: {df.isnull().sum().sum():,}", className="info-text"),
        html.P(f"Overall null percentage: {(df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100):.1f}%", className="info-text")
    ])
    
    return fig, summary


@app.callback(
    Output('datetime-chart', 'figure'),
    [Input('date-col-selector', 'value'),
     Input('time-unit', 'value')],
    [State('stored-cleaned-data', 'data')]
)
def update_datetime_chart(date_col, time_unit, data_json):
    """Update datetime chart based on selected column and time unit"""
    if not data_json or not date_col:
        return go.Figure()
    
    df = pd.read_json(io.StringIO(data_json), orient='split')
    
    if date_col not in df.columns:
        return go.Figure()
    
    try:
        # Extract time component based on selected unit
        if time_unit == 'year':
            time_data = df[date_col].dt.year
            title = f"Distribution by Year - {date_col}"
        elif time_unit == 'month':
            time_data = df[date_col].dt.month
            title = f"Distribution by Month - {date_col}"
        elif time_unit == 'day':
            time_data = df[date_col].dt.day
            title = f"Distribution by Day - {date_col}"
        elif time_unit == 'hour':
            time_data = df[date_col].dt.hour
            title = f"Distribution by Hour - {date_col}"
        elif time_unit == 'minute':
            time_data = df[date_col].dt.minute
            title = f"Distribution by Minute - {date_col}"
        elif time_unit == 'dayofweek':
            time_data = df[date_col].dt.dayofweek
            title = f"Distribution by Day of Week - {date_col}"
        else:
            time_data = df[date_col].dt.month
            title = f"Distribution by Month - {date_col}"
        
        # Count occurrences
        counts = time_data.value_counts().sort_index()
        
        # Create bar chart
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=counts.index,
            y=counts.values,
            name='Count',
            marker_color='#007bff'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title=time_unit.capitalize(),
            yaxis_title='Count',
            plot_bgcolor='white',
            paper_bgcolor='white',
            showlegend=False
        )
        
        return fig
        
    except Exception as e:
        # If datetime operations fail, return empty figure
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error: {str(e)[:100]}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False
        )
        return fig

@app.callback(
    Output('pipeline-summary', 'children'),
    [Input('stored-reports', 'data')]
)
def update_pipeline_summary(reports_data):
    """Update pipeline summary in Export tab"""
    if reports_data is None or 'final' not in reports_data:
        return html.Div([
            html.H5("Pipeline Summary", className="card-title"),
            html.P("No pipeline results available yet.", className="info-text"),
            html.P("Run the pipeline first to see summary.", className="info-text")
        ], className="empty-state")
    
    final = reports_data['final']
    
    summary_cards = []
    
    # Original vs Cleaned comparison
    summary_cards.append(html.Div([
        html.H5("Data Transformation", className="card-title"),
        html.Div([
            html.Div([
                html.H6("Original", className="comparison-label"),
                html.P(f"Rows: {final['original_shape'][0]:,}", className="info-text"),
                html.P(f"Columns: {final['original_shape'][1]:,}", className="info-text")
            ], className="comparison-col"),
            
            html.Div([
                html.H6("→", className="comparison-arrow")
            ], className="comparison-arrow-col"),
            
            html.Div([
                html.H6("Cleaned", className="comparison-label success"),
                html.P(f"Rows: {final['cleaned_shape'][0]:,}", className="info-text"),
                html.P(f"Columns: {final['cleaned_shape'][1]:,}", className="info-text")
            ], className="comparison-col"),
        ], className="comparison-row")
    ], className="summary-card"))
    
    # Columns removed
    if final['columns_removed']:
        removed_count = len(final['columns_removed'])
        removed_text = ', '.join(final['columns_removed'][:3])
        if removed_count > 3:
            removed_text += f'... and {removed_count - 3} more'
        
        summary_cards.append(html.Div([
            html.H5("Columns Removed", className="card-title"),
            html.P(f"{removed_count} columns were removed", className="info-text"),
            html.P(removed_text, className="columns-list")
        ], className="summary-card"))
    
    # Configuration used
    if 'config' in final:
        config = final['config']
        summary_cards.append(html.Div([
            html.H5("Configuration Used", className="card-title"),
            html.P(f"Outlier Factor: {config.get('outlier_factor', 'N/A')}", className="info-text"),
            html.P(f"Drop Threshold: {config.get('drop_threshold', 'N/A')*100:.0f}%", className="info-text"),
            html.P(f"Impute Strategy: {config.get('impute_strategy', 'N/A')}", className="info-text")
        ], className="summary-card"))
    
    # Pipeline steps
    if 'pipeline_steps' in final:
        steps = final['pipeline_steps']
        steps_list = [html.Li(step, className="step-item") for step in steps]
        summary_cards.append(html.Div([
            html.H5("Pipeline Steps", className="card-title"),
            html.Ul(steps_list, className="steps-list")
        ], className="summary-card"))
    
    return html.Div(summary_cards, className="summary-container")

# Add callback for value counts chart
@app.callback(
    Output('value-counts-chart', 'figure', allow_duplicate=True),
    [Input('cat-col-selector', 'value')],
    [State('stored-cleaned-data', 'data')],
    prevent_initial_call=True
)
def update_value_counts_chart(cat_col, data_json):
    """Update value counts pie chart for categorical column"""
    if not data_json or not cat_col:
        return go.Figure()
    
    df = pd.read_json(io.StringIO(data_json), orient='split')
    
    if cat_col not in df.columns:
        return go.Figure()
    
    try:
        # Create value counts for categorical column
        value_counts = df[cat_col].value_counts().reset_index()
        value_counts.columns = ['category', 'count']
        
        # Take top 10 categories
        top_categories = value_counts.head(10)
        
        # Create pie chart
        fig = px.pie(top_categories, values='count', names='category',
                    title=f"Top 10 Categories in {cat_col}",
                    color_discrete_sequence=px.colors.qualitative.Set3)
        
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            showlegend=True
        )
        
        return fig
        
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error: {str(e)[:100]}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False
        )
        return fig

# Enhanced Dashboard Callbacks
@app.callback(
    [Output('enhanced-visualization', 'figure'),
     Output('enhanced-statistics', 'children')],
    [Input('enhanced-col-selector', 'value'),
     Input('enhanced-vis-type', 'value')],
    [State('stored-cleaned-data', 'data')]
)
def update_enhanced_visualization(column, vis_type, data_json):
    """Update enhanced dashboard visualization"""
    from enhanced_dashboard import create_column_visualization, create_column_statistics
    
    if not data_json or not column:
        return go.Figure(), ""
    
    df = pd.read_json(io.StringIO(data_json), orient='split')
    
    if column not in df.columns:
        return go.Figure(), html.P("Selected column not found in dataset", 
                                  className="error-text")
    
    # Create visualization
    fig = create_column_visualization(df, column, vis_type)
    
    # Create statistics
    stats = create_column_statistics(df, column)
    
    return fig, stats

@app.callback(
    Output('enhanced-csv-download', 'data'),
    [Input('enhanced-download-csv', 'n_clicks')],
    [State('stored-cleaned-data', 'data')]
)
def download_enhanced_csv(n_clicks, data_json):
    """Download CSV from enhanced dashboard"""
    if n_clicks is None or data_json is None:
        return None
    
    df = pd.read_json(io.StringIO(data_json), orient='split')
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"enhanced_data_export_{timestamp}.csv"
    
    return dcc.send_data_frame(df.to_csv, filename, index=False)

@app.callback(
    Output('enhanced-report-download', 'data'),
    [Input('enhanced-download-report', 'n_clicks')],
    [State('stored-cleaned-data', 'data')]
)
def download_enhanced_report(n_clicks, data_json):
    """Download comprehensive report from enhanced dashboard"""
    from enhanced_dashboard import create_enhanced_data_summary
    
    if n_clicks is None or data_json is None:
        return None
    
    df = pd.read_json(io.StringIO(data_json), orient='split')
    
    # Create comprehensive report
    summary = create_enhanced_data_summary(df)
    
    # Convert report to text
    report_text = f"""
    ENHANCED DATA ANALYSIS REPORT
    =============================
    Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    
    BASIC INFORMATION
    -----------------
    Total Rows: {summary['basic_info']['total_rows']:,}
    Total Columns: {summary['basic_info']['total_columns']:,}
    Memory Usage: {summary['basic_info']['memory_usage_mb']:.2f} MB
    
    DATA TYPE DISTRIBUTION
    ----------------------
    Numerical Columns: {summary['data_types']['numerical']:,}
    Categorical Columns: {summary['data_types']['categorical']:,}
    Date/Time Columns: {summary['data_types']['datetime']:,}
    Boolean Columns: {summary['data_types']['boolean']:,}
    
    MISSING VALUES ANALYSIS
    -----------------------
    Total Null Values: {summary['missing_values']['total_nulls']:,}
    Columns with Nulls: {summary['missing_values']['columns_with_nulls']:,}
    Overall Null Percentage: {summary['missing_values']['null_percentage']:.2f}%
    
    COLUMN SAMPLES (First 10 columns)
    --------------------------------
    """
    
    for col, sample in summary['column_samples'].items():
        report_text += f"{col}: {sample}\n"
    
    # Add column list
    report_text += f"\nALL COLUMNS ({len(df.columns)} total):\n"
    report_text += "-" * 40 + "\n"
    for i, col in enumerate(df.columns, 1):
        report_text += f"{i:3}. {col}\n"
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"enhanced_report_{timestamp}.txt"
    
    return dict(content=report_text, filename=filename)

# Store enhanced summary data
@app.callback(
    Output('stored-enhanced-summary', 'data'),
    [Input('stored-cleaned-data', 'data')]
)
def update_enhanced_summary(data_json):
    """Store enhanced summary data"""
    if data_json is None:
        return None
    
    from enhanced_dashboard import create_enhanced_data_summary
    df = pd.read_json(io.StringIO(data_json), orient='split')
    summary = create_enhanced_data_summary(df)
    
    return summary

# ===== NEW CALLBACKS FOR PROCESSING OPTIONS =====

# Update outlier factor display
@app.callback(
    Output('outlier-factor-value', 'children'),
    Input('outlier-factor', 'value')
)
def update_outlier_display(value):
    return f"{value}"

# Update drop threshold display
@app.callback(
    Output('drop-threshold-value', 'children'),
    Input('drop-threshold', 'value')
)
def update_threshold_display(value):
    return f"{value}%"

# Update preview rows display
@app.callback(
    Output('preview-rows-value', 'children'),
    Input('preview-rows', 'value')
)
def update_preview_display(value):
    return f"{value}"

# Preset button callbacks
@app.callback(
    [Output('outlier-factor', 'value'),
     Output('drop-threshold', 'value'),
     Output('impute-strategy', 'value'),
     Output('preview-rows', 'value'),
     Output('preset-balanced', 'className'),
     Output('preset-aggressive', 'className'),
     Output('preset-conservative', 'className'),
     Output('preset-custom', 'className')],
    [Input('preset-balanced', 'n_clicks'),
     Input('preset-aggressive', 'n_clicks'),
     Input('preset-conservative', 'n_clicks'),
     Input('reset-options', 'n_clicks')],
    [State('outlier-factor', 'value'),
     State('drop-threshold', 'value'),
     State('impute-strategy', 'value'),
     State('preview-rows', 'value')]
)
def apply_presets(balanced_clicks, aggressive_clicks, conservative_clicks, reset_clicks, 
                  current_outlier, current_threshold, current_strategy, current_preview):
    ctx = dash.callback_context
    
    if not ctx.triggered:
        return dash.no_update
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # Reset to defaults
    if button_id == 'reset-options':
        return OUTLIER_FACTOR, DROP_THRESHOLD * 100, IMPUTE_STRATEGY, PREVIEW_ROWS, "preset-button", "preset-button", "preset-button", "preset-button active"
    
    # Balanced preset (default)
    if button_id == 'preset-balanced':
        return 1.5, 50, 'median', 10, "preset-button active", "preset-button", "preset-button", "preset-button"
    
    # Aggressive preset (more cleaning)
    elif button_id == 'preset-aggressive':
        return 2.0, 30, 'most_frequent', 5, "preset-button", "preset-button active", "preset-button", "preset-button"
    
    # Conservative preset (less cleaning)
    elif button_id == 'preset-conservative':
        return 1.2, 70, 'mean', 20, "preset-button", "preset-button", "preset-button active", "preset-button"
    
    # If custom values are changed, mark as custom
    return dash.no_update

# Detect custom changes and update preset button state
@app.callback(
    [Output('preset-balanced', 'className', allow_duplicate=True),
     Output('preset-aggressive', 'className', allow_duplicate=True),
     Output('preset-conservative', 'className', allow_duplicate=True),
     Output('preset-custom', 'className', allow_duplicate=True)],
    [Input('outlier-factor', 'value'),
     Input('drop-threshold', 'value'),
     Input('impute-strategy', 'value'),
     Input('preview-rows', 'value')],
    prevent_initial_call=True
)
def detect_custom_changes(outlier_value, threshold_value, strategy_value, preview_value):
    # Check if current values match any preset
    if (outlier_value == 1.5 and threshold_value == 50 and strategy_value == 'median' and preview_value == 10):
        return "preset-button active", "preset-button", "preset-button", "preset-button"
    elif (outlier_value == 2.0 and threshold_value == 30 and strategy_value == 'most_frequent' and preview_value == 5):
        return "preset-button", "preset-button active", "preset-button", "preset-button"
    elif (outlier_value == 1.2 and threshold_value == 70 and strategy_value == 'mean' and preview_value == 20):
        return "preset-button", "preset-button", "preset-button active", "preset-button"
    else:
        return "preset-button", "preset-button", "preset-button", "preset-button active"

debug_mode = os.environ.get('DEBUG', 'False').lower() == 'true'
if __name__ == '__main__':
    app.run(debug=debug_mode, host='0.0.0.0', port=8050)