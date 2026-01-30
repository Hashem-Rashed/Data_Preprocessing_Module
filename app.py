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
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "Modern Data Analysis Pipeline"

# Initialize pipeline manager
manager = PipelineManager()

# Global variable to store data
global_data = None
cleaned_data = None
reports = {}

# Custom CSS
app.css.append_css({
    'external_url': 'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css'
})

# App layout
app.layout = html.Div([
    # Header
    html.Div([
        html.H1("📊 Modern Data Analysis Pipeline", style={'textAlign': 'center', 'marginBottom': '20px'}),
        html.P("Automated data cleaning and visualization dashboard", 
               style={'textAlign': 'center', 'color': '#6c757d'}),
    ], style={'margin': '30px 0'}),
    
    # Tabs for different sections
    dcc.Tabs([
        # Tab 1: Data Upload & Processing
        dcc.Tab(label='📁 Data Processing', children=[
            html.Div([
                html.Div([
                    html.H3("Upload Data", style={'marginBottom': '20px'}),
                    
                    # File Upload
                    dcc.Upload(
                        id='upload-data',
                        children=html.Div([
                            'Drag and Drop or ',
                            html.A('Select Files')
                        ]),
                        style={
                            'width': '100%',
                            'height': '100px',
                            'lineHeight': '100px',
                            'borderWidth': '2px',
                            'borderStyle': 'dashed',
                            'borderRadius': '5px',
                            'textAlign': 'center',
                            'margin': '10px'
                        },
                        multiple=False
                    ),
                    
                    # Large file processing button (initially hidden)
                    html.Button("Process Large File (Chunked)", 
                               id="process-large-file-btn",
                               style={
                                   'width': '100%',
                                   'marginTop': '10px',
                                   'padding': '10px',
                                   'backgroundColor': '#fd7e14',
                                   'color': 'white',
                                   'border': 'none',
                                   'borderRadius': '5px',
                                   'cursor': 'pointer',
                                   'display': 'none'
                               }),
                    
                    # Continue with large file button
                    html.Button("Continue Anyway with Large File", 
                               id="continue-large-file-btn",
                               style={
                                   'width': '100%',
                                   'marginTop': '10px',
                                   'padding': '10px',
                                   'backgroundColor': '#6c757d',
                                   'color': 'white',
                                   'border': 'none',
                                   'borderRadius': '5px',
                                   'cursor': 'pointer',
                                   'display': 'none'
                               }),
                    
                    # Or use sample data
                    html.Button("Use Sample Data", 
                               id="sample-data-btn",
                               style={
                                   'width': '100%',
                                   'marginTop': '20px',
                                   'padding': '10px',
                                   'backgroundColor': '#6c757d',
                                   'color': 'white',
                                   'border': 'none',
                                   'borderRadius': '5px',
                                   'cursor': 'pointer'
                               }),
                    
                    # Current file info
                    html.Div(id='file-info', style={'marginTop': '20px', 'padding': '15px', 
                                                    'backgroundColor': '#f8f9fa', 'borderRadius': '5px'}),
                    
                    html.Hr(),
                    
                    # Processing Options
                    html.H4("Processing Options", style={'marginTop': '30px'}),
                    
                    html.Div([
                        html.Div([
                            html.Label("Outlier Factor (IQR)"),
                            dcc.Slider(
                                id='outlier-factor',
                                min=1,
                                max=3,
                                step=0.1,
                                value=OUTLIER_FACTOR,
                                marks={i: str(i) for i in [1, 1.5, 2, 2.5, 3]}
                            )
                        ], style={'width': '48%', 'display': 'inline-block'}),
                        
                        html.Div([
                            html.Label("Missing Values Threshold (%)"),
                            dcc.Slider(
                                id='drop-threshold',
                                min=0,
                                max=100,
                                step=5,
                                value=DROP_THRESHOLD * 100,
                                marks={i: f'{i}%' for i in [0, 25, 50, 75, 100]}
                            )
                        ], style={'width': '48%', 'display': 'inline-block', 'float': 'right'})
                    ]),
                    
                    html.Div([
                        html.Div([
                            html.Label("Imputation Strategy"),
                            dcc.Dropdown(
                                id='impute-strategy',
                                options=[
                                    {'label': 'Median', 'value': 'median'},
                                    {'label': 'Mean', 'value': 'mean'},
                                    {'label': 'Most Frequent', 'value': 'most_frequent'}
                                ],
                                value=IMPUTE_STRATEGY
                            )
                        ], style={'width': '48%', 'display': 'inline-block'}),
                        
                        html.Div([
                            html.Label("Preview Rows"),
                            dcc.Slider(
                                id='preview-rows',
                                min=5,
                                max=50,
                                step=5,
                                value=PREVIEW_ROWS,
                                marks={i: str(i) for i in [5, 10, 20, 30, 50]}
                            )
                        ], style={'width': '48%', 'display': 'inline-block', 'float': 'right'})
                    ], style={'marginTop': '20px'}),
                    
                    # Process Buttons
                    html.Div([
                        html.Button("Run Full Pipeline", 
                                   id="run-pipeline-btn",
                                   style={
                                       'width': '48%',
                                       'padding': '15px',
                                       'backgroundColor': '#007bff',
                                       'color': 'white',
                                       'border': 'none',
                                       'borderRadius': '5px',
                                       'cursor': 'pointer',
                                       'fontSize': '16px',
                                       'marginTop': '30px'
                                   }),
                        
                        html.Button("Step-by-Step Processing", 
                                   id="step-by-step-btn",
                                   style={
                                       'width': '48%',
                                       'padding': '15px',
                                       'backgroundColor': '#17a2b8',
                                       'color': 'white',
                                       'border': 'none',
                                       'borderRadius': '5px',
                                       'cursor': 'pointer',
                                       'fontSize': '16px',
                                       'marginTop': '30px',
                                       'float': 'right'
                                   })
                    ]),
                    
                    # Progress Indicator
                    dcc.Loading(
                        id="loading",
                        type="circle",
                        children=html.Div(id="loading-output")
                    ),
                    
                ], style={'width': '40%', 'display': 'inline-block', 'verticalAlign': 'top', 
                         'paddingRight': '30px', 'borderRight': '1px solid #ddd'}),
                
                # Results Column
                html.Div([
                    html.H3("Data Preview & Results", style={'marginBottom': '20px'}),
                    
                    # Tab for different previews
                    dcc.Tabs([
                        dcc.Tab(label='Raw Data', children=[
                            html.Div(id='raw-data-preview', style={'marginTop': '20px'})
                        ]),
                        
                        dcc.Tab(label='Cleaned Data', children=[
                            html.Div(id='cleaned-data-preview', style={'marginTop': '20px'})
                        ]),
                        
                        dcc.Tab(label='Data Quality Report', children=[
                            html.Div(id='quality-report', style={'marginTop': '20px'})
                        ])
                    ]),
                    
                    # Statistics Cards
                    html.Div(id='statistics-cards', style={'marginTop': '30px'}),
                    
                ], style={'width': '58%', 'display': 'inline-block', 'verticalAlign': 'top', 
                         'paddingLeft': '30px'})
            ], style={'display': 'flex'})
        ]),
        
        # Tab 2: Dashboard
        dcc.Tab(label='📈 Dashboard', children=[
            html.Div(id='dashboard-content', style={'marginTop': '20px'})
        ]),
        
        # Tab 3: Export
        dcc.Tab(label='💾 Export', children=[
            html.Div([
                html.Div([
                    html.H3("Export Results", style={'marginBottom': '30px'}),
                    
                    html.Div([
                        html.H5("Download Cleaned Data"),
                        html.P("Export the processed dataset to CSV format"),
                        html.Button("Download CSV", 
                                   id="download-csv-btn",
                                   style={
                                       'padding': '10px 20px',
                                       'backgroundColor': '#28a745',
                                       'color': 'white',
                                       'border': 'none',
                                       'borderRadius': '5px',
                                       'cursor': 'pointer',
                                       'marginTop': '10px'
                                   })
                    ], style={
                        'backgroundColor': 'white',
                        'padding': '20px',
                        'borderRadius': '5px',
                        'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                        'marginBottom': '20px'
                    }),
                    
                    html.Div([
                        html.H5("Quality Report"),
                        html.P("Download the detailed data quality report"),
                        html.Button("Download Report", 
                                   id="download-report-btn",
                                   style={
                                       'padding': '10px 20px',
                                       'backgroundColor': '#17a2b8',
                                       'color': 'white',
                                       'border': 'none',
                                       'borderRadius': '5px',
                                       'cursor': 'pointer',
                                       'marginTop': '10px'
                                   })
                    ], style={
                        'backgroundColor': 'white',
                        'padding': '20px',
                        'borderRadius': '5px',
                        'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                        'marginBottom': '20px'
                    }),
                    
                    # Download components
                    dcc.Download(id="download-data"),
                    dcc.Download(id="download-report"),
                    
                    html.Hr(),
                    
                    # Configuration Summary
                    html.H4("Current Configuration", style={'marginTop': '30px'}),
                    html.Div(id='config-summary')
                    
                ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),
                
                html.Div([
                    html.H3("Pipeline Summary", style={'marginBottom': '30px'}),
                    html.Div(id='pipeline-summary')
                ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 
                         'float': 'right'})
            ])
        ])
    ]),
    
    # Store for data
    dcc.Store(id='stored-data'),
    dcc.Store(id='stored-cleaned-data'),
    dcc.Store(id='stored-reports'),
    
    # Interval for updates
    dcc.Interval(id='interval-component', interval=1000, n_intervals=0)
], style={'padding': '20px', 'maxWidth': '1400px', 'margin': '0 auto'})

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
        return "No data loaded yet", None, {'display': 'none'}, {'display': 'none'}
    
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    global global_data
    
    # Default button styles
    large_file_btn_style = {
        'width': '100%',
        'marginTop': '10px',
        'padding': '10px',
        'backgroundColor': '#fd7e14',
        'color': 'white',
        'border': 'none',
        'borderRadius': '5px',
        'cursor': 'pointer',
        'display': 'none'
    }
    
    continue_btn_style = {
        'width': '100%',
        'marginTop': '10px',
        'padding': '10px',
        'backgroundColor': '#6c757d',
        'color': 'white',
        'border': 'none',
        'borderRadius': '5px',
        'cursor': 'pointer',
        'display': 'none'
    }
    
    if trigger_id == 'upload-data' and contents:
        try:
            content_type, content_string = contents.split(',')
            decoded = base64.b64decode(content_string)
            
            # Check file size
            file_size = len(decoded) / (1024 * 1024)  # Size in MB
            
            # For large files (>50MB), show warning and options
            if file_size > 50:
                warning_msg = html.Div([
                    html.H5(f"⚠️ Large File Detected: {filename}"),
                    html.P(f"Size: {file_size:.1f} MB"),
                    html.P("This file is large. For better performance:"),
                ])
                
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
                    html.H5(f"❌ Unsupported file format: {filename}"),
                    html.P("Supported formats: CSV, Excel (.xlsx), JSON")
                ], style={'color': '#721c24', 'backgroundColor': '#f8d7da', 
                         'padding': '15px', 'borderRadius': '5px'}), None, large_file_btn_style, continue_btn_style
            
            global_data = df
            info = [
                html.H5(f"✅ File Loaded: {filename}"),
                html.P(f"Shape: {df.shape[0]:,} rows × {df.shape[1]:,} columns"),
                html.P(f"Size: {file_size:.1f} MB"),
                html.P(f"Memory usage: {df.memory_usage(deep=True).sum() / (1024**2):.2f} MB"),
                html.P(f"Columns preview: {', '.join(df.columns[:5])}..." if len(df.columns) > 5 else f"Columns: {', '.join(df.columns)}")
            ]
            return info, df.to_json(date_format='iso', orient='split'), large_file_btn_style, continue_btn_style
            
        except Exception as e:
            error_msg = html.Div([
                html.H5(f"❌ Error loading file: {filename}"),
                html.P(f"Error: {str(e)}"),
                html.P("Make sure the file format is correct and not corrupted.")
            ], style={'color': '#721c24', 'backgroundColor': '#f8d7da', 
                     'padding': '15px', 'borderRadius': '5px', 'border': '1px solid #f5c6cb'})
            return error_msg, None, large_file_btn_style, continue_btn_style
    
    elif trigger_id == 'sample-data-btn':
        df = manager._create_sample_data()
        global_data = df
        info = [
            html.H5("📊 Sample Data Generated"),
            html.P(f"Shape: {df.shape[0]:,} rows × {df.shape[1]:,} columns"),
            html.P(f"Memory: {df.memory_usage(deep=True).sum() / (1024**2):.2f} MB"),
            html.P("Note: This is generated sample data for demonstration.")
        ]
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
                    html.H5(f"❌ Unsupported file format: {filename}")
                ]), None, large_file_btn_style, continue_btn_style
            
            global_data = df
            info = [
                html.H5(f"✅ Large File Loaded: {filename}"),
                html.P(f"Shape: {df.shape[0]:,} rows × {df.shape[1]:,} columns"),
                html.P(f"Size: {file_size:.1f} MB"),
                html.P("Processing may be slower due to file size."),
                html.P(f"Columns preview: {', '.join(df.columns[:5])}..." if len(df.columns) > 5 else f"Columns: {', '.join(df.columns)}")
            ]
            return info, df.to_json(date_format='iso', orient='split'), large_file_btn_style, continue_btn_style
            
        except Exception as e:
            error_msg = html.Div([
                html.H5(f"❌ Error loading large file"),
                html.P(f"Error: {str(e)}"),
                html.P("Try using the 'Process Large File (Chunked)' option.")
            ], style={'color': '#721c24', 'backgroundColor': '#f8d7da', 
                     'padding': '15px', 'borderRadius': '5px'})
            return error_msg, None, large_file_btn_style, continue_btn_style
    
    return "No data loaded yet", None, large_file_btn_style, continue_btn_style

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
            html.Div("✅ Pipeline completed!", 
                    style={'color': 'green', 'fontWeight': 'bold'}),
            html.Div(f"Cleaned data: {cleaned_df.shape[0]:,} rows × {cleaned_df.shape[1]:,} columns")
        ], style={
            'padding': '15px',
            'backgroundColor': '#d4edda',
            'border': '1px solid #c3e6cb',
            'borderRadius': '5px',
            'color': '#155724'
        })
        
        return cleaned_df.to_json(date_format='iso', orient='split'), reports, success_msg
        
    except Exception as e:
        error_msg = html.Div([
            html.Div(f"❌ Error: {str(e)}", 
                    style={'color': 'red', 'fontWeight': 'bold'})
        ], style={
            'padding': '15px',
            'backgroundColor': '#f8d7da',
            'border': '1px solid #f5c6cb',
            'borderRadius': '5px',
            'color': '#721c24'
        })
        return None, None, error_msg

@app.callback(
    Output('raw-data-preview', 'children'),
    [Input('stored-data', 'data'),
     Input('preview-rows', 'value')]
)
def preview_raw_data(data_json, preview_rows):
    if data_json is None:
        return "No data loaded yet."
    
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
        }
    )
    
    return html.Div([
        html.P(f"Showing first {preview_rows} rows of {len(df):,} total"),
        table
    ])

@app.callback(
    Output('cleaned-data-preview', 'children'),
    [Input('stored-cleaned-data', 'data'),
     Input('preview-rows', 'value')]
)
def preview_cleaned_data(data_json, preview_rows):
    if data_json is None:
        return "No cleaned data available. Run the pipeline first."
    
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
        }
    )
    
    return html.Div([
        html.P(f"Showing first {preview_rows} rows of {len(df):,} total"),
        table
    ])

@app.callback(
    Output('dashboard-content', 'children'),
    [Input('stored-cleaned-data', 'data')]
)
def update_dashboard(data_json):
    if data_json is None:
        return html.Div([
            html.H4("No data available for dashboard"),
            html.P("Please run the data pipeline first.")
        ])
    
    df = pd.read_json(io.StringIO(data_json), orient='split')
    return create_dashboard(df)

@app.callback(
    Output('quality-report', 'children'),
    [Input('stored-reports', 'data')]
)
def show_quality_report(reports_data):
    if reports_data is None:
        return "No quality report available yet."
    
    reports = reports_data
    
    cards = []
    
    # Final report card
    if 'final' in reports:
        final = reports['final']
        cards.append(html.Div([
            html.H5("Pipeline Summary"),
            html.P(f"Original: {final['original_shape'][0]:,} × {final['original_shape'][1]:,}"),
            html.P(f"Cleaned: {final['cleaned_shape'][0]:,} × {final['cleaned_shape'][1]:,}"),
            html.P(f"Columns Removed: {len(final['columns_removed'])}")
        ], style={
            'backgroundColor': 'white',
            'padding': '20px',
            'borderRadius': '5px',
            'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
            'marginBottom': '20px'
        }))
    
    # Missing values card
    if 'missing_values' in reports:
        null_report = reports['missing_values']
        
        # Handle different types of null_report
        if isinstance(null_report, dict):
            total_nulls = null_report.get('total_nulls', 0)
            total_rows = null_report.get('total_rows', 1)
            total_columns = null_report.get('total_columns', 1)
            columns_to_drop = null_report.get('columns_to_drop', [])
            
            total_cells = total_rows * total_columns
            null_percentage = (total_nulls / total_cells * 100) if total_cells > 0 else 0
            
            cards.append(html.Div([
                html.H5("Missing Values"),
                html.P(f"Total Nulls: {total_nulls:,}"),
                html.P(f"Null Percentage: {null_percentage:.1f}%"),
                html.P(f"Columns to Drop: {len(columns_to_drop)}")
            ], style={
                'backgroundColor': 'white',
                'padding': '20px',
                'borderRadius': '5px',
                'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                'marginBottom': '20px'
            }))
        else:
            # Handle error case
            cards.append(html.Div([
                html.H5("Missing Values"),
                html.P(f"Report type: {type(null_report).__name__}")
            ], style={
                'backgroundColor': '#f8f9fa',
                'padding': '20px',
                'borderRadius': '5px',
                'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                'marginBottom': '20px'
            }))
    
    # Outliers card - FIXED
    if 'outliers' in reports:
        outlier_report = reports['outliers']
        
        # Handle string case (error message)
        if isinstance(outlier_report, str):
            if outlier_report == 'outliers':
                cards.append(html.Div([
                    html.H5("Outliers"),
                    html.P("No numerical columns found for outlier analysis")
                ], style={
                    'backgroundColor': '#f8f9fa',
                    'padding': '20px',
                    'borderRadius': '5px',
                    'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                    'marginBottom': '20px'
                }))
            else:
                cards.append(html.Div([
                    html.H5("Outliers"),
                    html.P(f"Note: {outlier_report}")
                ], style={
                    'backgroundColor': '#f8f9fa',
                    'padding': '20px',
                    'borderRadius': '5px',
                    'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                    'marginBottom': '20px'
                }))
        elif isinstance(outlier_report, dict):
            # Check if it's an error report
            if 'error' in outlier_report:
                error_msg = outlier_report.get('error', 'Unknown error')
                cards.append(html.Div([
                    html.H5("Outliers"),
                    html.P(f"Error: {error_msg}")
                ], style={
                    'backgroundColor': '#f8f9fa',
                    'padding': '20px',
                    'borderRadius': '5px',
                    'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                    'marginBottom': '20px'
                }))
            else:
                # Calculate total outliers safely
                total_outliers = 0
                columns_with_outliers = 0
                
                for key, value in outlier_report.items():
                    # Skip summary/metadata entries
                    if key.startswith('_') or key == 'error':
                        continue
                    
                    # Check if value is a dictionary with outliers data
                    if isinstance(value, dict):
                        outliers_data = value.get('outliers')
                        if isinstance(outliers_data, dict):
                            count = outliers_data.get('count', 0)
                            total_outliers += count
                            if count > 0:
                                columns_with_outliers += 1
                
                cards.append(html.Div([
                    html.H5("Outliers"),
                    html.P(f"Total Outliers: {total_outliers:,}"),
                    html.P(f"Columns with outliers: {columns_with_outliers}")
                ], style={
                    'backgroundColor': 'white',
                    'padding': '20px',
                    'borderRadius': '5px',
                    'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                    'marginBottom': '20px'
                }))
        else:
            # Unknown format
            cards.append(html.Div([
                html.H5("Outliers"),
                html.P(f"Report type: {type(outlier_report).__name__}")
            ], style={
                'backgroundColor': '#f8f9fa',
                'padding': '20px',
                'borderRadius': '5px',
                'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                'marginBottom': '20px'
            }))
    
    return html.Div(cards)

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
            html.H5("Raw Data", style={'color': '#17a2b8'}),
            html.H3(f"{df_raw.shape[0]:,}", style={'margin': '10px 0'}),
            html.P("Rows", style={'color': '#6c757d'}),
            html.H3(f"{df_raw.shape[1]}", style={'margin': '10px 0'}),
            html.P("Columns", style={'color': '#6c757d'}),
        ], style={
            'backgroundColor': 'white',
            'padding': '20px',
            'borderRadius': '5px',
            'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
            'textAlign': 'center',
            'width': '23%',
            'display': 'inline-block',
            'marginRight': '2%'
        }))
    
    if cleaned_data_json:
        df_clean = pd.read_json(io.StringIO(cleaned_data_json), orient='split')
        cards.append(html.Div([
            html.H5("Cleaned Data", style={'color': '#28a745'}),
            html.H3(f"{df_clean.shape[0]:,}", style={'margin': '10px 0'}),
            html.P("Rows", style={'color': '#6c757d'}),
            html.H3(f"{df_clean.shape[1]}", style={'margin': '10px 0'}),
            html.P("Columns", style={'color': '#6c757d'}),
        ], style={
            'backgroundColor': 'white',
            'padding': '20px',
            'borderRadius': '5px',
            'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
            'textAlign': 'center',
            'width': '23%',
            'display': 'inline-block',
            'marginRight': '2%'
        }))
    
    if raw_data_json:
        df_raw = pd.read_json(io.StringIO(raw_data_json), orient='split')
        null_count = df_raw.isnull().sum().sum()
        total_cells = df_raw.shape[0] * df_raw.shape[1]
        null_percentage = (null_count / total_cells * 100) if total_cells > 0 else 0
        
        cards.append(html.Div([
            html.H5("Missing Values", style={'color': '#ffc107'}),
            html.H3(f"{null_count:,}", style={'margin': '10px 0'}),
            html.P("Null Values", style={'color': '#6c757d'}),
            html.H3(f"{null_percentage:.1f}%", style={'margin': '10px 0'}),
            html.P("Percentage", style={'color': '#6c757d'}),
        ], style={
            'backgroundColor': 'white',
            'padding': '20px',
            'borderRadius': '5px',
            'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
            'textAlign': 'center',
            'width': '23%',
            'display': 'inline-block',
            'marginRight': '2%'
        }))
    
    if raw_data_json:
        df_raw = pd.read_json(io.StringIO(raw_data_json), orient='split')
        num_cols = len(df_raw.select_dtypes(include=['int64', 'float64']).columns)
        cat_cols = len(df_raw.select_dtypes(include=['object', 'category']).columns)
        cards.append(html.Div([
            html.H5("Data Types", style={'color': '#007bff'}),
            html.H3(f"{num_cols}", style={'margin': '10px 0'}),
            html.P("Numerical", style={'color': '#6c757d'}),
            html.H3(f"{cat_cols}", style={'margin': '10px 0'}),
            html.P("Categorical", style={'color': '#6c757d'}),
        ], style={
            'backgroundColor': 'white',
            'padding': '20px',
            'borderRadius': '5px',
            'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
            'textAlign': 'center',
            'width': '23%',
            'display': 'inline-block'
        }))
    
    return html.Div(cards, style={'display': 'flex', 'justifyContent': 'space-between'})

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
        html.P(f"Outlier Factor (IQR): {outlier_factor}"),
        html.P(f"Drop Threshold: {drop_threshold}%"),
        html.P(f"Imputation Strategy: {impute_strategy}"),
        html.P(f"Preview Rows: {PREVIEW_ROWS}")
    ], style={
        'backgroundColor': '#f8f9fa',
        'padding': '15px',
        'borderRadius': '5px'
    })

# Dashboard callbacks
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
            html.Div("✅ Step-by-step processing completed!", 
                    style={'color': 'green', 'fontWeight': 'bold'}),
            html.Div(f"Cleaned data: {df.shape[0]:,} rows × {df.shape[1]:,} columns")
        ], style={
            'padding': '15px',
            'backgroundColor': '#d4edda',
            'border': '1px solid #c3e6cb',
            'borderRadius': '5px',
            'color': '#155724'
        })
        
        return df.to_json(date_format='iso', orient='split'), reports, success_msg
        
    except Exception as e:
        error_msg = html.Div([
            html.Div(f"❌ Error: {str(e)}", 
                    style={'color': 'red', 'fontWeight': 'bold'})
        ], style={
            'padding': '15px',
            'backgroundColor': '#f8d7da',
            'border': '1px solid #f5c6cb',
            'borderRadius': '5px',
            'color': '#721c24'
        })
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
        return go.Figure(), "No data available"
    
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
        html.H5(f"Threshold Analysis: {threshold_percent}%"),
        html.P(f"Columns that would be dropped: {len(columns_to_drop)}"),
        html.P(f"Columns that would be kept: {len(columns_to_keep)}"),
        html.P(f"Total null values: {df.isnull().sum().sum():,}"),
        html.P(f"Overall null percentage: {(df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100):.1f}%")
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
            html.H5("Pipeline Summary"),
            html.P("No pipeline results available yet."),
            html.P("Run the pipeline first to see summary.")
        ])
    
    final = reports_data['final']
    
    summary_cards = []
    
    # Original vs Cleaned comparison
    summary_cards.append(html.Div([
        html.H5("Data Transformation", style={'color': '#2c3e50'}),
        html.Div([
            html.Div([
                html.H6("Original", style={'color': '#6c757d', 'marginBottom': '10px'}),
                html.P(f"Rows: {final['original_shape'][0]:,}", style={'margin': '5px 0'}),
                html.P(f"Columns: {final['original_shape'][1]:,}", style={'margin': '5px 0'})
            ], style={'width': '45%', 'display': 'inline-block', 'textAlign': 'center'}),
            
            html.Div([
                html.H6("→", style={'color': '#007bff', 'fontSize': '24px', 'margin': '20px 10px'})
            ], style={'width': '10%', 'display': 'inline-block', 'textAlign': 'center'}),
            
            html.Div([
                html.H6("Cleaned", style={'color': '#28a745', 'marginBottom': '10px'}),
                html.P(f"Rows: {final['cleaned_shape'][0]:,}", style={'margin': '5px 0'}),
                html.P(f"Columns: {final['cleaned_shape'][1]:,}", style={'margin': '5px 0'})
            ], style={'width': '45%', 'display': 'inline-block', 'textAlign': 'center'})
        ])
    ], style={
        'backgroundColor': 'white',
        'padding': '20px',
        'borderRadius': '5px',
        'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
        'marginBottom': '15px'
    }))
    
    # Columns removed
    if final['columns_removed']:
        removed_count = len(final['columns_removed'])
        removed_text = ', '.join(final['columns_removed'][:3])
        if removed_count > 3:
            removed_text += f'... and {removed_count - 3} more'
        
        summary_cards.append(html.Div([
            html.H5("Columns Removed", style={'color': '#2c3e50'}),
            html.P(f"{removed_count} columns were removed", style={'marginBottom': '10px'}),
            html.P(removed_text, style={
                'fontSize': '12px',
                'color': '#6c757d',
                'backgroundColor': '#f8f9fa',
                'padding': '10px',
                'borderRadius': '3px',
                'wordBreak': 'break-word'
            })
        ], style={
            'backgroundColor': 'white',
            'padding': '20px',
            'borderRadius': '5px',
            'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
            'marginBottom': '15px'
        }))
    
    # Configuration used
    if 'config' in final:
        config = final['config']
        summary_cards.append(html.Div([
            html.H5("Configuration Used", style={'color': '#2c3e50'}),
            html.P(f"Outlier Factor: {config.get('outlier_factor', 'N/A')}"),
            html.P(f"Drop Threshold: {config.get('drop_threshold', 'N/A')*100:.0f}%"),
            html.P(f"Impute Strategy: {config.get('impute_strategy', 'N/A')}")
        ], style={
            'backgroundColor': 'white',
            'padding': '20px',
            'borderRadius': '5px',
            'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
            'marginBottom': '15px'
        }))
    
    # Pipeline steps
    if 'pipeline_steps' in final:
        steps = final['pipeline_steps']
        steps_list = [html.Li(step) for step in steps]
        summary_cards.append(html.Div([
            html.H5("Pipeline Steps", style={'color': '#2c3e50'}),
            html.Ul(steps_list, style={'paddingLeft': '20px'})
        ], style={
            'backgroundColor': 'white',
            'padding': '20px',
            'borderRadius': '5px',
            'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
        }))
    
    return html.Div(summary_cards)

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

debug_mode = os.environ.get('DEBUG', 'False').lower() == 'true'
if __name__ == '__main__':
    app.run(debug=debug_mode, host='0.0.0.0', port=8050)