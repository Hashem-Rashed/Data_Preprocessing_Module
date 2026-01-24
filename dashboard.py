"""
Dashboard components for data visualization with enhanced data type handling
"""
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import dcc, html
import json


def create_dashboard(df):
    """Create interactive dashboard from cleaned data"""
    
    if df.empty or df is None:
        return html.Div([
            html.H4("No data available for dashboard"),
            html.P("Please run the data pipeline first to generate cleaned data.")
        ], style={'textAlign': 'center', 'padding': '50px'})
    
    # Get column types with more granularity
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime', 'datetimetz']).columns.tolist()
    boolean_cols = df.select_dtypes(include=['bool']).columns.tolist()
    
    # Create enhanced data type report
    type_report = get_data_type_report(df)
    
    # Create tabs for different visualizations
    tabs = []
    
    # Tab 1: Enhanced Data Type Analysis
    tabs.append(dcc.Tab(label='📊 Data Types', children=[
        create_data_type_analysis_tab(df, type_report)
    ]))
    
    # Tab 2: Overview (enhanced)
    overview_tab = dcc.Tab(label='📈 Overview', children=[
        html.Div([
            create_enhanced_summary_card(df, type_report),
            
            # Interactive Visualizations Section
            html.Div([
                html.H4("Interactive Data Exploration", 
                       style={'marginBottom': '20px', 'color': '#2c3e50'}),
                
                html.Div([
                    html.Div([
                        create_detailed_data_types_chart(df)
                    ], style={'width': '48%', 'display': 'inline-block'}),
                    
                    html.Div([
                        create_missing_values_chart(df)
                    ], style={'width': '48%', 'display': 'inline-block', 'float': 'right'})
                ], style={'marginTop': '20px'}),
                
                html.Div([
                    create_memory_usage_chart(df)
                ], style={'marginTop': '20px'}),
                
                html.Div([
                    create_correlation_matrix(df)
                ], style={'marginTop': '20px'})
            ], style={
                'backgroundColor': 'white',
                'padding': '25px',
                'borderRadius': '10px',
                'boxShadow': '0 4px 6px rgba(0,0,0,0.1)',
                'marginTop': '20px'
            })
        ], style={'padding': '20px'})
    ])
    tabs.append(overview_tab)
    
    # Tab 3: Numerical Analysis (only if numerical columns exist)
    if numerical_cols:
        numerical_tab = dcc.Tab(label='🔢 Numerical', children=[
            html.Div([
                html.H4("Numerical Column Analysis", 
                       style={'marginBottom': '20px', 'color': '#2c3e50'}),
                
                html.Div([
                    html.Div([
                        html.Label("Select Numerical Column", 
                                 style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                        dcc.Dropdown(
                            id='num-col-selector',
                            options=[{'label': col, 'value': col} for col in numerical_cols],
                            value=numerical_cols[0] if numerical_cols else None,
                            style={'width': '100%'},
                            clearable=False
                        )
                    ], style={'width': '48%', 'display': 'inline-block'}),
                    
                    html.Div([
                        html.Label("Chart Type", 
                                 style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                        dcc.Dropdown(
                            id='num-chart-type',
                            options=[
                                {'label': 'Histogram', 'value': 'histogram'},
                                {'label': 'Box Plot', 'value': 'box'},
                                {'label': 'Violin Plot', 'value': 'violin'},
                                {'label': 'ECDF', 'value': 'ecdf'},
                                {'label': 'Density', 'value': 'density'}
                            ],
                            value='histogram',
                            style={'width': '100%'},
                            clearable=False
                        )
                    ], style={'width': '48%', 'display': 'inline-block', 'float': 'right'})
                ], style={'marginBottom': '30px', 'backgroundColor': '#f8f9fa', 
                         'padding': '15px', 'borderRadius': '5px'}),
                
                dcc.Graph(id='numerical-chart'),
                
                html.Div([
                    create_statistics_table(df, numerical_cols)
                ], style={'marginTop': '30px'})
            ], style={'padding': '20px'})
        ])
        tabs.append(numerical_tab)
    
    # Tab 4: Categorical Analysis (only if categorical columns exist)
    if categorical_cols:
        categorical_tab = dcc.Tab(label='📝 Categorical', children=[
            html.Div([
                html.H4("Categorical Column Analysis", 
                       style={'marginBottom': '20px', 'color': '#2c3e50'}),
                
                html.Div([
                    html.Div([
                        html.Label("Select Categorical Column", 
                                 style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                        dcc.Dropdown(
                            id='cat-col-selector',
                            options=[{'label': col, 'value': col} for col in categorical_cols],
                            value=categorical_cols[0] if categorical_cols else None,
                            style={'width': '100%'},
                            clearable=False
                        )
                    ], style={'width': '48%', 'display': 'inline-block'}),
                    
                    html.Div([
                        html.Label("Select Numerical Column (for comparison)", 
                                 style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                        dcc.Dropdown(
                            id='cat-num-col-selector',
                            options=[{'label': col, 'value': col} for col in numerical_cols],
                            value=numerical_cols[0] if numerical_cols else None,
                            style={'width': '100%'},
                            clearable=False
                        )
                    ], style={'width': '48%', 'display': 'inline-block', 'float': 'right'})
                ], style={'marginBottom': '30px', 'backgroundColor': '#f8f9fa', 
                         'padding': '15px', 'borderRadius': '5px'}),
                
                dcc.Graph(id='categorical-chart'),
                
                html.Div([
                    dcc.Graph(id='value-counts-chart')
                ], style={'marginTop': '30px'})
            ], style={'padding': '20px'})
        ])
        tabs.append(categorical_tab)
    
    # Tab 5: DateTime Analysis (only if datetime columns exist)
    if datetime_cols:
        datetime_tab = dcc.Tab(label='📅 DateTime', children=[
            html.Div([
                html.H4("Date/Time Column Analysis", 
                       style={'marginBottom': '20px', 'color': '#2c3e50'}),
                
                html.Div([
                    html.Div([
                        html.Label("Select Date/Time Column", 
                                 style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                        dcc.Dropdown(
                            id='date-col-selector',
                            options=[{'label': col, 'value': col} for col in datetime_cols],
                            value=datetime_cols[0] if datetime_cols else None,
                            style={'width': '100%'},
                            clearable=False
                        )
                    ], style={'width': '48%', 'display': 'inline-block'}),
                    
                    html.Div([
                        html.Label("Time Unit", 
                                 style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                        dcc.Dropdown(
                            id='time-unit',
                            options=[
                                {'label': 'Year', 'value': 'year'},
                                {'label': 'Month', 'value': 'month'},
                                {'label': 'Day', 'value': 'day'},
                                {'label': 'Hour', 'value': 'hour'},
                                {'label': 'Minute', 'value': 'minute'},
                                {'label': 'Day of Week', 'value': 'dayofweek'}
                            ],
                            value='month',
                            style={'width': '100%'},
                            clearable=False
                        )
                    ], style={'width': '48%', 'display': 'inline-block', 'float': 'right'})
                ], style={'marginBottom': '30px', 'backgroundColor': '#f8f9fa', 
                         'padding': '15px', 'borderRadius': '5px'}),
                
                dcc.Graph(id='datetime-chart'),
                
                html.Div([
                    create_datetime_statistics(df, datetime_cols)
                ], style={'marginTop': '30px'})
            ], style={'padding': '20px'})
        ])
        tabs.append(datetime_tab)
    
    # Tab 6: Relationships
    if len(numerical_cols) >= 2:
        relationships_tab = dcc.Tab(label='🔗 Relationships', children=[
            html.Div([
                html.H4("Column Relationships", 
                       style={'marginBottom': '20px', 'color': '#2c3e50'}),
                
                html.Div([
                    html.Div([
                        html.Label("X-axis Column", 
                                 style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                        dcc.Dropdown(
                            id='x-col-selector',
                            options=[{'label': col, 'value': col} for col in numerical_cols],
                            value=numerical_cols[0],
                            style={'width': '100%'},
                            clearable=False
                        )
                    ], style={'width': '32%', 'display': 'inline-block'}),
                    
                    html.Div([
                        html.Label("Y-axis Column", 
                                 style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                        dcc.Dropdown(
                            id='y-col-selector',
                            options=[{'label': col, 'value': col} for col in numerical_cols],
                            value=numerical_cols[1] if len(numerical_cols) > 1 else numerical_cols[0],
                            style={'width': '100%'},
                            clearable=False
                        )
                    ], style={'width': '32%', 'display': 'inline-block', 'marginLeft': '2%'}),
                    
                    html.Div([
                        html.Label("Color By (Optional)", 
                                 style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                        dcc.Dropdown(
                            id='color-col-selector',
                            options=[{'label': 'None', 'value': 'none'}] + 
                                    [{'label': col, 'value': col} for col in categorical_cols],
                            value='none',
                            style={'width': '100%'}
                        )
                    ], style={'width': '32%', 'display': 'inline-block', 'marginLeft': '2%'})
                ], style={'marginBottom': '30px', 'backgroundColor': '#f8f9fa', 
                         'padding': '15px', 'borderRadius': '5px'}),
                
                dcc.Graph(id='scatter-plot'),
                
                html.Div([
                    html.H4("Correlation Analysis", 
                           style={'marginBottom': '20px', 'color': '#2c3e50'}),
                    dcc.Graph(figure=create_correlation_heatmap(df))
                ], style={'marginTop': '30px'})
            ], style={'padding': '20px'})
        ])
        tabs.append(relationships_tab)
    
    return html.Div([
        dcc.Tabs(tabs, colors={
            "border": "white",
            "primary": "#007bff",
            "background": "#f8f9fa"
        })
    ])


def get_data_type_report(df):
    """Get detailed data type report"""
    report = {
        'total_columns': len(df.columns),
        'total_rows': len(df),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024**2),
        'type_distribution': {},
        'column_details': {},
        'null_summary': {}
    }
    
    # Count data types
    dtype_counts = df.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        report['type_distribution'][str(dtype)] = int(count)
    
    # Column details
    for col in df.columns:
        dtype = str(df[col].dtype)
        null_count = df[col].isnull().sum()
        unique_count = df[col].nunique()
        memory_bytes = df[col].memory_usage(deep=True)
        
        # Get sample value
        sample_val = df[col].iloc[0] if len(df) > 0 else None
        if pd.isna(sample_val):
            # Find first non-null value
            non_null = df[col].dropna()
            sample_val = non_null.iloc[0] if len(non_null) > 0 else "NaN"
        
        report['column_details'][col] = {
            'dtype': dtype,
            'null_count': int(null_count),
            'null_percentage': float((null_count / len(df)) * 100),
            'unique_values': int(unique_count),
            'memory_bytes': float(memory_bytes),
            'sample': str(sample_val)[:100]  # Truncate long samples
        }
    
    # Null summary
    total_cells = len(df) * len(df.columns)
    report['null_summary'] = {
        'total_nulls': df.isnull().sum().sum(),
        'columns_with_nulls': (df.isnull().sum() > 0).sum(),
        'null_percentage': (df.isnull().sum().sum() / total_cells * 100) if total_cells > 0 else 0
    }
    
    return report


def create_data_type_analysis_tab(df, type_report):
    """Create detailed data type analysis tab"""
    
    # Create table rows for column details
    table_rows = []
    for col, details in type_report['column_details'].items():
        # Color code based on null percentage
        null_color = '#28a745' if details['null_percentage'] == 0 else '#ffc107' if details['null_percentage'] < 10 else '#dc3545'
        
        table_rows.append(html.Tr([
            html.Td(col, style={'fontWeight': 'bold', 'fontSize': '12px', 'padding': '8px'}),
            html.Td(details['dtype'], style={'fontSize': '12px', 'padding': '8px'}),
            html.Td(f"{details['null_count']:,}", 
                   style={'fontSize': '12px', 'padding': '8px', 'textAlign': 'right'}),
            html.Td(f"{details['null_percentage']:.1f}%", 
                   style={'fontSize': '12px', 'padding': '8px', 'textAlign': 'right', 'color': null_color}),
            html.Td(f"{details['unique_values']:,}", 
                   style={'fontSize': '12px', 'padding': '8px', 'textAlign': 'right'}),
            html.Td(f"{details['memory_bytes'] / 1024:.1f} KB", 
                   style={'fontSize': '12px', 'padding': '8px', 'textAlign': 'right'})
        ]))
    
    # Create data type distribution chart
    dtype_fig = px.bar(
        x=list(type_report['type_distribution'].keys()),
        y=list(type_report['type_distribution'].values()),
        title="Data Type Distribution",
        labels={'x': 'Data Type', 'y': 'Count'},
        color=list(type_report['type_distribution'].values()),
        color_continuous_scale='Blues'
    )
    dtype_fig.update_traces(texttemplate='%{y}', textposition='outside')
    dtype_fig.update_layout(height=350, showlegend=False)
    
    return html.Div([
        html.H3("📊 Data Type Analysis", style={'marginBottom': '20px', 'color': '#2c3e50'}),
        
        # Summary cards
        html.Div([
            html.Div([
                html.H4(f"{type_report['total_columns']:,}", 
                       style={'color': '#007bff', 'margin': '0', 'fontSize': '28px'}),
                html.P("Total Columns", style={'color': '#6c757d', 'margin': '5px 0 0 0', 'fontSize': '14px'})
            ], style={
                'backgroundColor': 'white',
                'padding': '20px',
                'borderRadius': '8px',
                'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                'textAlign': 'center',
                'flex': '1',
                'marginRight': '15px'
            }),
            
            html.Div([
                html.H4(f"{len(type_report['type_distribution'])}", 
                       style={'color': '#28a745', 'margin': '0', 'fontSize': '28px'}),
                html.P("Unique Types", style={'color': '#6c757d', 'margin': '5px 0 0 0', 'fontSize': '14px'})
            ], style={
                'backgroundColor': 'white',
                'padding': '20px',
                'borderRadius': '8px',
                'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                'textAlign': 'center',
                'flex': '1',
                'marginRight': '15px'
            }),
            
            html.Div([
                html.H4(f"{type_report['null_summary']['columns_with_nulls']:,}", 
                       style={'color': '#ffc107', 'margin': '0', 'fontSize': '28px'}),
                html.P("With Nulls", style={'color': '#6c757d', 'margin': '5px 0 0 0', 'fontSize': '14px'})
            ], style={
                'backgroundColor': 'white',
                'padding': '20px',
                'borderRadius': '8px',
                'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                'textAlign': 'center',
                'flex': '1',
                'marginRight': '15px'
            }),
            
            html.Div([
                html.H4(f"{type_report['memory_usage_mb']:.1f}", 
                       style={'color': '#17a2b8', 'margin': '0', 'fontSize': '28px'}),
                html.P("MB Memory", style={'color': '#6c757d', 'margin': '5px 0 0 0', 'fontSize': '14px'})
            ], style={
                'backgroundColor': 'white',
                'padding': '20px',
                'borderRadius': '8px',
                'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                'textAlign': 'center',
                'flex': '1'
            })
        ], style={'display': 'flex', 'marginBottom': '30px'}),
        
        # Data type distribution chart
        html.Div([
            dcc.Graph(figure=dtype_fig)
        ], style={'marginBottom': '30px', 'backgroundColor': 'white', 
                 'padding': '15px', 'borderRadius': '8px'}),
        
        # Column details table
        html.Div([
            html.H4("Column Details", 
                   style={'marginBottom': '15px', 'color': '#2c3e50'}),
            html.Div([
                html.Table([
                    html.Thead(
                        html.Tr([
                            html.Th("Column", style={'padding': '12px', 'fontSize': '14px', 
                                                    'backgroundColor': '#f8f9fa', 'textAlign': 'left'}),
                            html.Th("Data Type", style={'padding': '12px', 'fontSize': '14px', 
                                                       'backgroundColor': '#f8f9fa', 'textAlign': 'left'}),
                            html.Th("Null Count", style={'padding': '12px', 'fontSize': '14px', 
                                                        'backgroundColor': '#f8f9fa', 'textAlign': 'right'}),
                            html.Th("Null %", style={'padding': '12px', 'fontSize': '14px', 
                                                    'backgroundColor': '#f8f9fa', 'textAlign': 'right'}),
                            html.Th("Unique", style={'padding': '12px', 'fontSize': '14px', 
                                                    'backgroundColor': '#f8f9fa', 'textAlign': 'right'}),
                            html.Th("Memory", style={'padding': '12px', 'fontSize': '14px', 
                                                    'backgroundColor': '#f8f9fa', 'textAlign': 'right'})
                        ], style={'borderBottom': '2px solid #dee2e6'})
                    ),
                    html.Tbody(table_rows)
                ], style={
                    'width': '100%',
                    'borderCollapse': 'collapse',
                    'fontSize': '12px',
                    'border': '1px solid #dee2e6'
                })
            ], style={
                'maxHeight': '400px',
                'overflowY': 'auto',
                'borderRadius': '8px',
                'border': '1px solid #dee2e6'
            })
        ], style={'backgroundColor': 'white', 'padding': '20px', 
                 'borderRadius': '8px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'})
    ], style={'padding': '20px'})


def create_enhanced_summary_card(df, type_report):
    """Create enhanced summary statistics card"""
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime', 'datetimetz']).columns.tolist()
    boolean_cols = df.select_dtypes(include=['bool']).columns.tolist()
    
    return html.Div([
        html.H3("📈 Dataset Summary", style={'marginBottom': '25px', 'color': '#2c3e50'}),
        html.Div([
            html.Div([
                html.H2(f"{df.shape[0]:,}", 
                       style={'color': '#007bff', 'margin': '10px 0', 'fontSize': '36px'}),
                html.P("Total Rows", style={'color': '#6c757d', 'fontSize': '14px', 'fontWeight': '500'})
            ], style={
                'textAlign': 'center',
                'flex': '1',
                'marginRight': '20px',
                'borderRight': '1px solid #eee'
            }),
            
            html.Div([
                html.H2(f"{df.shape[1]}", 
                       style={'color': '#28a745', 'margin': '10px 0', 'fontSize': '36px'}),
                html.P("Total Columns", style={'color': '#6c757d', 'fontSize': '14px', 'fontWeight': '500'})
            ], style={
                'textAlign': 'center',
                'flex': '1',
                'marginRight': '20px',
                'borderRight': '1px solid #eee'
            }),
            
            html.Div([
                html.H2(f"{len(numerical_cols)}", 
                       style={'color': '#17a2b8', 'margin': '10px 0', 'fontSize': '36px'}),
                html.P("Numerical", style={'color': '#6c757d', 'fontSize': '14px', 'fontWeight': '500'})
            ], style={
                'textAlign': 'center',
                'flex': '1',
                'marginRight': '20px',
                'borderRight': '1px solid #eee'
            }),
            
            html.Div([
                html.H2(f"{len(categorical_cols)}", 
                       style={'color': '#ffc107', 'margin': '10px 0', 'fontSize': '36px'}),
                html.P("Categorical", style={'color': '#6c757d', 'fontSize': '14px', 'fontWeight': '500'})
            ], style={
                'textAlign': 'center',
                'flex': '1'
            })
        ], style={'display': 'flex', 'marginBottom': '25px', 'paddingBottom': '25px', 
                 'borderBottom': '1px solid #eee'}),
        
        # Second row of stats
        html.Div([
            html.Div([
                html.H3(f"{len(datetime_cols)}", 
                       style={'color': '#6f42c1', 'margin': '5px 0', 'fontSize': '28px'}),
                html.P("DateTime", style={'color': '#6c757d', 'fontSize': '13px', 'fontWeight': '500'})
            ], style={
                'textAlign': 'center',
                'flex': '1',
                'marginRight': '15px'
            }),
            
            html.Div([
                html.H3(f"{len(boolean_cols)}", 
                       style={'color': '#e83e8c', 'margin': '5px 0', 'fontSize': '28px'}),
                html.P("Boolean", style={'color': '#6c757d', 'fontSize': '13px', 'fontWeight': '500'})
            ], style={
                'textAlign': 'center',
                'flex': '1',
                'marginRight': '15px'
            }),
            
            html.Div([
                html.H3(f"{type_report['memory_usage_mb']:.1f}", 
                       style={'color': '#20c997', 'margin': '5px 0', 'fontSize': '28px'}),
                html.P("MB Memory", style={'color': '#6c757d', 'fontSize': '13px', 'fontWeight': '500'})
            ], style={
                'textAlign': 'center',
                'flex': '1',
                'marginRight': '15px'
            }),
            
            html.Div([
                html.H3(f"{type_report['null_summary']['null_percentage']:.1f}%", 
                       style={'color': '#fd7e14', 'margin': '5px 0', 'fontSize': '28px'}),
                html.P("Null Values", style={'color': '#6c757d', 'fontSize': '13px', 'fontWeight': '500'})
            ], style={
                'textAlign': 'center',
                'flex': '1'
            })
        ], style={'display': 'flex'})
    ], style={
        'backgroundColor': 'white',
        'padding': '30px',
        'borderRadius': '12px',
        'boxShadow': '0 4px 12px rgba(0,0,0,0.15)',
        'border': '1px solid #e9ecef'
    })


def create_detailed_data_types_chart(df):
    """Create chart showing detailed data type distribution"""
    dtype_counts = df.dtypes.value_counts()
    
    fig = px.bar(
        x=list(dtype_counts.index.astype(str)),
        y=dtype_counts.values,
        title="Data Type Distribution",
        labels={'x': 'Data Type', 'y': 'Count'},
        color=dtype_counts.values,
        color_continuous_scale='Blues'
    )
    
    fig.update_layout(
        height=350,
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    fig.update_traces(texttemplate='%{y}', textposition='outside')
    fig.update_xaxes(tickangle=45)
    
    return dcc.Graph(figure=fig)


def create_missing_values_chart(df):
    """Create chart showing missing values"""
    missing_counts = df.isnull().sum()
    missing_counts = missing_counts[missing_counts > 0]
    
    if len(missing_counts) > 0:
        fig = px.bar(
            x=missing_counts.index,
            y=missing_counts.values,
            title="Missing Values by Column",
            labels={'x': 'Column', 'y': 'Missing Count'},
            color=missing_counts.values,
            color_continuous_scale='Reds'
        )
        fig.update_traces(texttemplate='%{y}', textposition='outside')
        fig.update_layout(
            height=350,
            showlegend=False,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        fig.update_xaxes(tickangle=45)
    else:
        # Create a simple message chart if no missing values
        fig = go.Figure()
        fig.add_annotation(
            text="✅ No missing values!",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=20, color="green")
        )
        fig.update_layout(
            title="Missing Values by Column",
            height=350,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
    
    return dcc.Graph(figure=fig)


def create_memory_usage_chart(df):
    """Create chart showing memory usage by column"""
    memory_usage = df.memory_usage(deep=True).sort_values(ascending=False)
    memory_usage = memory_usage[memory_usage > 0]  # Remove zero usage
    
    if len(memory_usage) > 0:
        # Take top 15 columns or all if less than 15
        top_n = min(15, len(memory_usage))
        fig = px.bar(
            x=memory_usage.index[:top_n],
            y=memory_usage.values[:top_n] / 1024,  # Convert to KB
            title=f"Memory Usage by Column (Top {top_n})",
            labels={'x': 'Column', 'y': 'Memory (KB)'},
            color=memory_usage.values[:top_n],
            color_continuous_scale='Viridis'
        )
        fig.update_traces(texttemplate='%{y:.1f} KB', textposition='outside')
        fig.update_layout(
            height=350,
            showlegend=False,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        fig.update_xaxes(tickangle=45)
    else:
        fig = go.Figure()
        fig.add_annotation(
            text="No memory usage data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False
        )
        fig.update_layout(
            title="Memory Usage by Column",
            height=350,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
    
    return dcc.Graph(figure=fig)


def create_datetime_statistics(df, datetime_cols):
    """Create statistics for datetime columns"""
    if not datetime_cols:
        return html.Div([
            html.H5("No datetime columns available", style={'color': '#6c757d'}),
            html.P("The dataset doesn't contain any datetime columns.")
        ], style={'textAlign': 'center', 'padding': '30px'})
    
    stats_rows = []
    for col in datetime_cols[:10]:  # Show first 10 datetime columns
        if col in df.columns:
            col_data = df[col].dropna()
            if len(col_data) > 0:
                try:
                    # Extract datetime components
                    min_date = col_data.min()
                    max_date = col_data.max()
                    date_range = max_date - min_date
                    
                    # Calculate statistics
                    stats_rows.append(html.Tr([
                        html.Td(col, style={'fontSize': '13px', 'padding': '10px', 'fontWeight': 'bold'}),
                        html.Td(min_date.strftime('%Y-%m-%d'), style={'fontSize': '13px', 'padding': '10px'}),
                        html.Td(max_date.strftime('%Y-%m-%d'), style={'fontSize': '13px', 'padding': '10px'}),
                        html.Td(str(date_range.days) + ' days', style={'fontSize': '13px', 'padding': '10px'}),
                        html.Td(f"{len(col_data):,}", style={'fontSize': '13px', 'padding': '10px', 'textAlign': 'right'}),
                        html.Td(f"{df[col].isnull().sum():,}", style={'fontSize': '13px', 'padding': '10px', 'textAlign': 'right'}),
                        html.Td(f"{(df[col].isnull().sum() / len(df) * 100):.1f}%", 
                               style={'fontSize': '13px', 'padding': '10px', 'textAlign': 'right'})
                    ]))
                except Exception as e:
                    stats_rows.append(html.Tr([
                        html.Td(col, style={'fontSize': '13px', 'padding': '10px', 'fontWeight': 'bold'}),
                        html.Td("Error", colSpan=6, style={'fontSize': '13px', 'padding': '10px', 'color': 'red'})
                    ]))
    
    return html.Div([
        html.H5("DateTime Column Statistics", style={'marginBottom': '15px', 'color': '#2c3e50'}),
        html.Div([
            html.Table([
                html.Thead(
                    html.Tr([
                        html.Th("Column", style={'fontSize': '14px', 'padding': '12px', 'backgroundColor': '#f8f9fa'}),
                        html.Th("Min Date", style={'fontSize': '14px', 'padding': '12px', 'backgroundColor': '#f8f9fa'}),
                        html.Th("Max Date", style={'fontSize': '14px', 'padding': '12px', 'backgroundColor': '#f8f9fa'}),
                        html.Th("Range", style={'fontSize': '14px', 'padding': '12px', 'backgroundColor': '#f8f9fa'}),
                        html.Th("Non-Null", style={'fontSize': '14px', 'padding': '12px', 'backgroundColor': '#f8f9fa'}),
                        html.Th("Null Count", style={'fontSize': '14px', 'padding': '12px', 'backgroundColor': '#f8f9fa'}),
                        html.Th("Null %", style={'fontSize': '14px', 'padding': '12px', 'backgroundColor': '#f8f9fa'})
                    ])
                ),
                html.Tbody(stats_rows)
            ], style={
                'width': '100%',
                'borderCollapse': 'collapse',
                'border': '1px solid #dee2e6',
                'fontSize': '12px'
            })
        ], style={'maxHeight': '300px', 'overflowY': 'auto', 'borderRadius': '5px'})
    ], style={
        'backgroundColor': 'white',
        'padding': '20px',
        'borderRadius': '8px',
        'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
    })


def create_correlation_matrix(df):
    """Create correlation matrix heatmap"""
    numerical_df = df.select_dtypes(include=[np.number])
    
    if numerical_df.shape[1] < 2:
        fig = go.Figure()
        fig.add_annotation(
            text="Not enough numerical columns for correlation analysis",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color="#6c757d")
        )
        fig.update_layout(
            title="Correlation Matrix",
            height=400,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        return dcc.Graph(figure=fig)
    
    corr_matrix = numerical_df.corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=corr_matrix.round(2).values,
        texttemplate='%{text}',
        textfont={"size": 10},
        hoverinfo="text"
    ))
    
    fig.update_layout(
        title="Correlation Matrix",
        height=400,
        xaxis_title="Columns",
        yaxis_title="Columns",
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    return dcc.Graph(figure=fig)


def create_statistics_table(df, numerical_cols):
    """Create statistics table for numerical columns"""
    if not numerical_cols:
        return html.Div([
            html.H5("No numerical columns available", style={'color': '#6c757d'}),
            html.P("The dataset doesn't contain any numerical columns.")
        ], style={'textAlign': 'center', 'padding': '30px'})
    
    # Take only first 10 numerical columns for display
    display_cols = numerical_cols[:10]
    stats_df = df[display_cols].describe().round(2).T
    stats_df = stats_df.reset_index()
    stats_df.columns = ['Column', 'Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max']
    
    # Create HTML table
    table_header = [html.Thead(html.Tr([
        html.Th(col, style={'padding': '12px', 'fontSize': '14px', 'backgroundColor': '#f8f9fa'}) 
        for col in stats_df.columns
    ]))]
    
    table_body = []
    for i in range(len(stats_df)):
        row = []
        for col in stats_df.columns:
            value = stats_df.iloc[i][col]
            if isinstance(value, (int, np.integer)):
                display_value = f"{value:,}"
            elif isinstance(value, (float, np.floating)):
                display_value = f"{value:.2f}"
            else:
                display_value = str(value)
            
            row.append(html.Td(display_value, style={'padding': '10px', 'fontSize': '12px'}))
        table_body.append(html.Tr(row))
    
    table = html.Table(
        table_header + [html.Tbody(table_body)],
        style={
            'width': '100%',
            'borderCollapse': 'collapse',
            'border': '1px solid #dee2e6',
            'fontSize': '12px'
        }
    )
    
    return html.Div([
        html.H5("Descriptive Statistics", style={'marginBottom': '15px', 'color': '#2c3e50'}),
        html.P(f"Showing statistics for first {len(display_cols)} numerical columns", 
              style={'color': '#6c757d', 'fontSize': '14px', 'marginBottom': '15px'}),
        html.Div([
            table
        ], style={
            'maxHeight': '400px',
            'overflowY': 'auto',
            'borderRadius': '5px'
        })
    ], style={
        'backgroundColor': 'white',
        'padding': '20px',
        'borderRadius': '8px',
        'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
    })


def create_correlation_heatmap(df):
    """Create correlation heatmap"""
    numerical_df = df.select_dtypes(include=[np.number])
    
    if numerical_df.shape[1] < 2:
        fig = go.Figure()
        fig.add_annotation(
            text="Not enough numerical columns",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False
        )
        fig.update_layout(
            title="Correlation Heatmap",
            height=400,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        return fig
    
    corr = numerical_df.corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.columns,
        colorscale='RdBu',
        zmid=0
    ))
    
    fig.update_layout(
        title="Correlation Heatmap",
        height=400,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    return fig


# These functions are called by callbacks in app.py
def create_numerical_chart(df, col, chart_type='histogram'):
    """Create chart for numerical column"""
    if col not in df.columns:
        return go.Figure()
    
    if chart_type == 'histogram':
        fig = px.histogram(df, x=col, title=f"Distribution of {col}")
        fig.update_layout(bargap=0.1, plot_bgcolor='white', paper_bgcolor='white')
    elif chart_type == 'box':
        fig = px.box(df, y=col, title=f"Box Plot of {col}")
        fig.update_layout(plot_bgcolor='white', paper_bgcolor='white')
    elif chart_type == 'violin':
        fig = px.violin(df, y=col, title=f"Violin Plot of {col}")
        fig.update_layout(plot_bgcolor='white', paper_bgcolor='white')
    elif chart_type == 'ecdf':
        fig = px.ecdf(df, x=col, title=f"ECDF of {col}")
        fig.update_layout(plot_bgcolor='white', paper_bgcolor='white')
    elif chart_type == 'density':
        fig = px.density_contour(df, x=col, title=f"Density Plot of {col}")
        fig.update_traces(contours_coloring="fill", contours_showlabels=True)
        fig.update_layout(plot_bgcolor='white', paper_bgcolor='white')
    else:
        fig = px.histogram(df, x=col, title=f"Distribution of {col}")
        fig.update_layout(bargap=0.1, plot_bgcolor='white', paper_bgcolor='white')
    
    return fig


def create_categorical_chart(df, cat_col, num_col=None):
    """Create chart for categorical column"""
    if cat_col not in df.columns:
        return go.Figure()
    
    if num_col and num_col in df.columns:
        # Create grouped box plot
        fig = px.box(df, x=cat_col, y=num_col, 
                    title=f"{num_col} by {cat_col}")
        fig.update_layout(plot_bgcolor='white', paper_bgcolor='white')
    else:
        # Create bar chart of value counts
        value_counts = df[cat_col].value_counts().reset_index()
        value_counts.columns = [cat_col, 'count']
        
        fig = px.bar(value_counts.head(20), x=cat_col, y='count',
                    title=f"Value Counts for {cat_col}")
        fig.update_layout(plot_bgcolor='white', paper_bgcolor='white')
    
    return fig


def create_scatter_plot(df, x_col, y_col, color_col=None):
    """Create scatter plot"""
    if x_col not in df.columns or y_col not in df.columns:
        return go.Figure()
    
    if color_col and color_col != 'none' and color_col in df.columns:
        fig = px.scatter(df, x=x_col, y=y_col, color=color_col,
                        title=f"{y_col} vs {x_col} (colored by {color_col})",
                        hover_data=df.columns.tolist()[:5])  # Limit hover data
        fig.update_layout(plot_bgcolor='white', paper_bgcolor='white')
    else:
        fig = px.scatter(df, x=x_col, y=y_col,
                        title=f"{y_col} vs {x_col}",
                        hover_data=df.columns.tolist()[:5])
        fig.update_layout(plot_bgcolor='white', paper_bgcolor='white')
    
    return fig