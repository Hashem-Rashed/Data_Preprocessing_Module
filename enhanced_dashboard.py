"""
Enhanced Dashboard - Fixed with working controls and better visualization
"""
from dash import dcc, html
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime

def create_enhanced_dashboard(df):
    """
    Create enhanced dashboard that properly displays processed data
    """
    
    if df is None or df.empty:
        return html.Div([
            html.H1("🎯 Enhanced Data Dashboard", className="dashboard-title"),
            html.P("No data available. Please run the data processing pipeline first.", 
                   className="dashboard-subtitle"),
        ], className="header")
    
    # Process data for enhanced dashboard
    df = df.copy()
    
    # Calculate statistics
    total_rows = len(df)
    total_columns = len(df.columns)
    
    # Identify column types
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime', 'datetimetz']).columns.tolist()
    
    # Memory usage
    memory_usage_mb = df.memory_usage(deep=True).sum() / (1024**2)
    
    # Null values
    null_count = df.isnull().sum().sum()
    total_cells = total_rows * total_columns
    null_percentage = (null_count / total_cells * 100) if total_cells > 0 else 0
    
    # Get default column for visualization
    default_column = None
    if len(numerical_cols) > 0:
        default_column = numerical_cols[0]
    elif len(categorical_cols) > 0:
        default_column = categorical_cols[0]
    elif len(df.columns) > 0:
        default_column = df.columns[0]
    
    return html.Div([
        # Header
        html.Div([
            html.H1("🎯 Enhanced Data Analytics Dashboard", className="dashboard-title"),
            html.P("Interactive visualization of processed data with comprehensive analytics", 
                   className="dashboard-subtitle"),
        ], className="header"),
        
        # Main dashboard content
        html.Div([
            # First Section: Key Metrics Cards
            html.Div([
                html.Div([
                    html.Div([
                        html.Div("Total Records", className="stat-label"),
                        html.Div(f"{total_rows:,}", className="stat-number"),
                        html.Div("Processed rows", className="stat-description")
                    ], className="stat-card"),
                    
                    html.Div([
                        html.Div("Total Columns", className="stat-label"),
                        html.Div(f"{total_columns}", className="stat-number"),
                        html.Div("Features available", className="stat-description")
                    ], className="stat-card"),
                    
                    html.Div([
                        html.Div("Memory Usage", className="stat-label"),
                        html.Div(f"{memory_usage_mb:.1f} MB", className="stat-number"),
                        html.Div("Optimized storage", className="stat-description")
                    ], className="stat-card"),
                    
                    html.Div([
                        html.Div("Missing Values", className="stat-label"),
                        html.Div(f"{null_count:,}", className="stat-number"),
                        html.Div(f"{null_percentage:.1f}% of data", className="stat-description")
                    ], className="stat-card"),
                ], className="statistics-grid"),
            ], className="dashboard-content"),
            
            # Second Section: Interactive Analysis - FIXED WITH PROPER IDS
            html.Div([
                html.H3("🔍 Interactive Data Analysis", className="app-section-title"),
                
                html.Div([
                    # Left panel - Controls
                    html.Div([
                        html.Div([
                            html.Label("Select Column for Analysis", className="app-form-label"),
                            dcc.Dropdown(
                                id='enhanced-col-selector',
                                options=[{'label': col, 'value': col} for col in df.columns],
                                value=default_column,
                                className="filter-dropdown",
                                clearable=False
                            )
                        ], className="form-group"),
                        
                        html.Div([
                            html.Label("Visualization Type", className="app-form-label"),
                            dcc.Dropdown(
                                id='enhanced-vis-type',
                                options=[
                                    {'label': '📊 Histogram', 'value': 'histogram'},
                                    {'label': '📦 Box Plot', 'value': 'box'},
                                    {'label': '📋 Bar Chart', 'value': 'bar'},
                                    {'label': '📈 Line Chart', 'value': 'line'},
                                    {'label': '🎯 Scatter Plot', 'value': 'scatter'},
                                    {'label': '📊 Density Plot', 'value': 'density'},
                                    {'label': '📊 Violin Plot', 'value': 'violin'}
                                ],
                                value='histogram',
                                className="filter-dropdown",
                                clearable=False
                            )
                        ], className="form-group"),
                        
                        # Download buttons section
                        html.Div([
                            html.Label("Export Options", className="app-form-label"),
                            html.Div([
                                html.Button(
                                    "📥 Download CSV",
                                    id='enhanced-download-csv',
                                    className="dashboard-button success",
                                    style={'width': '100%', 'marginBottom': '10px'}
                                ),
                                html.Button(
                                    "📄 Download Report",
                                    id='enhanced-download-report',
                                    className="dashboard-button info",
                                    style={'width': '100%'}
                                )
                            ], style={'display': 'flex', 'flexDirection': 'column', 'gap': '10px'})
                        ], className="form-group"),
                        
                        # Column statistics - will be updated by callbacks
                        html.Div(id='enhanced-statistics', className="info-card"),
                        
                    ], className="left-panel"),
                    
                    # Right panel - Visualization
                    html.Div([
                        html.Div([
                            html.H4("Data Visualization", className="app-card-title"),
                            dcc.Graph(
                                id='enhanced-visualization',
                                config={'displayModeBar': True, 'displaylogo': False},
                                style={'height': '500px', 'width': '100%'}
                            )
                        ], style={'height': '100%'})
                    ], className="right-panel"),
                ], className="processing-container"),
            ], className="dashboard-content"),
            
            # Third Section: Data Quality Dashboard
            html.Div([
                html.H3("📊 Data Quality Assessment", className="app-section-title"),
                html.Div([
                    dcc.Graph(
                        id='enhanced-quality-chart',
                        figure=create_data_quality_chart(df),
                        config={'displayModeBar': True, 'displaylogo': False},
                        style={'height': '450px', 'width': '100%'}
                    )
                ], className="dashboard-content"),
            ]),
            
            # Fourth Section: Statistical Overview
            html.Div([
                html.H3("📈 Statistical Overview", className="app-section-title"),
                html.Div([
                    html.Div([
                        dcc.Graph(
                            id='enhanced-statistics-chart',
                            figure=create_numerical_statistics_chart(df),
                            config={'displayModeBar': True, 'displaylogo': False},
                            style={'height': '450px', 'width': '100%'}
                        )
                    ], className="dashboard-content"),
                ]),
            ]),
            
            # Fifth Section: Missing Values Analysis
            html.Div([
                html.H3("⚠️ Missing Values Analysis", className="app-section-title"),
                html.Div([
                    dcc.Graph(
                        id='enhanced-missing-chart',
                        figure=create_missing_values_chart(df),
                        config={'displayModeBar': True, 'displaylogo': False},
                        style={'height': '450px', 'width': '100%'}
                    )
                ], className="dashboard-content"),
            ]),
            
            # Sixth Section: Data Preview
            html.Div([
                html.H3("📋 Processed Data Preview", className="app-section-title"),
                html.Div([
                    html.P(f"Showing first 10 rows of {total_rows:,} total rows", className="app-info-text"),
                    create_data_preview_table(df)
                ], className="preview-container")
            ], className="dashboard-content"),
            
        ], className="dashboard-content"),
    ])

def create_column_visualization(df, column, vis_type='histogram'):
    """Create visualization for a specific column"""
    if column is None or column not in df.columns:
        return create_empty_chart("Select a column to visualize")
    
    col_data = df[column].dropna()
    
    # Set theme colors
    colors = ['#6495ed', '#5d8aa8', '#4caf50', '#ff9800', '#9c27b0', '#f44336', '#2196f3']
    
    try:
        if vis_type == 'histogram':
            if pd.api.types.is_numeric_dtype(col_data):
                fig = px.histogram(
                    df, 
                    x=column,
                    title=f"📊 Distribution of {column}",
                    nbins=min(30, len(col_data.unique())),
                    color_discrete_sequence=[colors[0]],
                    opacity=0.8
                )
                # Add mean and median lines
                mean_val = col_data.mean()
                median_val = col_data.median()
                
                fig.add_vline(
                    x=mean_val,
                    line_dash="dash",
                    line_color=colors[3],
                    annotation_text=f"Mean: {mean_val:.2f}",
                    annotation_position="top right"
                )
                
                fig.add_vline(
                    x=median_val,
                    line_dash="dot",
                    line_color=colors[2],
                    annotation_text=f"Median: {median_val:.2f}",
                    annotation_position="top left"
                )
            else:
                # For categorical data, show bar chart of value counts
                value_counts = df[column].value_counts().head(15).reset_index()
                value_counts.columns = [column, 'count']
                fig = px.bar(
                    value_counts,
                    x=column,
                    y='count',
                    title=f"📋 Top 15 Values in {column}",
                    color='count',
                    color_continuous_scale='Blues'
                )
                fig.update_layout(xaxis_tickangle=45)
        
        elif vis_type == 'box':
            if pd.api.types.is_numeric_dtype(col_data):
                fig = px.box(
                    df,
                    y=column,
                    title=f"📦 Box Plot of {column}",
                    color_discrete_sequence=[colors[2]]
                )
            else:
                fig = create_empty_chart("Box plot only available for numerical columns")
        
        elif vis_type == 'bar':
            # For categorical data, show top values
            if not pd.api.types.is_numeric_dtype(col_data):
                value_counts = df[column].value_counts().head(20).reset_index()
                value_counts.columns = [column, 'count']
                fig = px.bar(
                    value_counts,
                    x=column,
                    y='count',
                    title=f"📊 Top 20 Values in {column}",
                    color='count',
                    color_continuous_scale='Viridis'
                )
                fig.update_layout(xaxis_tickangle=45)
            else:
                # For numerical, show histogram as bars
                fig = px.histogram(
                    df,
                    x=column,
                    title=f"📊 Distribution of {column}",
                    nbins=20,
                    color_discrete_sequence=[colors[0]]
                )
        
        elif vis_type == 'line':
            if pd.api.types.is_numeric_dtype(col_data):
                sorted_data = col_data.sort_values().reset_index(drop=True)
                fig = px.line(
                    x=range(len(sorted_data)),
                    y=sorted_data,
                    title=f"📈 Line Chart of {column}",
                    color_discrete_sequence=[colors[3]]
                )
                fig.update_layout(
                    xaxis_title="Index",
                    yaxis_title=column
                )
            else:
                fig = create_empty_chart("Line chart only available for numerical columns")
        
        elif vis_type == 'scatter':
            # Need another column for scatter plot
            numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if column in numerical_cols and len(numerical_cols) > 1:
                other_cols = [c for c in numerical_cols if c != column]
                fig = px.scatter(
                    df,
                    x=column,
                    y=other_cols[0],
                    title=f"🎯 {column} vs {other_cols[0]}",
                    color_discrete_sequence=[colors[5]]
                )
            else:
                fig = create_empty_chart("Scatter plot requires another numerical column")
        
        elif vis_type == 'density':
            if pd.api.types.is_numeric_dtype(col_data):
                fig = px.density_contour(
                    df,
                    x=column,
                    title=f"📊 Density Plot of {column}",
                    color_discrete_sequence=[colors[0]]
                )
                fig.update_traces(contours_coloring="fill", contours_showlabels=True)
            else:
                fig = create_empty_chart("Density plot only available for numerical columns")
        
        elif vis_type == 'violin':
            if pd.api.types.is_numeric_dtype(col_data):
                fig = px.violin(
                    df,
                    y=column,
                    title=f"📊 Violin Plot of {column}",
                    color_discrete_sequence=[colors[1]]
                )
            else:
                fig = create_empty_chart("Violin plot only available for numerical columns")
        
        else:
            # Default to histogram
            fig = px.histogram(
                df,
                x=column,
                title=f"📊 Distribution of {column}",
                nbins=30,
                color_discrete_sequence=[colors[0]]
            )
        
        # Update layout for dark theme compatibility
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white', family='Segoe UI, Tahoma, Geneva, Verdana, sans-serif', size=12),
            xaxis=dict(
                gridcolor='rgba(255,255,255,0.1)',
                linecolor='rgba(255,255,255,0.2)',
                title_font=dict(color='white', size=14),
                tickfont=dict(color='white', size=12)
            ),
            yaxis=dict(
                gridcolor='rgba(255,255,255,0.1)',
                linecolor='rgba(255,255,255,0.2)',
                title_font=dict(color='white', size=14),
                tickfont=dict(color='white', size=12)
            ),
            title=dict(
                text=fig.layout.title.text if fig.layout.title else '',
                font=dict(color='white', size=18, family='Segoe UI, Tahoma, Geneva, Verdana, sans-serif'),
                x=0.5,
                xanchor='center'
            ),
            hoverlabel=dict(
                bgcolor='rgba(0,0,0,0.8)',
                font_color='white',
                font_size=12
            ),
            margin=dict(t=50, b=50, l=50, r=50),
            height=450
        )
        
    except Exception as e:
        fig = create_empty_chart(f"Error creating visualization: {str(e)[:50]}")
    
    return fig

def create_column_statistics(df, column):
    """Create statistics display for a specific column"""
    if column is None or column not in df.columns:
        return html.Div("Select a column to see statistics", className="info-text")
    
    col_data = df[column]
    
    # Calculate statistics
    null_count = col_data.isnull().sum()
    non_null_count = len(col_data) - null_count
    null_percentage = (null_count / len(col_data) * 100) if len(col_data) > 0 else 0
    unique_count = col_data.nunique()
    
    # Create statistics display
    if pd.api.types.is_numeric_dtype(col_data):
        numeric_data = col_data.dropna()
        if len(numeric_data) > 0:
            stats = html.Div([
                html.H5(f"Statistics for {column}", className="card-title"),
                html.P(f"Type: Numerical", className="info-text"),
                html.P(f"Null values: {null_count:,} ({null_percentage:.1f}%)", className="info-text"),
                html.P(f"Unique values: {unique_count:,}", className="info-text"),
                html.P(f"Mean: {numeric_data.mean():.2f}", className="info-text"),
                html.P(f"Median: {numeric_data.median():.2f}", className="info-text"),
                html.P(f"Std Dev: {numeric_data.std():.2f}", className="info-text"),
                html.P(f"Min: {numeric_data.min():.2f}", className="info-text"),
                html.P(f"Max: {numeric_data.max():.2f}", className="info-text"),
                html.P(f"Range: {numeric_data.max() - numeric_data.min():.2f}", className="info-text")
            ], className="success-card")
        else:
            stats = html.Div([
                html.H5(f"Statistics for {column}", className="card-title"),
                html.P("All values are null", className="warning-text")
            ], className="warning-card")
    else:
        # Categorical/string data
        stats = html.Div([
            html.H5(f"Statistics for {column}", className="card-title"),
            html.P(f"Type: Categorical", className="info-text"),
            html.P(f"Null values: {null_count:,} ({null_percentage:.1f}%)", className="info-text"),
            html.P(f"Unique values: {unique_count:,}", className="info-text"),
            html.P(f"Most frequent: {col_data.mode().iloc[0] if not col_data.mode().empty else 'N/A'}", 
                  className="info-text")
        ], className="info-card")
    
    return stats

def create_data_quality_chart(df):
    """Create data quality assessment chart"""
    # Calculate quality metrics
    total_rows = len(df)
    total_columns = len(df.columns)
    total_cells = total_rows * total_columns
    
    # Completeness (percentage of non-null values)
    null_count = df.isnull().sum().sum()
    completeness = 100 - (null_count / total_cells * 100) if total_cells > 0 else 100
    
    # Uniqueness (percentage of non-duplicate rows)
    duplicate_rows = df.duplicated().sum()
    uniqueness = 100 - (duplicate_rows / total_rows * 100) if total_rows > 0 else 100
    
    # Variability (percentage of columns with more than 1 unique value)
    variable_cols = sum([1 for col in df.columns if df[col].nunique() > 1])
    variability = (variable_cols / total_columns * 100) if total_columns > 0 else 100
    
    # Create subplot for metrics
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            f'Completeness: {completeness:.1f}%', 
            f'Uniqueness: {uniqueness:.1f}%',
            f'Variability: {variability:.1f}%',
            'Data Types Distribution'
        ],
        specs=[[{'type': 'indicator'}, {'type': 'indicator'}],
               [{'type': 'indicator'}, {'type': 'pie'}]]
    )
    
    # Colors
    colors = ['#4caf50', '#ff9800', '#f44336', '#6495ed', '#9c27b0']
    
    # Completeness gauge
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=completeness,
        domain={'row': 0, 'column': 0},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': colors[0] if completeness > 90 else colors[1] if completeness > 75 else colors[2]},
            'steps': [
                {'range': [0, 75], 'color': 'rgba(244, 67, 54, 0.2)'},
                {'range': [75, 90], 'color': 'rgba(255, 152, 0, 0.2)'},
                {'range': [90, 100], 'color': 'rgba(76, 175, 80, 0.2)'}
            ]
        }
    ), row=1, col=1)
    
    # Uniqueness gauge
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=uniqueness,
        domain={'row': 0, 'column': 1},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': colors[0] if uniqueness > 95 else colors[1] if uniqueness > 90 else colors[2]},
            'steps': [
                {'range': [0, 90], 'color': 'rgba(244, 67, 54, 0.2)'},
                {'range': [90, 95], 'color': 'rgba(255, 152, 0, 0.2)'},
                {'range': [95, 100], 'color': 'rgba(76, 175, 80, 0.2)'}
            ]
        }
    ), row=1, col=2)
    
    # Variability gauge
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=variability,
        domain={'row': 1, 'column': 0},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': colors[0] if variability > 90 else colors[1] if variability > 75 else colors[2]},
            'steps': [
                {'range': [0, 75], 'color': 'rgba(244, 67, 54, 0.2)'},
                {'range': [75, 90], 'color': 'rgba(255, 152, 0, 0.2)'},
                {'range': [90, 100], 'color': 'rgba(76, 175, 80, 0.2)'}
            ]
        }
    ), row=2, col=1)
    
    # Data type distribution pie chart
    type_counts = {
        'Numerical': len(df.select_dtypes(include=[np.number]).columns),
        'Categorical': len(df.select_dtypes(include=['object', 'category']).columns),
        'Datetime': len(df.select_dtypes(include=['datetime', 'datetimetz']).columns)
    }
    type_counts = {k: v for k, v in type_counts.items() if v > 0}
    
    if type_counts:
        fig.add_trace(go.Pie(
            labels=list(type_counts.keys()),
            values=list(type_counts.values()),
            marker=dict(colors=colors[:len(type_counts)]),
            textinfo='label+percent',
            hole=0.3,
            domain={'row': 1, 'column': 1}
        ), row=2, col=2)
    
    fig.update_layout(
        height=500,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', family='Segoe UI, Tahoma, Geneva, Verdana, sans-serif', size=12),
        margin=dict(t=80, b=50, l=50, r=50)
    )
    
    # Update subplot titles
    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(color='white', size=14)
    
    return fig

def create_numerical_statistics_chart(df):
    """Create statistical overview of numerical columns"""
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numerical_cols) == 0:
        return create_empty_chart("No numerical columns found")
    
    # Calculate basic statistics for first 5 numerical columns
    stats_data = []
    display_cols = numerical_cols[:5]
    for col in display_cols:
        col_data = df[col].dropna()
        if len(col_data) > 0:
            stats_data.append({
                'Column': col,
                'Mean': col_data.mean(),
                'Median': col_data.median(),
                'Std': col_data.std(),
                'Min': col_data.min(),
                'Max': col_data.max(),
                'Range': col_data.max() - col_data.min()
            })
    
    if not stats_data:
        return create_empty_chart("No numerical data available")
    
    # Create horizontal bar charts for better readability
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=['Mean Values', 'Median Values', 'Standard Deviation', 
                       'Minimum Values', 'Maximum Values', 'Value Ranges'],
        vertical_spacing=0.15,
        horizontal_spacing=0.15
    )
    
    colors = ['#6495ed', '#5d8aa8', '#4caf50', '#ff9800', '#9c27b0']
    
    # Mean values
    fig.add_trace(go.Bar(
        x=[d['Mean'] for d in stats_data],
        y=[d['Column'][:20] for d in stats_data],
        orientation='h',
        marker_color=colors[0],
        text=[f"{d['Mean']:.2f}" for d in stats_data],
        textposition='auto',
        name='Mean'
    ), row=1, col=1)
    
    # Median values
    fig.add_trace(go.Bar(
        x=[d['Median'] for d in stats_data],
        y=[d['Column'][:20] for d in stats_data],
        orientation='h',
        marker_color=colors[1],
        text=[f"{d['Median']:.2f}" for d in stats_data],
        textposition='auto',
        name='Median'
    ), row=1, col=2)
    
    # Standard deviation
    fig.add_trace(go.Bar(
        x=[d['Std'] for d in stats_data],
        y=[d['Column'][:20] for d in stats_data],
        orientation='h',
        marker_color=colors[2],
        text=[f"{d['Std']:.2f}" for d in stats_data],
        textposition='auto',
        name='Std Dev'
    ), row=2, col=1)
    
    # Minimum values
    fig.add_trace(go.Bar(
        x=[d['Min'] for d in stats_data],
        y=[d['Column'][:20] for d in stats_data],
        orientation='h',
        marker_color=colors[3],
        text=[f"{d['Min']:.2f}" for d in stats_data],
        textposition='auto',
        name='Min'
    ), row=2, col=2)
    
    # Maximum values
    fig.add_trace(go.Bar(
        x=[d['Max'] for d in stats_data],
        y=[d['Column'][:20] for d in stats_data],
        orientation='h',
        marker_color=colors[4],
        text=[f"{d['Max']:.2f}" for d in stats_data],
        textposition='auto',
        name='Max'
    ), row=3, col=1)
    
    # Value ranges
    fig.add_trace(go.Bar(
        x=[d['Range'] for d in stats_data],
        y=[d['Column'][:20] for d in stats_data],
        orientation='h',
        marker_color='#f44336',
        text=[f"{d['Range']:.2f}" for d in stats_data],
        textposition='auto',
        name='Range'
    ), row=3, col=2)
    
    fig.update_layout(
        height=700,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', family='Segoe UI, Tahoma, Geneva, Verdana, sans-serif', size=12),
        showlegend=False,
        margin=dict(t=80, b=50, l=150, r=50)
    )
    
    # Update axes and titles
    for i in range(1, 7):
        fig.update_yaxes(
            tickfont=dict(color='white', size=10),
            gridcolor='rgba(255,255,255,0.1)',
            row=(i-1)//2 + 1,
            col=(i-1)%2 + 1
        )
        fig.update_xaxes(
            tickfont=dict(color='white', size=10),
            gridcolor='rgba(255,255,255,0.1)',
            row=(i-1)//2 + 1,
            col=(i-1)%2 + 1
        )
    
    # Update subplot titles
    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(color='white', size=12)
    
    return fig

def create_missing_values_chart(df):
    """Create chart showing missing values"""
    # Calculate missing values per column
    null_counts = df.isnull().sum()
    null_percentages = (null_counts / len(df)) * 100
    
    # Get columns with missing values
    missing_df = pd.DataFrame({
        'Column': null_percentages.index,
        'Missing_Percentage': null_percentages.values,
        'Missing_Count': null_counts.values
    })
    
    # Sort and take top 20
    missing_df = missing_df[missing_df['Missing_Count'] > 0]
    missing_df = missing_df.sort_values('Missing_Percentage', ascending=True).head(20)
    
    if len(missing_df) == 0:
        # Create a success message
        fig = go.Figure()
        fig.add_annotation(
            text="✅ Excellent! No missing values found in the dataset.",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=20, color="#4caf50")
        )
    else:
        # Create horizontal bar chart for better readability
        fig = go.Figure()
        
        # Color based on missing percentage
        colors = []
        for pct in missing_df['Missing_Percentage']:
            if pct > 50:
                colors.append('#f44336')
            elif pct > 10:
                colors.append('#ff9800')
            else:
                colors.append('#ffc107')
        
        fig.add_trace(go.Bar(
            x=missing_df['Missing_Percentage'],
            y=missing_df['Column'],
            orientation='h',
            marker_color=colors,
            text=[f"{pct:.1f}% ({count:,})" for pct, count in 
                  zip(missing_df['Missing_Percentage'], missing_df['Missing_Count'])],
            textposition='outside',
            textfont=dict(color='white', size=11),
            hovertemplate='<b>%{y}</b><br>Missing: %{x:.1f}%<br>Count: %{customdata:,}<extra></extra>',
            customdata=missing_df['Missing_Count']
        ))
    
    fig.update_layout(
        title="Missing Values Analysis (Columns with Missing Data)",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', family='Segoe UI, Tahoma, Geneva, Verdana, sans-serif', size=12),
        xaxis=dict(
            title='Missing Values (%)',
            gridcolor='rgba(255,255,255,0.1)',
            linecolor='rgba(255,255,255,0.2)',
            title_font=dict(color='white', size=14),
            tickfont=dict(color='white', size=12),
            range=[0, max(missing_df['Missing_Percentage'].max() * 1.1, 10) if len(missing_df) > 0 else 100]
        ),
        yaxis=dict(
            title='Columns',
            gridcolor='rgba(255,255,255,0.1)',
            linecolor='rgba(255,255,255,0.2)',
            title_font=dict(color='white', size=14),
            tickfont=dict(color='white', size=11),
            autorange='reversed'  # Highest on top
        ),
        title_font=dict(color='white', size=18),
        height=max(400, len(missing_df) * 25) if len(missing_df) > 0 else 400,
        margin=dict(t=80, b=50, l=200, r=50)
    )
    
    return fig

def create_data_preview_table(df):
    """Create a preview table of the data"""
    preview_df = df.head(10)
    
    # Create HTML table with better styling
    header = [html.Tr([
        html.Th("#", style={
            'padding': '12px', 
            'fontWeight': 'bold', 
            'textAlign': 'center',
            'backgroundColor': 'rgba(100, 149, 237, 0.4)',
            'color': 'white',
            'border': '1px solid rgba(255,255,255,0.2)'
        })
    ] + [
        html.Th(col[:15] + '...' if len(col) > 15 else col, 
                style={
                    'padding': '12px', 
                    'fontWeight': 'bold', 
                    'textAlign': 'left',
                    'backgroundColor': 'rgba(100, 149, 237, 0.4)',
                    'color': 'white',
                    'border': '1px solid rgba(255,255,255,0.2)'
                }) 
        for col in preview_df.columns[:6]  # Show first 6 columns only
    ])]
    
    rows = []
    for i in range(len(preview_df)):
        row_cells = []
        
        # Row number
        row_cells.append(html.Td(
            i+1, 
            style={
                'padding': '10px', 
                'fontWeight': 'bold', 
                'color': '#6495ed',
                'textAlign': 'center',
                'backgroundColor': 'rgba(100, 149, 237, 0.1)',
                'border': '1px solid rgba(255,255,255,0.1)'
            }
        ))
        
        # Data cells
        for col in preview_df.columns[:6]:
            value = preview_df.iloc[i][col]
            
            if pd.isna(value):
                display_value = "NaN"
                cell_style = {
                    'padding': '10px', 
                    'color': '#ff9800', 
                    'fontStyle': 'italic',
                    'backgroundColor': 'rgba(255, 255, 255, 0.05)',
                    'border': '1px solid rgba(255,255,255,0.1)'
                }
            elif isinstance(value, (int, np.integer)):
                display_value = f"{value:,}"
                cell_style = {
                    'padding': '10px', 
                    'color': '#4caf50',
                    'backgroundColor': 'rgba(255, 255, 255, 0.05)',
                    'border': '1px solid rgba(255,255,255,0.1)'
                }
            elif isinstance(value, (float, np.floating)):
                display_value = f"{value:.2f}"
                cell_style = {
                    'padding': '10px', 
                    'color': '#2196f3',
                    'backgroundColor': 'rgba(255, 255, 255, 0.05)',
                    'border': '1px solid rgba(255,255,255,0.1)'
                }
            else:
                display_value = str(value)[:25] + "..." if len(str(value)) > 25 else str(value)
                cell_style = {
                    'padding': '10px', 
                    'color': 'white',
                    'backgroundColor': 'rgba(255, 255, 255, 0.05)',
                    'border': '1px solid rgba(255,255,255,0.1)'
                }
            
            row_cells.append(html.Td(display_value, style=cell_style))
        
        rows.append(html.Tr(row_cells))
    
    return html.Div([
        html.Table(
            header + rows,
            style={
                'width': '100%',
                'borderCollapse': 'collapse',
                'fontSize': '13px',
                'border': '1px solid rgba(255,255,255,0.2)',
                'borderRadius': '8px',
                'overflow': 'hidden'
            }
        )
    ], style={'overflowX': 'auto', 'maxHeight': '400px'})

def create_empty_chart(message="No data available"):
    """Create an empty chart with message"""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=18, color="white", family='Segoe UI, Tahoma, Geneva, Verdana, sans-serif')
    )
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', family='Segoe UI, Tahoma, Geneva, Verdana, sans-serif'),
        height=300,
        margin=dict(t=50, b=50, l=50, r=50)
    )
    return fig

def create_enhanced_data_summary(df):
    """Create enhanced data summary for reporting"""
    summary = {
        'basic_info': {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024**2)
        },
        'data_types': {
            'numerical': len(df.select_dtypes(include=[np.number]).columns),
            'categorical': len(df.select_dtypes(include=['object', 'category']).columns),
            'datetime': len(df.select_dtypes(include=['datetime', 'datetimetz']).columns),
            'boolean': len(df.select_dtypes(include=['bool']).columns)
        },
        'missing_values': {
            'total_nulls': df.isnull().sum().sum(),
            'columns_with_nulls': (df.isnull().sum() > 0).sum(),
            'null_percentage': (df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100) if len(df.columns) > 0 else 0
        },
        'column_samples': {}
    }
    
    # Add sample values for each column
    for col in df.columns[:10]:  # First 10 columns
        non_null = df[col].dropna()
        if len(non_null) > 0:
            summary['column_samples'][col] = str(non_null.iloc[0])
        else:
            summary['column_samples'][col] = "NaN"
    
    return summary