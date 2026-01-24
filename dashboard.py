"""
Dashboard components for data visualization
"""
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc


def create_dashboard(df):
    """Create interactive dashboard from cleaned data"""
    
    if df.empty:
        return html.Div("No data available for dashboard")
    
    # Get column types
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Create tabs for different visualizations
    tabs = []
    
    # Tab 1: Overview
    overview_tab = dcc.Tab(label='Overview', children=[
        html.Div([
            dbc.Row([
                dbc.Col([
                    create_summary_card(df)
                ], width=12)
            ]),
            
            dbc.Row([
                dbc.Col([
                    create_data_types_chart(df)
                ], width=6),
                dbc.Col([
                    create_missing_values_chart(df)
                ], width=6)
            ], className="mt-3"),
            
            dbc.Row([
                dbc.Col([
                    create_correlation_matrix(df)
                ], width=12)
            ], className="mt-3")
        ])
    ])
    tabs.append(overview_tab)
    
    # Tab 2: Numerical Analysis
    if numerical_cols:
        numerical_tab = dcc.Tab(label='Numerical Analysis', children=[
            html.Div([
                dbc.Row([
                    dbc.Col([
                        html.H4("Select Numerical Column"),
                        dcc.Dropdown(
                            id='num-col-selector',
                            options=[{'label': col, 'value': col} for col in numerical_cols],
                            value=numerical_cols[0] if numerical_cols else None
                        )
                    ], width=6),
                    
                    dbc.Col([
                        html.H4("Chart Type"),
                        dcc.Dropdown(
                            id='num-chart-type',
                            options=[
                                {'label': 'Histogram', 'value': 'histogram'},
                                {'label': 'Box Plot', 'value': 'box'},
                                {'label': 'Violin Plot', 'value': 'violin'},
                                {'label': 'ECDF', 'value': 'ecdf'}
                            ],
                            value='histogram'
                        )
                    ], width=6)
                ]),
                
                dcc.Graph(id='numerical-chart'),
                
                dbc.Row([
                    dbc.Col([
                        create_statistics_table(df, numerical_cols)
                    ], width=12)
                ], className="mt-3")
            ])
        ])
        tabs.append(numerical_tab)
    
    # Tab 3: Categorical Analysis
    if categorical_cols:
        categorical_tab = dcc.Tab(label='Categorical Analysis', children=[
            html.Div([
                dbc.Row([
                    dbc.Col([
                        html.H4("Select Categorical Column"),
                        dcc.Dropdown(
                            id='cat-col-selector',
                            options=[{'label': col, 'value': col} for col in categorical_cols],
                            value=categorical_cols[0] if categorical_cols else None
                        )
                    ], width=6),
                    
                    dbc.Col([
                        html.H4("Select Numerical Column (for comparison)"),
                        dcc.Dropdown(
                            id='cat-num-col-selector',
                            options=[{'label': col, 'value': col} for col in numerical_cols],
                            value=numerical_cols[0] if numerical_cols else None
                        )
                    ], width=6)
                ]),
                
                dcc.Graph(id='categorical-chart'),
                
                dbc.Row([
                    dbc.Col([
                        dcc.Graph(id='value-counts-chart')
                    ], width=12)
                ], className="mt-3")
            ])
        ])
        tabs.append(categorical_tab)
    
    # Tab 4: Relationships
    if len(numerical_cols) >= 2:
        relationships_tab = dcc.Tab(label='Relationships', children=[
            html.Div([
                dbc.Row([
                    dbc.Col([
                        html.H4("X-axis Column"),
                        dcc.Dropdown(
                            id='x-col-selector',
                            options=[{'label': col, 'value': col} for col in numerical_cols],
                            value=numerical_cols[0]
                        )
                    ], width=4),
                    
                    dbc.Col([
                        html.H4("Y-axis Column"),
                        dcc.Dropdown(
                            id='y-col-selector',
                            options=[{'label': col, 'value': col} for col in numerical_cols],
                            value=numerical_cols[1] if len(numerical_cols) > 1 else numerical_cols[0]
                        )
                    ], width=4),
                    
                    dbc.Col([
                        html.H4("Color By (Optional)"),
                        dcc.Dropdown(
                            id='color-col-selector',
                            options=[{'label': 'None', 'value': 'none'}] + 
                                    [{'label': col, 'value': col} for col in categorical_cols],
                            value='none'
                        )
                    ], width=4)
                ]),
                
                dcc.Graph(id='scatter-plot'),
                
                dbc.Row([
                    dbc.Col([
                        html.H4("Correlation Analysis"),
                        dcc.Graph(figure=create_correlation_heatmap(df))
                    ], width=12)
                ], className="mt-3")
            ])
        ])
        tabs.append(relationships_tab)
    
    return html.Div([
        dcc.Tabs(tabs)
    ])


def create_summary_card(df):
    """Create summary statistics card"""
    card = dbc.Card([
        dbc.CardBody([
            html.H4("Dataset Summary", className="card-title"),
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H2(f"{df.shape[0]:,}", className="text-primary", 
                               style={'color': '#007bff'}),
                        html.P("Total Rows", className="text-muted")
                    ], className="text-center")
                ], width=3),
                
                dbc.Col([
                    html.Div([
                        html.H2(f"{df.shape[1]}", className="text-success",
                               style={'color': '#28a745'}),
                        html.P("Total Columns", className="text-muted")
                    ], className="text-center")
                ], width=3),
                
                dbc.Col([
                    html.Div([
                        html.H2(f"{df.select_dtypes(include=[np.number]).shape[1]}", 
                               className="text-info", style={'color': '#17a2b8'}),
                        html.P("Numerical Columns", className="text-muted")
                    ], className="text-center")
                ], width=3),
                
                dbc.Col([
                    html.Div([
                        html.H2(f"{df.select_dtypes(include=['object', 'category']).shape[1]}", 
                               className="text-warning", style={'color': '#ffc107'}),
                        html.P("Categorical Columns", className="text-muted")
                    ], className="text-center")
                ], width=3)
            ])
        ])
    ])
    return card


def create_data_types_chart(df):
    """Create chart showing data type distribution"""
    dtype_counts = df.dtypes.value_counts()
    
    fig = px.pie(
        values=dtype_counts.values,
        names=dtype_counts.index.astype(str),
        title="Data Type Distribution",
        hole=0.4
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(height=300)
    
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
            labels={'x': 'Column', 'y': 'Missing Count'}
        )
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
        fig.update_layout(title="Missing Values by Column")
    
    fig.update_layout(height=300)
    return dcc.Graph(figure=fig)


def create_correlation_matrix(df):
    """Create correlation matrix heatmap"""
    numerical_df = df.select_dtypes(include=[np.number])
    
    if numerical_df.shape[1] < 2:
        fig = go.Figure()
        fig.add_annotation(
            text="Not enough numerical columns for correlation analysis",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False
        )
        fig.update_layout(title="Correlation Matrix", height=400)
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
        yaxis_title="Columns"
    )
    
    return dcc.Graph(figure=fig)


def create_statistics_table(df, numerical_cols):
    """Create statistics table for numerical columns"""
    if not numerical_cols:
        return html.Div("No numerical columns available")
    
    stats_df = df[numerical_cols].describe().round(2).T
    stats_df = stats_df.reset_index()
    stats_df.columns = ['Column', 'Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max']
    
    # Create HTML table
    table = html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in stats_df.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(stats_df.iloc[i][col]) for col in stats_df.columns
            ]) for i in range(len(stats_df))
        ])
    ], className="table table-striped table-bordered table-hover table-sm")
    
    return html.Div([
        html.H5("Descriptive Statistics"),
        table
    ])


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
        height=400
    )
    
    return fig


# These functions are not used directly but are called by callbacks in app.py
def create_numerical_chart(df, col, chart_type='histogram'):
    """Create chart for numerical column"""
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


def create_categorical_chart(df, cat_col, num_col=None):
    """Create chart for categorical column"""
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


def create_scatter_plot(df, x_col, y_col, color_col=None):
    """Create scatter plot"""
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