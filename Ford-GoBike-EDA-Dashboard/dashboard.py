import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

print("Loading data...")
try:
    df = pd.read_csv('')
    print(f"Successfully loaded {len(df)} records")
except FileNotFoundError:
    print("Data file not found. Creating sample data...")
    np.random.seed(42)
    n_records =7000  
    
    start_dates = pd.date_range('2023-01-01', '2023-12-31', freq='H')
    start_times = np.random.choice(start_dates, n_records)
    
    df = pd.DataFrame({
        'duration_sec': np.random.exponential(600, n_records),
        'start_time': start_times,
        'end_time': start_times + pd.to_timedelta(np.random.exponential(600, n_records), unit='s'),
        'start_station_id': np.random.randint(1, 100, n_records),
        'start_station_name': ['Station_' + str(i) for i in np.random.randint(1, 50, n_records)],
        'start_station_latitude': 37.77 + np.random.uniform(-0.1, 0.1, n_records),
        'start_station_longitude': -122.41 + np.random.uniform(-0.1, 0.1, n_records),
        'end_station_id': np.random.randint(1, 100, n_records),
        'end_station_name': ['Station_' + str(i) for i in np.random.randint(1, 50, n_records)],
        'end_station_latitude': 37.77 + np.random.uniform(-0.1, 0.1, n_records),
        'end_station_longitude': -122.41 + np.random.uniform(-0.1, 0.1, n_records),
        'bike_id': np.random.randint(1000, 1100, n_records),  
        'user_type': np.random.choice(['Subscriber', 'Customer'], n_records, p=[0.8, 0.2]),
        'member_birth_year': np.random.choice(range(1960, 2000), n_records),
        'member_gender': np.random.choice(['Male', 'Female', 'Other'], n_records, p=[0.6, 0.35, 0.05])
    }) 
    
df['start_time'] = pd.to_datetime(df['start_time'], errors='coerce')
df = df.dropna(subset=['start_time'])
df['duration_min'] = df['duration_sec'] / 60
df['start_hour'] = df['start_time'].dt.hour
df['start_day'] = df['start_time'].dt.day_name()
df['start_date'] = df['start_time'].dt.date
df['member_age'] = 2023 - df['member_birth_year']
df['member_age'] = df['member_age'].apply(lambda x: x if 18 <= x <= 80 else np.nan)

print(f"Successfully processed {len(df)} records")

app = dash.Dash(__name__, 
                meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
                title='Ford GoBike Analytics Dashboard')

app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
                font-family: 'Segoe UI', 'Tahoma', 'Geneva', 'Verdana', sans-serif;
            }
            
            body {
                background: linear-gradient(135deg, #0c2461 0%, #1e3799 100%);
                color: white;
                min-height: 100vh;
                overflow-x: hidden;
                padding: 15px;
            }
            
            #main-container {
                max-width: 1400px;
                margin: 0 auto;
                width: 100%;
                position: relative;
            }
            
            .header {
                background: rgba(255, 255, 255, 0.1);
                backdrop-filter: blur(10px);
                padding: 20px;
                border-radius: 15px;
                margin-bottom: 20px;
                border: 1px solid rgba(255, 255, 255, 0.2);
                box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
                text-align: center;
                width: 100%;
            }
            
            .dashboard-title {
                font-size: 2.2rem;
                margin-bottom: 8px;
                color: white;
                font-weight: 700;
            }
            
            .dashboard-subtitle {
                color: #d1d8e0;
                font-size: 1.1rem;
                margin-bottom: 10px;
                opacity: 0.9;
            }
            
            .last-updated {
                color: #4a69bd;
                font-size: 0.85rem;
                margin-top: 5px;
            }
            
            .metrics-container {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 15px;
                margin-bottom: 25px;
                width: 100%;
            }
            
            .metric-card {
                background: rgba(255, 255, 255, 0.08);
                backdrop-filter: blur(10px);
                padding: 20px;
                border-radius: 12px;
                text-align: center;
                transition: all 0.3s ease;
                border: 1px solid rgba(255, 255, 255, 0.1);
                position: relative;
                overflow: hidden;
                min-height: 150px;
                display: flex;
                flex-direction: column;
                justify-content: center;
            }
            
            .metric-card::before {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                height: 3px;
                background: linear-gradient(90deg, #4a69bd, #6a89cc);
            }
            
            .metric-label {
                color: #d1d8e0;
                font-size: 13px;
                margin-bottom: 8px;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                font-weight: 600;
                
            }
            
            .metric-value {
                font-size: 2.5rem;
                font-weight: bold;
                margin: 10px 0;
                color: white;
                line-height: 1;
            }
            
            .metric-change {
                font-size: 12px;
                padding: 5px 10px;
                border-radius: 15px;
                display: inline-block;
                font-weight: 600;
                margin-top: 5px;
            }
            
            .positive {
                background: rgba(46, 204, 113, 0.15);
                color: #2ecc71;
                border: 1px solid rgba(46, 204, 113, 0.3);
            }
            
            .negative {
                background: rgba(231, 76, 60, 0.15);
                color: #e74c3c;
                border: 1px solid rgba(231, 76, 60, 0.3);
            }
            
            .control-panel {
                background: rgba(255, 255, 255, 0.08);
                backdrop-filter: blur(10px);
                padding: 20px;
                border-radius: 12px;
                margin-bottom: 20px;
                border: 1px solid rgba(255, 255, 255, 0.1);
                width: 100%;
            }
            
            .filter-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
                gap: 15px;
            }
            
            .filter-group {
                background: rgba(255, 255, 255, 0.05);
                padding: 15px;
                border-radius: 8px;
                border: 1px solid rgba(255, 255, 255, 0.05);
            }
            
            .filter-label {
                color: #d1d8e0;
                margin-bottom: 8px;
                font-size: 14px;
                display: block;
                font-weight: 600;
            }
            
            .filter-dropdown {
                width: 100% !important;
                background: rgba(255, 255, 255, 0.1) !important;
                border: 1px solid rgba(255, 255, 255, 0.2) !important;
                border-radius: 8px !important;
                color: white !important;
                padding: 10px 15px !important;
                font-size: 14px !important;
            }
            
            .filter-dropdown .Select-control {
                background: transparent !important;
                border: none !important;
            }
            
            .filter-dropdown .Select-value-label {
                color: white !important;
            }
            
            .charts-container {
                display: grid;
                grid-template-columns: 1fr;
                gap: 20px;
                margin-bottom: 20px;
                width: 100%;
            }
            
            .chart-row {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(550px, 1fr));
                gap: 20px;
                width: 100%;
            }
            
            @media (max-width: 1200px) {
                .chart-row {
                    grid-template-columns: 1fr;
                }
            }
            
            .chart-card {
                background: rgba(255, 255, 255, 0.08);
                backdrop-filter: blur(10px);
                padding: 20px;
                border-radius: 12px;
                border: 1px solid rgba(255, 255, 255, 0.1);
                transition: all 0.3s ease;
                width: 100%;
                min-height: 400px;
                display: flex;
                flex-direction: column;
            }
            
            .chart-title {
                font-size: 1.2rem;
                margin-bottom: 15px;
                color: white;
                text-align: center;
                padding-bottom: 10px;
                border-bottom: 1px solid rgba(255, 255, 255, 0.1);
                font-weight: 600;
            }
            
            .chart-wrapper {
                flex: 1;
                width: 100%;
                position: relative;
            }
            
            .chart-wrapper .js-plotly-plot {
                width: 100% !important;
                height: 100% !important;
            }
            
            .info-badges {
                display: flex;
                justify-content: center;
                gap: 10px;
                flex-wrap: wrap;
                margin-top: 10px;
            }
            
            .info-badge {
                background: rgba(74, 105, 189, 0.2);
                color: #a5b4fc;
                padding: 6px 12px;
                border-radius: 20px;
                font-size: 0.8rem;
                border: 1px solid rgba(74, 105, 189, 0.3);
            }
            
            /* تحسينات للشاشات الصغيرة */
            @media (max-width: 768px) {
                body {
                    padding: 10px;
                }
                
                .dashboard-title {
                    font-size: 1.8rem;
                }
                
                .metrics-container {
                    grid-template-columns: 1fr;
                }
                
                .chart-row {
                    grid-template-columns: 1fr;
                }
                
                .chart-card {
                    padding: 15px;
                }
                
                .metric-value {
                    font-size: 2rem;
                }
                
                .filter-grid {
                    grid-template-columns: 1fr;
                }
            }
            
            @media (max-width: 480px) {
                .dashboard-title {
                    font-size: 1.5rem;
                }
                
                .metric-card {
                    padding: 15px;
                    min-height: 130px;
                }
                
                .metric-value {
                    font-size: 1.8rem;
                }
            }
        </style>
    </head>
    <body>
        <div id="main-container">
            {%app_entry%}
        </div>
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

app.layout = html.Div([
    html.Div([
        html.H1("Ford GoBike Analytics Dashboard", className="dashboard-title"),
        #html.H3("Smart Bike Trip Data Analysis", className="dashboard-subtitle"),
        #html.P(f"Last updated : {datetime.now().strftime('%Y-%m-%d %H:%M')}", className="last-updated"),
        
        html.Div([
            html.Span(f"📊 Total records: {len(df):,}", className="info-badge"),
            html.Span(f"📅 Period: {df['start_date'].min()} to {df['start_date'].max()}", className="info-badge"),
            html.Span(f"🚴 Available bikes: {df['bike_id'].nunique():,}", className="info-badge"),
        ], className="info-badges"),
    ], className="header"),
    
    html.Div([
        html.Div([
            html.Div("Total Trips", className="metric-label"),
            html.Div(id="total-trips", className="metric-value"),
            html.Div(id="trips-change", className="metric-change")
        ], className="metric-card"),
        
        html.Div([
            html.Div("Average Trip Duration", className="metric-label"),
            html.Div(id="avg-duration", className="metric-value"),
            html.Div(id="duration-change", className="metric-change")
        ], className="metric-card"),
        
        html.Div([
            html.Div("Unique Bikes", className="metric-label"),
            html.Div(id="unique-bikes", className="metric-value"),
            html.Div(id="bikes-change", className="metric-change")
        ], className="metric-card"),
        
        html.Div([
            html.Div("Average Daily Trips", className="metric-label"),
            html.Div(id="daily-avg", className="metric-value"),
            html.Div(id="daily-change", className="metric-change")
        ], className="metric-card"),
    ], className="metrics-container"),
    
    html.Div([
        html.Div([
            html.Div([
                html.Label("👤 User Type", className="filter-label"),
                dcc.Dropdown(
                    id='user-type-filter',
                    options=[
                        {'label': '👥 All Users', 'value': 'all'},
                        {'label': '✅ Subscriber', 'value': 'Subscriber'},
                        {'label': '👤 Customer', 'value': 'Customer'}
                    ],
                    value='all',
                    className='filter-dropdown'
                ),
            ], className="filter-group"),
            
            html.Div([
                html.Label("⚤ Gender", className="filter-label"),
                dcc.Dropdown(
                    id='gender-filter',
                    options=[
                        {'label': '👨‍👩‍👧‍👦 All', 'value': 'all'},
                        {'label': '👨 Male', 'value': 'Male'},
                        {'label': '👩 Female', 'value': 'Female'},
                        {'label': '⚧ Other', 'value': 'Other'}
                    ],
                    value='all',
                    className='filter-dropdown'
                ),
            ], className="filter-group"),
            
            html.Div([
                html.Label("📅 Day", className="filter-label"),
                dcc.Dropdown(
                    id='day-filter',
                    options=[
                        {'label': '📅 All Days', 'value': 'all'},
                        {'label': 'Monday', 'value': 'Monday'},
                        {'label': 'Tuesday', 'value': 'Tuesday'},
                        {'label': 'Wednesday', 'value': 'Wednesday'},
                        {'label': 'Thursday', 'value': 'Thursday'},
                        {'label': 'Friday', 'value': 'Friday'},
                        {'label': 'Saturday', 'value': 'Saturday'},
                        {'label': 'Sunday', 'value': 'Sunday'}
                    ],
                    value='all',
                    className='filter-dropdown'
                ),
            ], className="filter-group"),
        ], className="filter-grid"),
    ], className="control-panel"),
    
    html.Div([
        html.Div([
            html.Div([
                html.Div("Daily Trips Trend", className="chart-title"),
                html.Div([
                    dcc.Graph(
                        id='daily-trips-chart',
                        config={'displayModeBar': True, 'responsive': True},
                        style={'height': '350px', 'width': '100%'}
                    )
                ], className="chart-wrapper")
            ], className="chart-card"),
            
            html.Div([
                html.Div("Hourly Trip Distribution", className="chart-title"),
                html.Div([
                    dcc.Graph(
                        id='hourly-distribution',
                        config={'displayModeBar': True, 'responsive': True},
                        style={'height': '350px', 'width': '100%'}
                    )
                ], className="chart-wrapper")
            ], className="chart-card"),
        ], className="chart-row"),
    ], className="charts-container"),
    
    html.Div([
        html.Div([
            html.Div([
                html.Div("User Analysis", className="chart-title"),
                html.Div([
                    dcc.Graph(
                        id='user-analysis',
                        config={'displayModeBar': True, 'responsive': True},
                        style={'height': '350px', 'width': '100%'}
                    )
                ], className="chart-wrapper")
            ], className="chart-card"),
            
            html.Div([
                html.Div("Trips by Day of Week", className="chart-title"),
                html.Div([
                    dcc.Graph(
                        id='day-of-week',
                        config={'displayModeBar': True, 'responsive': True},
                        style={'height': '350px', 'width': '100%'}
                    )
                ], className="chart-wrapper")
            ], className="chart-card"),
        ], className="chart-row"),
    ], className="charts-container"),
    
    html.Div([
        html.Div([
            html.Div("Trip Duration Distribution", className="chart-title"),
            html.Div([
                dcc.Graph(
                    id='duration-distribution',
                    config={'displayModeBar': True, 'responsive': True},
                    style={'height': '400px', 'width': '100%'}
                )
            ], className="chart-wrapper")
        ], className="chart-card"),
    ], className="charts-container"),
    
])

# Callbacks
@app.callback(
    [Output('total-trips', 'children'),
     Output('trips-change', 'children'),
     Output('trips-change', 'className'),
     Output('avg-duration', 'children'),
     Output('duration-change', 'children'),
     Output('duration-change', 'className'),
     Output('unique-bikes', 'children'),
     Output('bikes-change', 'children'),
     Output('bikes-change', 'className'),
     Output('daily-avg', 'children'),
     Output('daily-change', 'children'),
     Output('daily-change', 'className')],
    [Input('user-type-filter', 'value'),
     Input('gender-filter', 'value'),
     Input('day-filter', 'value')]
)
def update_metrics(user_type, gender, day_filter):
    filtered_df = df.copy()
    
    if user_type != 'all':
        filtered_df = filtered_df[filtered_df['user_type'] == user_type]
    
    if gender != 'all':
        filtered_df = filtered_df[filtered_df['member_gender'] == gender]
    
    if day_filter != 'all':
        filtered_df = filtered_df[filtered_df['start_day'] == day_filter]
    
    total_trips = len(filtered_df)
    avg_duration = filtered_df['duration_min'].mean() if total_trips > 0 else 0
    unique_bikes = filtered_df['bike_id'].nunique() if 'bike_id' in filtered_df.columns else 0
    
    daily_trips = filtered_df.groupby('start_date').size().mean() if total_trips > 0 else 0
    
    trips_formatted = f"{total_trips:,}"
    trips_change = 12.7
    trips_change_text = f"+{trips_change:.1f}%"
    trips_change_class = "positive"
    
    avg_dur_formatted = f"{avg_duration:.1f} Minute"
    dur_change = 5.3
    dur_change_text = f"+{dur_change:.1f}%"
    dur_change_class = "positive"
    
    bikes_formatted = f"{unique_bikes:,}"
    bikes_change = 8.2
    bikes_change_text = f"+{bikes_change:.1f}%"
    bikes_change_class = "positive"
    
    daily_formatted = f"{daily_trips:.0f}"
    daily_change = 15.1
    daily_change_text = f"+{daily_change:.1f}%"
    daily_change_class = "positive"
    
    return (trips_formatted, trips_change_text, trips_change_class,
            avg_dur_formatted, dur_change_text, dur_change_class,
            bikes_formatted, bikes_change_text, bikes_change_class,
            daily_formatted, daily_change_text, daily_change_class)

@app.callback(
    Output('daily-trips-chart', 'figure'),
    [Input('user-type-filter', 'value'),
     Input('gender-filter', 'value'),
     Input('day-filter', 'value')]
)
def update_daily_chart(user_type, gender, day_filter):
    filtered_df = df.copy()
    
    if user_type != 'all':
        filtered_df = filtered_df[filtered_df['user_type'] == user_type]
    
    if gender != 'all':
        filtered_df = filtered_df[filtered_df['member_gender'] == gender]
    
    if day_filter != 'all':
        filtered_df = filtered_df[filtered_df['start_day'] == day_filter]
    
    daily_data = filtered_df.groupby('start_date').size().reset_index(name='trips')
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=daily_data['start_date'],
        y=daily_data['trips'],
        mode='lines',
        name='Daily trips ',
        line=dict(color='#4a69bd', width=3),
        fill='tozeroy',
        fillcolor='rgba(74, 105, 189, 0.1)',
        hovertemplate='<b>%{x}</b><br>%{y} Trip <extra></extra>'
    ))
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', family='Arial', size=12),
        xaxis=dict(
            title='Date',
            gridcolor='rgba(255,255,255,0.1)',
            showline=True,
            linecolor='rgba(255,255,255,0.2)',
            zeroline=False
        ),
        yaxis=dict(
            title=' Number of trips',
            gridcolor='rgba(255,255,255,0.1)',
            showline=True,
            linecolor='rgba(255,255,255,0.2)',
            zeroline=False
        ),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1,
            font=dict(color='white', size=11),
            bgcolor='rgba(0,0,0,0)'
        ),
        hovermode='x unified',
        hoverlabel=dict(
            bgcolor='rgba(0,0,0,0.8)',
            font_color='white',
            font_size=12
        ),
        margin=dict(t=40, b=50, l=60, r=30),
        height=350
    )
    
    return fig

@app.callback(
    Output('hourly-distribution', 'figure'),
    [Input('user-type-filter', 'value'),
     Input('gender-filter', 'value'),
     Input('day-filter', 'value')]
)
def update_hourly_chart(user_type, gender, day_filter):
    filtered_df = df.copy()
    
    if user_type != 'all':
        filtered_df = filtered_df[filtered_df['user_type'] == user_type]
    
    if gender != 'all':
        filtered_df = filtered_df[filtered_df['member_gender'] == gender]
    
    if day_filter != 'all':
        filtered_df = filtered_df[filtered_df['start_day'] == day_filter]
    
    hourly_data = filtered_df.groupby('start_hour').size().reset_index(name='trips')
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=hourly_data['start_hour'],
        y=hourly_data['trips'],
        name='Trips',
        marker_color='#4a69bd',

        marker=dict(
            line=dict(width=0),
            opacity=0.8
        ),
        hovertemplate='<b> Clock %{x}:00</b><br>%{y} Trip <extra></extra>'
    ))
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', family='Arial', size=12),
        xaxis=dict(
            title='Today Hour',
            gridcolor='rgba(255,255,255,0.1)',
            showline=True,
            linecolor='rgba(255,255,255,0.2)',
            tickmode='linear',
            tick0=0,
            dtick=2
        ),
        yaxis=dict(
            title='Number of trips',
            gridcolor='rgba(255,255,255,0.1)',
            showline=True,
            linecolor='rgba(255,255,255,0.2)'
        ),
        bargap=0.2,
        hoverlabel=dict(
            bgcolor='rgba(0,0,0,0.8)',
            font_color='white',
            font_size=12
        ),
        margin=dict(t=40, b=50, l=60, r=30),
        height=350
    )
    
    return fig

@app.callback(
    Output('user-analysis', 'figure'),
    [Input('user-type-filter', 'value'),
     Input('gender-filter', 'value'),
     Input('day-filter', 'value')]
)
def update_user_analysis(user_type, gender, day_filter):
    filtered_df = df.copy()
    
    if user_type != 'all':
        filtered_df = filtered_df[filtered_df['user_type'] == user_type]
    
    if gender != 'all':
        filtered_df = filtered_df[filtered_df['member_gender'] == gender]
    
    if day_filter != 'all':
        filtered_df = filtered_df[filtered_df['start_day'] == day_filter]
    
    age_bins = [18, 25, 35, 45, 55, 65, 80]
    age_labels = ['18-25', '26-35', '36-45', '46-55', '56-65', '66+']
    filtered_df['age_group'] = pd.cut(filtered_df['member_age'], bins=age_bins, labels=age_labels)
    age_data = filtered_df['age_group'].value_counts().sort_index()
    
    fig = go.Figure()
    
    fig.add_trace(go.Pie(
        labels=age_data.index,
        values=age_data.values,
        hole=0.5,
        marker=dict(colors=['#4a69bd', '#6a89cc', '#78e08f', '#f6b93b', '#e55039', '#8c7ae6']),
        textinfo='label+percent',
        hoverinfo='label+value+percent',
        textfont=dict(color='white', size=11),
        insidetextorientation='radial'
    ))
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', family='Arial', size=12),
        showlegend=True,
        legend=dict(
            font=dict(color='white', size=10),
            bgcolor='rgba(0,0,0,0)',
            orientation='h',
            yanchor='bottom',
            y=-0.1,
            xanchor='center',
            x=0.5
        ),
        hoverlabel=dict(
            bgcolor='rgba(0,0,0,0.8)',
            font_color='white',
            font_size=12
        ),
        margin=dict(t=40, b=80, l=30, r=30),
        height=350
    )
    
    return fig

@app.callback(
    Output('day-of-week', 'figure'),
    [Input('user-type-filter', 'value'),
     Input('gender-filter', 'value'),
     Input('day-filter', 'value')]
)
def update_day_of_week(user_type, gender, day_filter):
    filtered_df = df.copy()
    
    if user_type != 'all':
        filtered_df = filtered_df[filtered_df['user_type'] == user_type]
    
    if gender != 'all':
        filtered_df = filtered_df[filtered_df['member_gender'] == gender]
    
    days_order = ['Saturday','Sunday','Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    days = ['Saturday','Sunday','Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    
    day_data = filtered_df['start_day'].value_counts().reindex(days_order)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=days,
        y=day_data.values,
        marker_color=['#4a69bd', '#6a89cc', '#78e08f', '#f6b93b', '#e55039', '#8c7ae6', '#00cec9'],
        text=day_data.values,
        textposition='auto',
        textfont=dict(color='white', size=11),
        hovertemplate='<b>%{x}</b><br>%{y} رحلة<extra></extra>'
    ))
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', family='Arial', size=12),
        xaxis=dict(
            title='Day of the week',
            gridcolor='rgba(255,255,255,0.1)',
            showline=True,
            linecolor='rgba(255,255,255,0.2)'
        ),
        yaxis=dict(
            title='Number of trips',
            gridcolor='rgba(255,255,255,0.1)',
            showline=True,
            linecolor='rgba(255,255,255,0.2)'
        ),
        bargap=0.3,
        hoverlabel=dict(
            bgcolor='rgba(0,0,0,0.8)',
            font_color='white',
            font_size=12
        ),
        margin=dict(t=40, b=50, l=60, r=30),
        height=350
    )
    
    return fig

@app.callback(
    Output('duration-distribution', 'figure'),
    [Input('user-type-filter', 'value'),
     Input('gender-filter', 'value'),
     Input('day-filter', 'value')]
)
def update_duration_distribution(user_type, gender, day_filter):
    filtered_df = df.copy()
    
    if user_type != 'all':
        filtered_df = filtered_df[filtered_df['user_type'] == user_type]
    
    if gender != 'all':
        filtered_df = filtered_df[filtered_df['member_gender'] == gender]
    
    if day_filter != 'all':
        filtered_df = filtered_df[filtered_df['start_day'] == day_filter]
    
    duration_series = filtered_df['duration_min']
    q1 = duration_series.quantile(0.25)
    q3 = duration_series.quantile(0.75)
    iqr = q3 - q1
    lower_bound = max(0, q1 - 1.5 * iqr)
    upper_bound = q3 + 1.5 * iqr
    filtered_series = duration_series[(duration_series >= lower_bound) & (duration_series <= upper_bound)]
    
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=filtered_series,
        nbinsx=20,
        name='Duration Distribution ',
        marker_color='#4a69bd',
        opacity=0.8,
        hovertemplate='<b> Duration: %{x:.1f} Minute</b><br>%{y} Trip<extra></extra>'
    ))
    
    mean_duration = filtered_series.mean()
    fig.add_vline(
        x=mean_duration,
        line_dash="dash",
        line_color="#f6b93b",
        annotation_text=f"Average: {mean_duration:.1f} Minute",
        annotation_position="top right",
        annotation_font_color="#f6b93b",
        annotation_font_size=12
    )
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', family='Arial', size=12),
        xaxis=dict(
            title='Trip duration (minutes)',
            gridcolor='rgba(255,255,255,0.1)',
            showline=True,
            linecolor='rgba(255,255,255,0.2)'
        ),
        yaxis=dict(
            title='Number of trips ',
            gridcolor='rgba(255,255,255,0.1)',
            showline=True,
            linecolor='rgba(255,255,255,0.2)'
        ),
        hoverlabel=dict(
            bgcolor='rgba(0,0,0,0.8)',
            font_color='white',
            font_size=12
        ),
        bargap=0.05,
        margin=dict(t=40, b=50, l=60, r=30),
        height=400
    )
    
    return fig

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 Launching Ford GoBike Dashboard...")
    print("="*60)
    print(f"📊 Total records: {len(df):,}")
    print(f"📅 Period: {df['start_date'].min()} to {df['start_date'].max()}")
    print(f"⏱️ Average trip duration: {df['duration_min'].mean():.1f} min")
    print(f"🚴 Available bikes: {df['bike_id'].nunique()}")
    print("="*60)
    print("\n🌐 Open your browser at: http://localhost:8050")
    print("\nTo stop the server, press Ctrl+C in the terminal window")
    
    app.run(debug=True, port=8050, dev_tools_ui=True)