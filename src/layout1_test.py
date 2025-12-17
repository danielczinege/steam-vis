import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from dash import Dash, dcc, html

# ==========================================
# 1. Stacked Area Chart
# ==========================================
def create_stacked_area_chart():
    years = list(range(2015, 2026))
    genres = ['FPS', 'RPG', 'Horror', 'Strategy']
    data = []
    
    for year in years:
        for genre in genres:
            count = np.random.randint(50, 200) + (year - 2015) * 20
            data.append({'Year': year, 'Genre': genre, 'Count': count})
            
    df = pd.DataFrame(data)

    fig = px.area(
        df, 
        x="Year", 
        y="Count", 
        color="Genre", 
        title="Games Released per Genre per Year"
    )
    # Align height with the bubble chart for symmetry
    fig.update_layout(height=400, margin=dict(l=40, r=40, t=40, b=40))
    return fig

# ==========================================
# 2. Bubble Chart (RESIZED)
# ==========================================
def create_bubble_chart():
    languages = {
        'English': 1000, 'Chinese': 800, 'Russian': 600, 'Spanish': 500,
        'Japanese': 450, 'German': 400, 'Korean': 300, 'French': 250,
        'Italian': 200, 'Portuguese': 150, 'Polish': 100, 'Turkish': 80
    }

    sorted_langs = sorted(languages.items(), key=lambda x: x[1], reverse=True)
    
    bubbles = []
    
    # Collision Detection (Standard)
    for i, (lang, count) in enumerate(sorted_langs):
        r = np.sqrt(count) 
        
        if not bubbles:
            bubbles.append({'Language': lang, 'Games': count, 'x': 0, 'y': 0, 'r': r})
            continue
            
        angle = 0
        dist = 0
        placed = False
        step = 0.5 
        
        while not placed:
            x = dist * np.cos(angle)
            y = dist * np.sin(angle)
            
            collision = False
            for b in bubbles:
                d = np.sqrt((x - b['x'])**2 + (y - b['y'])**2)
                if d < (r + b['r']) * 1.1:
                    collision = True
                    break
            
            if not collision:
                bubbles.append({'Language': lang, 'Games': count, 'x': x, 'y': y, 'r': r})
                placed = True
            else:
                angle += 0.5
                dist += step * 0.1
                
    df = pd.DataFrame(bubbles)
    
    # Calculate Font Size based on Radius (r)
    df['font_size'] = df['r'] * 0.6 

    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['x'],
        y=df['y'],
        mode='markers+text',
        marker=dict(
            size=df['Games'], 
            sizemode='area',
            sizeref=2.*max(df['Games'])/(100**2), 
            color=df['Games'],
            colorscale='Viridis',
            showscale=True,
            line=dict(width=1, color='white')
        ),
        text=df['Language'],
        textposition="middle center",
        textfont=dict(
            family="Arial",
            size=df['font_size'], 
            color="white"
        ),
        hovertemplate='<b>%{text}</b><br>Games: %{marker.color}<extra></extra>',
        showlegend=False
    ))
    
    fig.update_layout(
        title="Supported Languages",
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False, visible=False),
        plot_bgcolor='white',
        # --- UPDATED HEIGHT AND MARGINS ---
        height=400,  # Reduced from 600
        # Removed hardcoded width to let it fill the container, 
        # or you can use width=400 to keep it perfectly square.
        margin=dict(l=20, r=20, t=40, b=20) 
    )
    
    return fig

# ==========================================
# 3. UpSet Plot
# ==========================================
def create_upset_plot():
    # Data Prep
    combinations = [
        {'label': 'Win',       'count': 1200, 'sets': ['Win']},
        {'label': 'Win+Mac',   'count': 300,  'sets': ['Win', 'Mac']},
        {'label': 'Win+Lin',   'count': 400,  'sets': ['Win', 'Lin']},
        {'label': 'All',       'count': 150,  'sets': ['Win', 'Mac', 'Lin']},
        {'label': 'Mac Only',  'count': 50,   'sets': ['Mac']}, 
    ]
    
    df = pd.DataFrame(combinations)
    df = df.sort_values('count', ascending=False)
    sorted_labels = df['label'].tolist()
    categories = ['Win', 'Lin', 'Mac']
    
    # Calculate totals
    marginal_counts = {cat: 0 for cat in categories}
    for item in combinations:
        for s in item['sets']:
            if s in marginal_counts:
                marginal_counts[s] += item['count']
    
    # Layout: 2x2 Grid
    fig = make_subplots(
        rows=2, cols=2, 
        column_widths=[0.25, 0.75], 
        row_heights=[0.7, 0.3],   
        vertical_spacing=0.05,
        horizontal_spacing=0.05,
        shared_xaxes=True
    )

    # 1. Intersection Bars (Top Right)
    fig.add_trace(go.Bar(
        x=df['label'],
        y=df['count'],
        name="Intersection Size",
        marker_color='#3366CC'
    ), row=1, col=2)

    # 2. Marginal Bars (Bottom Left)
    fig.add_trace(go.Bar(
        x=[marginal_counts[c] for c in categories],
        y=categories,
        orientation='h',
        name="Total Size",
        marker_color='#109618'
    ), row=2, col=1)
    
    # 3. Matrix (Bottom Right)
    all_x, all_y = [], []
    for label in sorted_labels:
        for os_name in categories:
            all_x.append(label)
            all_y.append(os_name)

    fig.add_trace(go.Scatter(
        x=all_x, y=all_y,
        mode='markers',
        marker=dict(color='lightgray', size=10),
        hoverinfo='skip',
        showlegend=False
    ), row=2, col=2)

    for i, row in df.iterrows():
        active_sets = row['sets']
        active_y = [c for c in categories if c in active_sets]
        
        if len(active_y) > 1:
            fig.add_trace(go.Scatter(
                x=[row['label']] * len(active_y),
                y=active_y,
                mode='lines',
                line=dict(color='black', width=3),
                showlegend=False,
                hoverinfo='skip'
            ), row=2, col=2)
        
        fig.add_trace(go.Scatter(
            x=[row['label']] * len(active_y),
            y=active_y,
            mode='markers',
            marker=dict(color='black', size=14),
            showlegend=False,
            hoverinfo='skip'
        ), row=2, col=2)

    # Layout Alignment Logic
    fig.update_layout(
        title="OS Compatibility",
        margin=dict(l=40, r=40, t=60, b=40),
        height=350 # Slightly reduced height for the bottom chart too
    )

    fig.update_xaxes(categoryorder='array', categoryarray=sorted_labels, row=1, col=2)
    fig.update_xaxes(categoryorder='array', categoryarray=sorted_labels, row=2, col=2)
    fig.update_xaxes(showticklabels=False, row=1, col=2)

    fig.update_yaxes(categoryorder='array', categoryarray=categories, row=2, col=1)
    fig.update_yaxes(categoryorder='array', categoryarray=categories, row=2, col=2)
    fig.update_yaxes(showticklabels=False, row=2, col=2) 
    
    fig.update_yaxes(title="Games Count", row=1, col=2)
    fig.update_xaxes(title="Set Size", row=2, col=1)
    fig.update_xaxes(autorange="reversed", row=2, col=1)

    return fig

# ==========================================
# Main Execution
# ==========================================
if __name__ == "__main__":
    app = Dash(__name__)

    app.layout = html.Div([
        html.H1("Steam Graphs Prototype", style={'textAlign': 'center', 'marginBottom': '10px', 'marginTop': '10px'}),
        
        html.Div([
            html.Div([dcc.Graph(figure=create_stacked_area_chart())], style={'width': '49%', 'display': 'inline-block', 'verticalAlign': 'top'}),
            html.Div([dcc.Graph(figure=create_bubble_chart())], style={'width': '49%', 'display': 'inline-block', 'verticalAlign': 'top'}),
        ], style={'display': 'flex', 'justifyContent': 'space-between'}),

        html.Div([
            dcc.Graph(figure=create_upset_plot())
        ], style={'width': '90%', 'margin': '0 auto', 'marginTop': '10px'})
    ])

    print("Running server... Check http://127.0.0.1:8050/")
    app.run(debug=True)
