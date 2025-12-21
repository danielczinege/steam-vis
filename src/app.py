from dash import Dash, html, dcc, Input, Output, State, ctx
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import calendar

# Import existing helper functions for Layout 2
from layout2_prep import load_dataset, l2_get_data, get_range_boxplot
# Import the bubble chart class
from bubble_chart import BubbleChartPlotly

app = Dash(__name__)
server = app.server

# ==========================================
# DATA LOADING & PROCESSING
# ==========================================

# Load Data Once - Keep as DataFrame
DF_MAIN = load_dataset()  # This is already a pandas DataFrame
DATA_RAW = DF_MAIN.to_numpy()  # Only convert to numpy if needed for Layout 2
TOTAL_GAMES = len(DF_MAIN) # Calculate total games for the title

# ==========================================
# LAYOUT 1: AGGREGATION & HELPER FUNCTIONS
# ==========================================

def get_aggregated_data(df, time_col):
    """
    Aggregates data into a matrix for Heatmap and a DataFrame for Stacked Area.
    Also returns real counts of unique games per time period.
    """
    time_counts = df[time_col].value_counts()

    # 2. Prepare Pairs for Heatmap
    data_pairs = []
    
    for t, g_val in zip(df[time_col], df['genres']):
        genres = list(g_val)
        for g in genres:
            data_pairs.append((t, g))
            
    if not data_pairs:
        return pd.DataFrame(), pd.DataFrame(), pd.Series(dtype=int)
    
    df_pairs = pd.DataFrame(data_pairs, columns=[time_col, 'genre'])
    
    # Matrix for Heatmap
    matrix_df = pd.crosstab(df_pairs['genre'], df_pairs[time_col])
    
    # Long Format for Stacked Area
    long_df = df_pairs.groupby([time_col, 'genre']).size().reset_index(name='count')
    
    return matrix_df, long_df, time_counts

# ==========================================
# LAYOUT 1: PLOT FUNCTIONS
# ==========================================

def create_complex_heatmap(matrix_df, time_counts, time_col):
    if matrix_df.empty:
        return go.Figure().update_layout(title="No Data")

    # Sort Matrix
    matrix_df = matrix_df.sort_index(ascending=False) 
    matrix_df = matrix_df.sort_index(axis=1)          
    
    genres = matrix_df.index.tolist()
    times = matrix_df.columns.tolist() # The x-axis values of the heatmap
    z_data = matrix_df.values
    
    total_per_time = time_counts.reindex(times, fill_value=0)
    
    total_per_genre = matrix_df.sum(axis=1)
    
    # Create Subplots
    fig = make_subplots(
        rows=2, cols=2,
        shared_xaxes=True,
        shared_yaxes=True,
        column_widths=[0.8, 0.2],
        row_heights=[0.2, 0.8],
        vertical_spacing=0.03,
        horizontal_spacing=0.03,
        print_grid=False
    )
    
    # 1. Top Histogram
    fig.add_trace(go.Bar(
        x=times, y=total_per_time,
        marker_color='#444', showlegend=False, 
        hoverinfo='x+y',
        name="Total Games"
    ), row=1, col=1)
    
    # 2. Main Heatmap
    fig.add_trace(go.Heatmap(
        z=z_data, x=times, y=genres,
        colorscale='Viridis',
        hovertemplate='<b>%{y}</b><br>%{x}: %{z} Games<extra></extra>',
    ), row=2, col=1)
    
    # 3. Right Histogram
    fig.add_trace(go.Bar(
        y=genres, x=total_per_genre,
        orientation='h', marker_color='#444', showlegend=False, hoverinfo='y+x'
    ), row=2, col=2)
    
    title_text = "Year" if time_col == "release_year" else "Month"
    fig.update_layout(
        title=f"Genre Density over {title_text}",
        margin=dict(l=60, r=20, t=60, b=40),
        plot_bgcolor='white',
        height=450
    )
    fig.update_xaxes(showticklabels=False, row=1, col=1)
    fig.update_yaxes(showticklabels=False, row=2, col=2)
    
    # ADD THESE LINES HERE - Show all x-axis labels vertically
    fig.update_xaxes(
        tickmode='linear',  # Show all ticks
        tickangle=-90,      # Rotate labels vertically
        row=2, col=1        # Apply to heatmap only
    )
    
    return fig

def add_heatmap_highlight(fig, selected_time, selected_genre):
    """
    Add highlighting shapes to indicate the selected cell/bar in the heatmap.
    """
    # Get the layout domains to position shapes correctly
    # The heatmap is in row=2, col=1
    
    if selected_time is not None:
        # Highlight vertical line for selected time (column)
        fig.add_vline(
            x=selected_time, 
            line_dash="dash", 
            line_color="red", 
            line_width=2,
            row=2, col=1
        )
        # Also highlight in top histogram
        fig.add_vline(
            x=selected_time,
            line_dash="dash",
            line_color="red",
            line_width=2,
            row=1, col=1
        )
    
    if selected_genre is not None:
        # Highlight horizontal line for selected genre (row)
        fig.add_hline(
            y=selected_genre,
            line_dash="dash",
            line_color="red",
            line_width=2,
            row=2, col=1
        )
        # Also highlight in right histogram
        fig.add_hline(
            y=selected_genre,
            line_dash="dash",
            line_color="red",
            line_width=2,
            row=2, col=2
        )
    
    return fig

def create_stacked_area(long_df, time_col):
    if long_df.empty:
        return go.Figure()
    
    # Filter Top 10 Genres
    top_genres = long_df.groupby('genre')['count'].sum().nlargest(10).index
    df_filt = long_df[long_df['genre'].isin(top_genres)].copy()
    
    # Pivot to fill zeros for missing years (fixes continuity issues)
    df_pivot = df_filt.pivot_table(index=time_col, columns='genre', values='count', fill_value=0).reset_index()
    df_final = df_pivot.melt(id_vars=[time_col], var_name='genre', value_name='count')
    df_final.sort_values(by=time_col, inplace=True)
    
    fig = px.area(
        df_final, 
        x=time_col, 
        y="count", 
        color="genre",
        title="Growth of Top 10 Genres"
    )
    fig.update_layout(height=450, margin=dict(l=20, r=20, t=60, b=40))
    return fig

def create_upset_plot(df, current_selection=""):
    # Pre-filter
    cols = ['windows', 'mac', 'linux']
    df_os = df[cols].fillna(0).astype(int)
    
    # Create signatures "101"
    df_os['sig'] = df_os.apply(lambda x: f"{x['windows']}{x['mac']}{x['linux']}", axis=1)
    counts = df_os['sig'].value_counts()
    
    sig_map = {
        '100': ('Win', ['Win']), '010': ('Mac', ['Mac']), '001': ('Lin', ['Lin']),
        '110': ('Win+Mac', ['Win', 'Mac']), '101': ('Win+Lin', ['Win', 'Lin']),
        '011': ('Mac+Lin', ['Mac', 'Lin']), '111': ('All', ['Win', 'Mac', 'Lin'])
    }
    
    plot_data = []
    for sig, count in counts.items():
        if sig in sig_map and count > 0:
            label, sets = sig_map[sig]
            plot_data.append({'label': label, 'count': count, 'sets': sets})
            
    if not plot_data:
        return go.Figure().update_layout(title=f"No OS Data{current_selection}")
        
    df_plot = pd.DataFrame(plot_data).sort_values('count', ascending=False)
    
    fig = make_subplots(
        rows=2, cols=2, column_widths=[0.2, 0.8], row_heights=[0.7, 0.3],
        shared_xaxes=True, horizontal_spacing=0.05, vertical_spacing=0.05
    )
    
    # Bar Chart
    fig.add_trace(go.Bar(x=df_plot['label'], y=df_plot['count'], marker_color='#3366CC', showlegend=False), row=1, col=2)
    
    # Matrix
    categories = ['Win', 'Mac', 'Lin']
    for idx, row_data in df_plot.iterrows():
        label, active = row_data['label'], row_data['sets']
        # Grey dots
        fig.add_trace(go.Scatter(
            x=[label]*3, y=categories, mode='markers', 
            marker=dict(color='lightgray', size=10), showlegend=False, hoverinfo='skip'
        ), row=2, col=2)
        # Active dots
        fig.add_trace(go.Scatter(
            x=[label]*len(active), y=active, mode='lines+markers',
            line=dict(color='black', width=3), marker=dict(color='black', size=14),
            showlegend=False, hoverinfo='skip'
        ), row=2, col=2)
        
    # Totals
    totals = [df_os['windows'].sum(), df_os['mac'].sum(), df_os['linux'].sum()]
    fig.add_trace(go.Bar(
        y=categories, x=totals, orientation='h', marker_color='#109618', showlegend=False
    ), row=2, col=1)
    
    fig.update_xaxes(autorange="reversed", row=2, col=1)
    # Modified title
    fig.update_layout(title=f"OS Compatibility{current_selection}", height=400)
    return fig

# Modified function signature to accept current_selection text
def create_bubble_chart(df, current_selection=""):
    lang_counts = {}
    for raw in df['supported_languages']:
        for l in raw:
            l = l.strip()
            if l:
                lang_counts[l] = lang_counts.get(l, 0) + 1

    if not lang_counts:
        return go.Figure().update_layout(title=f"No Data{current_selection}")

    # Top 40
    sorted_langs = sorted(lang_counts.items(), key=lambda x: x[1], reverse=True)[:40]
    
    # Prepare data for BubbleChartPlotly
    labels = [x[0] for x in sorted_langs]
    counts = [x[1] for x in sorted_langs]
    # Pass dummy colors to init (we will color by count in the actual plot)
    colors = ['#000000'] * len(labels) 
    
    # We set plot_diameter to 400 to match the approximate height of the view
    chart = BubbleChartPlotly(labels=labels, area=counts, colors=colors, plot_diameter=400)
    chart.collapse() # Run the physics simulation
    df_b = chart.to_dataframe()
    
    # Create the figure
    fig = go.Figure(go.Scatter(
        x=df_b['x'], y=df_b['y'], 
        mode='markers+text',
        text=df_b['label'],
        customdata=df_b['count'], # Pass original count here
        hovertemplate='<b>%{text}</b><br>Games: %{customdata}<extra></extra>', # Custom hover
        marker=dict(
            # size in scatter is diameter in px. 
            # radius in df_b is calculated based on plot_diameter.
            size=df_b['radius'] * 2, 
            sizemode='diameter', 
            color=df_b['size'], # Color based on the scaled area (proportional to count)
            colorscale='Teal',
            showscale=True
        ),
        textfont=dict(size=10, color='black')
    ))
    
    fig.update_layout(
        title=f"Top Supported Languages{current_selection}",
        xaxis=dict(visible=False), 
        yaxis=dict(visible=False, scaleanchor="x", scaleratio=1), # Keep aspect ratio circular
        height=400, 
        plot_bgcolor='white',
        margin=dict(l=10, r=10, t=40, b=10)
    )
    return fig

# ==========================================
# APP LAYOUT
# ==========================================

app.layout = html.Div(style={'backgroundColor': '#f0f0f0', 'padding': '20px', 'fontFamily': 'sans-serif'}, children=[
    html.H1(f"Steam Game Analytics (March 2025) - {TOTAL_GAMES:,} Games", style={'textAlign': 'center'}),
    
    # --- LAYOUT 1 SECTION ---
    html.Div([
        html.H3("Trends & Compatibility"),
        # Controls
        html.Div([
            html.Label("Time Aggregation: ", style={'fontWeight': 'bold', 'marginRight': '10px'}),
            dcc.RadioItems(
                id='time_selector',
                options=[{'label': 'Year', 'value': 'release_year'}, {'label': 'Month', 'value': 'release_month'}],
                value='release_year',
                inline=True,
                style={'display': 'inline-block'}
            ),
            html.Button("Reset All Filters", id='reset_btn', style={'marginLeft': '20px'})
        ], style={'padding': '15px', 'backgroundColor': 'white', 'marginBottom': '15px', 'borderRadius': '5px'}),
        
        # Row 1: Heatmap + Stacked Area
        html.Div([
            html.Div(dcc.Graph(id='heatmap_plot'), style={'width': '49%', 'display': 'inline-block'}),
            html.Div(dcc.Graph(id='stacked_plot'), style={'width': '49%', 'display': 'inline-block', 'float': 'right'})
        ], style={'marginBottom': '15px'}),
        
        # Row 2: Upset + Bubble
        html.Div([
            html.Div(dcc.Graph(id='upset_plot'), style={'width': '49%', 'display': 'inline-block'}),
            html.Div(dcc.Graph(id='bubble_plot'), style={'width': '49%', 'display': 'inline-block', 'float': 'right'})
        ])
    ]),

    html.Hr(style={'margin': '30px 0', 'borderTop': '2px solid #ccc'}),

    # Layout 2
        html.H2("Distribution Analysis"),
         # Layout 2 genres plot  
        html.Div( # html div element with one child node - a graph container
            [ dcc.Store(id="genres_figures_cached", data={}),
              dcc.Graph(id='l2_genres'), # id is used in callbacks to identify input and output elements
            ],
            style={'width': '80%', 'float':'left', 'justify-content': 'space-between'}
        ),
    
        # Options for L2 plots
        html.Div(children=[ #html div element with 2 child nodes - text and radiobutton container (with radio buttons as children)
            'Show:', 
            dcc.RadioItems([
                {'label': html.Div(['Distribution'],
                 style={'margin-bottom': '1%', 'display':'inline'}), 'value': 'dist'},

                {'label': html.Div(['Average'],
                 style={'margin-bottom': '1%', 'display':'inline'}), 'value': 'avg'}], # options
                value='dist', # default
                id='l2_show_choice'),
            
            "Quantity:",
            dcc.RadioItems([
                {'label': html.Div(['% of positive reviews'],
                 style={'margin-bottom': '1%', 'display':'inline'}), 'value': 'reviews'},

                {'label': html.Div(['Count'],
                 style={'margin-bottom': '1%', 'display':'inline'}), 'value': 'count'},

                {'label': html.Div(['Price'],
                 style={'margin-bottom': '1%', 'display':'inline'}), 'value': 'price'}
                ],
                value='price', 
                id='l2_quantity_choice'
            ),
            'Filter free games:', 
            dcc.RadioItems([
                {'label': html.Div(['Yes'],
                 style={'margin-bottom': '1%', 'display':'inline'}), 'value': True},

                {'label': html.Div(['No'],
                 style={'margin-bottom': '1%', 'display':'inline'}), 'value': False}], # options
                value=False, # default
                id='l2_free_games_choice')
           
            ], #inline layout of radiobuttons
            style={'width': '15%', 'display': 'inline-block', 'float':'left', 'margin': '2%'} #css styling  
            
        ),
    
        # Prices plot   
        html.Div(
            [dcc.Store(id="prices_figures_cached", data={}),
             dcc.Graph(id='l2_price')],
            style={'width': '45%', 'margin-top':'2%', 'margin-right':'2%', 'display': 'inline-block'}),

        # Number of dlcs plot
        html.Div(
            [dcc.Store(id="dlcs_figures_cached", data={}), 
             dcc.Graph(id="l2_dlcs")],
            style={'width': '45%', 'margin-top':'2%', 'display': 'inline-block'}
        )
    ],
 )


# ==========================================
# CALLBACKS - LAYOUT 1
# ==========================================

def convert_month_number_to_name(month_num):
    """Convert month number to month name."""
    if 1 <= month_num <= 12:
        return calendar.month_name[month_num]
    return str(month_num)

@app.callback(
    Output('heatmap_plot', 'figure'),
    Output('stacked_plot', 'figure'),
    Output('upset_plot', 'figure'),
    Output('bubble_plot', 'figure'),
    Input('heatmap_plot', 'clickData'),
    Input('reset_btn', 'n_clicks'),
    Input('time_selector', 'value'),
    State('time_selector', 'value')
)
def update_all_layout1_charts(clickData, reset, time_changed, time_col):
    ctx_id = ctx.triggered_id
    df_filt = DF_MAIN.copy()
    
    # Store selection info for highlighting
    selected_time = None
    selected_genre = None
    
    # Variable to hold the text description of the selection
    selection_title_suffix = ""

    if ctx_id == 'heatmap_plot' and clickData:
        pt = clickData['points'][0]
        curve_num = pt.get('curveNumber', 1)
        x_val = pt.get('x')
        y_val = pt.get('y')
        
        # Convert x_val to proper type
        if x_val and str(x_val).replace('.','',1).isdigit():
            x_val = float(x_val)
            if time_col in ['release_year', 'release_month']:
                x_val = int(x_val)

        title_text = "year" if time_col == "release_year" else "month"

        # Trace 0=TopHist (Time), 1=Heatmap (Time+Genre), 2=RightHist (Genre)
        if curve_num == 0: 
            df_filt = df_filt[df_filt[time_col] == x_val]
            selected_time = x_val
            selection_title_suffix = f" ({title_text}: {x_val if time_col == 'release_year' else convert_month_number_to_name(x_val)})"
        elif curve_num == 1:
            df_filt = df_filt[df_filt[time_col] == x_val]
            df_filt = df_filt[df_filt['genres'].apply(
                lambda x: str(y_val) in list(x)
            )]
            selected_time = x_val
            selected_genre = y_val
            selection_title_suffix = f" (genre: {y_val}, {title_text}: {x_val if time_col == 'release_year' else convert_month_number_to_name(x_val)})"
        elif curve_num == 2:
            df_filt = df_filt[df_filt['genres'].apply(
                lambda x: str(y_val) in list(x)
            )]
            selected_genre = y_val
            selection_title_suffix = f" (genre: {y_val})"
    
    # Generate all charts
    matrix, long_df, time_counts = get_aggregated_data(DF_MAIN, time_col)
    fig_heatmap = create_complex_heatmap(matrix, time_counts, time_col)
    fig_stacked = create_stacked_area(long_df, time_col)
    
    # Add highlighting if there's a selection
    if ctx_id == 'heatmap_plot' and (selected_time is not None or selected_genre is not None):
        fig_heatmap = add_heatmap_highlight(fig_heatmap, selected_time, selected_genre)

    # Pass the selection title suffix to the plot creators
    fig_upset = create_upset_plot(df_filt, selection_title_suffix)
    fig_bubble = create_bubble_chart(df_filt, selection_title_suffix)
    
    return fig_heatmap, fig_stacked, fig_upset, fig_bubble

# ==========================================
# CALLBACKS - LAYOUT 2
# ==========================================

@app.callback(
    # Figures
    Output("l2_genres", "figure"),
    Output("l2_price", "figure"),
    Output("l2_dlcs", "figure"),
    # Output caches
    Output("genres_figures_cached", "data"),
    Output("prices_figures_cached", "data"),
    Output("dlcs_figures_cached", "data"),
    # Output radio button value
    Output("l2_show_choice", "value"),
    # Input options
    Input("l2_show_choice", "value"),
    Input("l2_quantity_choice", "value"),
    Input("l2_free_games_choice", "value"),
    # Input caches
    State("genres_figures_cached", "data"),
    State("prices_figures_cached", "data"),
    State("dlcs_figures_cached", "data")
)
def update_by_genres(show: str, quantity: str, free: bool, 
                     genres_cache, prices_cache, dlcs_cache):
                         
    key: str = f"{show}/{quantity}/{free}"
    # We cannot show the "distribution" of number of games so we always switch to "average"
    radio_value = show if not quantity == "count" else "avg"
    # Only need to check one cache - if cached for genres, must be cached for other figures as well
    if key in genres_cache:
        return genres_cache[key], prices_cache[key], dlcs_cache[key], genres_cache, prices_cache, dlcs_cache, radio_value

    dist = True if (show == "dist" and not quantity == "count") else False
    qtty = 0 if quantity == "reviews" else (1 if quantity == "count" else 2)
    
    (genres_data, prices_data, dlcs_data) = l2_get_data(DATA_RAW, dist, qtty, free)
    
    y_val: str = genres_data.columns.values[0] if not dist else genres_data.columns.values[1]
    
    y_axis_label = y_val
    if quantity == 'price':
        y_axis_label = f"{y_val} ($)"

    # Barchart with average values    
    if not dist:
        title = f"Average {y_val.title()} Per" if y_val != "Count" else "Number of Games Per"
        genres_fig = px.bar(genres_data, labels={'index': 'Genre', 'value': y_val}, title=f"{title} Genre")
        prices_fig = px.bar(prices_data, labels={'index': 'Prices', 'value': y_val}, title=f"{title} Price Range")
        dlcs_fig = px.bar(dlcs_data, labels={'index': 'Number of Dlcs', 'value': y_val}, title=f"{title} Number Of Dlcs")
    # Box plots
    else:
        title = f"Distribution Of {y_val.title()} Per"
        genres_fig = px.box(genres_data, y=y_val, x="Genre", color="Genre", title=f"{title} Genre")
        prices_fig = px.box(prices_data, y=y_val, x="Price ranges", color="Price ranges", title=f"{title} Price Range")
        dlcs_fig =  px.box(dlcs_data, y=y_val, x="Number of Dlcs", color="Number of Dlcs", title=f"{title} Number Of Dlcs")
        
        # Do not plot outliers
        genres_fig.update_traces(boxpoints='outliers', marker=dict(opacity=0))
        lower, upper = get_range_boxplot(genres_data, "Genre", y_val)
        genres_fig.update_yaxes(range=[lower, upper])
        
        if y_val != "Price":
            prices_fig.update_traces(boxpoints='outliers', marker=dict(opacity=0))
            lower, upper = get_range_boxplot(prices_data, "Price ranges", y_val)
            prices_fig.update_yaxes(range=[lower, upper])
        
        dlcs_fig.update_traces(boxpoints='outliers', marker=dict(opacity=0))
        lower, upper = get_range_boxplot(dlcs_data, "Number of Dlcs", y_val)
        dlcs_fig.update_yaxes(range=[lower, upper])

    genres_fig.update_yaxes(title_text=y_axis_label)
    prices_fig.update_yaxes(title_text=y_axis_label)
    dlcs_fig.update_yaxes(title_text=y_axis_label)

    prices_fig.update_xaxes(title_text="Price Ranges ($)")

    genres_fig.layout.update(showlegend=False)
    prices_fig.layout.update(showlegend=False)
    dlcs_fig.layout.update(showlegend=False)
    
    genres_cache[key] = genres_fig.to_dict()
    prices_cache[key] = prices_fig.to_dict()
    dlcs_cache[key] = dlcs_fig.to_dict()
        
    return genres_fig, prices_fig, dlcs_fig, genres_cache, prices_cache, dlcs_cache, radio_value

#********RUNNING THE APP*************************************************
if __name__ == '__main__':
    app.run(jupyter_mode="external", debug=True)
