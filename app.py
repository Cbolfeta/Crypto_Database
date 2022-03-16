## Necessary dependencies
from dash import dash, Input, Output, callback, dash_table, html, dcc
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import eventstudy as es
from pandas_datareader import data as pdr
import yfinance as yfin
import statsmodels.api as sm
import getFamaFrenchFactors
import numpy as np
import plotly.graph_objs as go


# Load the Curated Event CryptoCurrencies
cripto = pd.read_csv('news3k_relevant_FinBERT.csv')
cripto = cripto.dropna()
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
ticker = "BTC-USD"
start = "2018-01-01"
end = "2022-02-01"

stock_data = pdr.get_data_yahoo(ticker, start, end)
stock_data2 = pdr.get_data_yahoo(ticker, start, end).reset_index()
ff3_daily=pd.read_csv("FF3_daily.csv")
es.Single.import_FamaFrench("{}_famafrench.csv".format(ticker))
es.Single.import_returns("{}_returns.csv".format(ticker))


event = es.Single.FamaFrench_3factor(
    security_ticker = ticker,
    event_date = np.datetime64('2020-12-28'),
    event_window = (-2,+4),
    estimation_size = 100,
    buffer_size = 30
)
index = np.array([-2,-1,0,1,2,3,4])
index = pd.Series(index)


CAR = event.CAR
CAR = pd.Series(CAR)

AR = event.AR
AR = pd.Series(AR)

var_CAR = event.var_CAR
var_CAR = pd.Series(var_CAR)

var_AR = event.var_AR
var_AR = pd.Series(var_AR)

plot_series = pd.concat([index,CAR,AR,var_CAR,var_AR], axis=1)
df = pd.DataFrame(plot_series).reset_index()
df.columns = ['index','Days', 'CAR','AR','var_CAR','var_AR']

fig = go.Figure(data=[go.Candlestick(x=stock_data2['Date'],
                                     open=stock_data2['Open'],
                                     high=stock_data2['High'],
                                     low=stock_data2['Low'],
                                     close=stock_data2['Close'])])

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
# Set up the app layout

dates = cripto.date.unique().tolist()

app.layout = html.Div([
    html.Div([
        html.H1("BTC Event Dashboard"),
        html.Img(src='assets/btc.png')
    ], className = 'banner'),


    html.Div([
        html.Div([
            html.P('Historic candle plot for BTC', className = 'fix_label', style={'color':'black','margin-top':'2px'}),
            dcc.Graph(id = 'timeseries', figure = fig),
        ])
    ]),
    html.Div([
        html.Div([
            html.P('Select a date', className = 'fix_label', style={'color':'black','margin-top':'2px'}),
            dcc.Dropdown(
                id = 'date-filter',
                options=[{'label':date, 'value':date} for date in dates], #'Dates' is the filter
                value =dates[0],
                clearable = False,
                searchable = False,
                className = 'dropdown', style={'fontSize': "24px",'textAlign': 'center'},
            ),
        ])
    ]),
    html.Div([
        html.Div(children=[
            html.Div(children=
            dcc.Graph(
                id = 'indicator',
                figure = {}),
                style={'width': '100%', 'display': 'inline-block'}
            ),
            html.Div(children =
            dash_table.DataTable(
                id='table-container',
                sort_action="native",
                hidden_columns=['FinBERT_neutral','Unnamed: 0','key','link','img','AR-2','AR-1','AR0','AR1','AR2','AR3','AR4','Relevant_pos','Relevant_neg'],
                columns=[{'id': c, 'name': c} for c in cripto.columns.values],
            ),
            ),
            html.Div(children=[
            dbc.Button(
                id='btn',
                children=[html.I(className="fa fa-download mr-1"), "Download"],
                color='info',
                className='mt-1')
            ]),
            dcc.Download(id='download-component'),
        ],className='m-4')
    ]),
    html.Div([
        html.Div([
            html.P('AR returns from -2 to +4 days', className = 'fix_label', style={'color':'black','margin-top':'2px'}),
            dcc.Graph(id = 'scatter',figure = {}),
        ])
    ]),
])
@app.callback(
    Output("download-component","data"),
    Input("btn","n_clicks"),
    prevent_initial_call=True)
def func(n_clicks):
    return dcc.send_file("news3k_relevant_FinBERT.csv")
@app.callback(
    Output('indicator', 'figure'),
    [Input('date-filter', 'value')])
def update_indicator(date):
    s_df = stock_data2[stock_data2.Date == date]
    gauge = go.Figure(go.Indicator(
        mode = "number+gauge+delta",
        gauge = {'shape': "bullet"},
        delta = {'reference': int(s_df["Open"])},
        value = int(s_df["Close"]),
        domain = {'x': [0.1, 1], 'y': [0.2, 0.9]},
        title = {'text': "Closing price"}))
    return gauge

@app.callback(
    Output('table-container', 'data'),
    [Input('date-filter', 'value')])
def display_table(date):
    dff = cripto[cripto.date == date]
    return dff.to_dict('records')

@app.callback(
    Output('scatter', 'figure'),
    [Input('date-filter','value')])
def update_chart(date):
    event = es.Single.FamaFrench_3factor(
        security_ticker = ticker,
        event_date = np.datetime64(date),
        event_window = (-2,+4),
        estimation_size = 100,
        buffer_size = 30
    )
    index = np.array([-2,-1,0,1,2,3,4])
    index = pd.Series(index)


    CAR = event.CAR
    CAR = pd.Series(CAR)

    AR = event.AR
    AR = pd.Series(AR)

    var_CAR = event.var_CAR
    var_CAR = pd.Series(var_CAR)

    var_AR = event.var_AR
    var_AR = pd.Series(var_AR)

    plot_series = pd.concat([index,CAR,AR,var_CAR,var_AR], axis=1)
    df = pd.DataFrame(plot_series).reset_index()
    df.columns = ['index','Days', 'CAR','AR','var_CAR','var_AR']

    fig = go.Figure([
        go.Scatter(
            name='AR measurement',
            x=df['Days'],
            y=df['AR'],
            mode='lines',
            line=dict(color='rgb(31, 119, 180)'),
        ),
        go.Scatter(
            name='Upper Bound',
            x=df['Days'],
            y=df['AR']+df['var_CAR'],
            mode='lines',
            marker=dict(color="#444"),
            line=dict(width=0),
            showlegend=False
        ),
        go.Scatter(
            name='Lower Bound',
            x=df['Days'],
            y=df['AR']-df['var_CAR'],
            marker=dict(color="#444"),
            line=dict(width=0),
            mode='lines',
            fillcolor='rgba(68, 68, 68, 0.3)',
            fill='tonexty',
            showlegend=False
        )])
    fig.update_layout(
        yaxis_title='AR',
        hovermode="x"
    )
    return fig

# Run server
## Importing Fama French 3 Factors and Stock Return
if __name__ == '__main__':
     app.run_server(debug=True)


