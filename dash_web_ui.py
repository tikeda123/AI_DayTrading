import dash
from dash import html, dcc
import plotly.graph_objects as go
import pandas as pd

from common.config_manager import ConfigManager
from common.trading_logger import TradingLogger
from TradingAnalysisKit.trading_analysis import TradingAnalysis

# Load data
config_path = '/Users/ikedatoshihiko/workspace/btctrading_wk/offline/btctrading_offline_ver2_0/aitrading_settings_ver2.json'

    # ConfigManager インスタンスを作成
config_manager = ConfigManager(config_path)

    # TradingLogger インスタンスを作成
trading_logger = TradingLogger(config_manager)

    # TradingAnalysis インスタンスを作成
trading_analysis = TradingAnalysis(config_manager, trading_logger)
df = trading_analysis.BolingerBand()

# Create a Dash application
app = dash.Dash(__name__)

# Create a candlestick chart
candlestick = go.Figure(data=[go.Candlestick(x=df['date'],
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'])])

# Define the layout of the app
app.layout = html.Div(children=[
    html.H1(children='Bitcoin Trading Candlestick Chart'),
    dcc.Graph(
        id='btc-candlestick',
        figure=candlestick
    )
])

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
