import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

# App Configurations
START = "2018-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.set_page_config(
    page_title="Stock Forecast App",
    page_icon="📈",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .block-container {
        padding-top: 2rem;
    }
    h1 {
        color: #002244;
        text-align: center;
    }
    .css-1aumxhk {
        padding: 2rem 5rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and Description
st.title('📈 Stock Forecast App')
st.markdown("""
    Welcome to the Stock Forecast App. Select a stock from the dropdown menu and choose the number of years to predict. 
    The app will provide historical data and forecast future stock prices using the Prophet model.
    """)

# Sidebar for User Inputs
st.sidebar.header('User Inputs')
# List of 50 companies with ticker symbols and full names
stocks = {
    'AAPL': 'Apple Inc.',
    'MSFT': 'Microsoft Corporation',
    'GOOGL': 'Alphabet Inc.',
    'AMZN': 'Amazon.com, Inc.',
    'TSLA': 'Tesla, Inc.',
    'META': 'Meta Platforms, Inc.',
    'NVDA': 'NVIDIA Corporation',
    'BRK-B': 'Berkshire Hathaway Inc.',
    'JPM': 'JPMorgan Chase & Co.',
    'JNJ': 'Johnson & Johnson',
    'V': 'Visa Inc.',
    'UNH': 'UnitedHealth Group Incorporated',
    'WMT': 'Walmart Inc.',
    'PG': 'Procter & Gamble Co.',
    'DIS': 'The Walt Disney Company',
    'MA': 'Mastercard Incorporated',
    'HD': 'The Home Depot, Inc.',
    'VZ': 'Verizon Communications Inc.',
    'ADBE': 'Adobe Inc.',
    'NFLX': 'Netflix, Inc.',
    'PYPL': 'PayPal Holdings, Inc.',
    'CRM': 'Salesforce.com, Inc.',
    'CSCO': 'Cisco Systems, Inc.',
    'IBM': 'International Business Machines Corporation',
    'INTC': 'Intel Corporation',
    'ORCL': 'Oracle Corporation',
    'QCOM': 'QUALCOMM Incorporated',
    'MCD': "McDonald's Corporation",
    'KO': 'The Coca-Cola Company',
    'PEP': 'PepsiCo, Inc.',
    'NKE': 'NIKE, Inc.',
    'SBUX': 'Starbucks Corporation',
    'CMCSA': 'Comcast Corporation',
    'GOOG': 'Alphabet Inc. (Class C)',
    'CVX': 'Chevron Corporation',
    'XOM': 'Exxon Mobil Corporation',
    'ABBV': 'AbbVie Inc.',
    'MRK': 'Merck & Co., Inc.',
    'ABB': 'ABB Ltd',
    'NVO': 'Novo Nordisk A/S',
    'TM': 'Toyota Motor Corporation',
    'TSM': 'Taiwan Semiconductor Manufacturing Company Limited',
    'LMT': 'Lockheed Martin Corporation',
    'BA': 'The Boeing Company',
    'GM': 'General Motors Company',
    'F': 'Ford Motor Company',
    'BABA': 'Alibaba Group Holding Limited',
    'JD': 'JD.com, Inc.',
    'T': 'AT&T Inc.',
    'AMGN': 'Amgen Inc.',
    'GILD': 'Gilead Sciences, Inc.'
}
selected_stock = st.sidebar.selectbox('Select Stock', options=list(stocks.keys()), format_func=lambda x: stocks[x])
n_years = st.sidebar.slider('Years of Prediction:', 1, 4)
period = n_years * 365

@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
data_load_state.text('Loading data... done!')

# Raw Data Section
st.subheader('Raw Data')
st.write(data.tail())

# Plot raw data
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="Stock Open"))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="Stock Close"))
    fig.layout.update(title_text='Time Series Data with Rangeslider', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

# Additional Stock Statistics
st.subheader('Stock Statistics')
st.write(f"**Company Name:** {stocks[selected_stock]}")
st.write(f"**Stock Ticker:** {selected_stock}")
st.write(f"**Market Cap:** {yf.Ticker(selected_stock).info['marketCap']:,}")
st.write(f"**52-Week High:** ${yf.Ticker(selected_stock).info['fiftyTwoWeekHigh']}")
st.write(f"**52-Week Low:** ${yf.Ticker(selected_stock).info['fiftyTwoWeekLow']}")

# Predict forecast with Prophet
df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# Forecast Data Section
st.subheader('Forecast Data')
st.write(forecast.tail())

# Forecast Plot
st.write(f'Forecast plot for {n_years} years')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

# Forecast Components
st.write("Forecast components")
fig2 = m.plot_components(forecast)
st.write(fig2)

# Daily Price Change
st.subheader('Daily Price Change')
data['Daily Change'] = data['Close'].diff()
st.write(data[['Date', 'Daily Change']].tail())

# Historical Data Chart
st.subheader('Historical Data Chart')
fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name='Close Price'))
fig3.update_layout(title_text='Historical Closing Prices', xaxis_title='Date', yaxis_title='Close Price')
st.plotly_chart(fig3)
