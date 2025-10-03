import requests
import pandas as pd
import plotly.graph_objects as go
from datetime import date, timedelta, datetime, time 
import os
import numpy as np
from vnstock import Screener
from dash import dcc, html, dash_table, Dash
from dash.dependencies import Input, Output
from cachetools import cached, TTLCache

# ======================
# --- PHẦN 1: CÁC HÀM LẤY VÀ XỬ LÝ DỮ LIỆU ---
# ======================

DATA_DIR = "data"

@cached(cache=TTLCache(maxsize=1, ttl=86400))
def get_stock_map():

    try:
        df = Screener().stock(params={"exchangeName": "HOSE,HNX,UPCOM"}, limit=2000)
        df = df[['ticker', 'exchange']].rename(columns={'ticker': 'symbol'})
        return df
    
    except Exception as e:
        print(f"Error fetching stock map: {e}")
        return pd.DataFrame(columns=['symbol', 'exchange'])


def get_ticker_list():

    stock_map = get_stock_map()
    if stock_map.empty:
        return []
    
    stock_map = stock_map.sort_values('symbol').reset_index(drop=True)
    
    # Tạo danh sách options
    options = [
        {'label': f"{row['symbol']} ({row['exchange']})", 'value': row['symbol']}
        for index, row in stock_map.iterrows()
    ]
    return options


def get_tick_size(price, exch):

    exch = exch.upper()
    if exch == "HOSE":
        if price < 10000:
            return 10
        elif price < 50000:
            return 50
        else:
            return 100
    elif exch in ["HNX", "UPCOM"]:
        return 100
    else:
        return 10

def round_to_tick(price, tick):
    return np.round(price / tick) * tick


def apply_exchange_limits(df: pd.DataFrame, exchange: str) -> pd.DataFrame:

    if df.empty or 'Close' not in df.columns:
        return df

    for col in ['Open', 'High', 'Low', 'Close']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    exch = exchange.upper()
    
    # 1. Xác định biên độ dao động
    limits = {'HOSE': 0.07, 'HNX': 0.10, 'UPCOM': 0.15}
    limit = limits.get(exch, 0.07) 
    # 2. Tính giá tham chiếu 
    df['Reference'] = df['Close'].shift(1)
    
    # 3. Tính giá trần và giá sàn
    df['Ceiling'] = df['Reference'] * (1 + limit)
    df['Floor'] = df['Reference'] * (1 - limit)
    
    # 4. Tính Bước giá bằng Vectorization 
    df['Tick_Size'] = 100 

    if exch == 'HOSE':
        df.loc[(df['Reference'] >= 10000) & (df['Reference'] < 50000), 'Tick_Size'] = 50
        df.loc[df['Reference'] < 10000, 'Tick_Size'] = 10
    
    # 5. Làm tròn giá Trần/Sàn theo Bước giá (Vectorization)
    df['Ceiling'] = round_to_tick(df['Ceiling'], df['Tick_Size'])
    df['Floor'] = round_to_tick(df['Floor'], df['Tick_Size'])
    
    # 6. "Kẹp" các giá trị trong khoảng [sàn, trần]
    for col in ['Open', 'High', 'Low', 'Close']:
        if col in df.columns:
            # Kẹp giá trị trong khoảng [Floor, Ceiling]
            df[col] = df[col].clip(lower=df['Floor'], upper=df['Ceiling'])

            # Xử lý các giá trị còn lại là NaN hoặc = 0 sau khi kẹp (thay thế bằng Floor hoặc Reference)
            df.loc[(df[col].isna()) | (df[col] == 0), col] = df['Floor'].where(~df['Floor'].isna(), df['Reference'])

    for col in ['Open', 'High', 'Low', 'Close']:
        if col in df.columns:
            df[col] = df[col].round().astype(int)

    # 7. Đảm bảo logic High >= {Open, Close} >= Low sau khi kẹp
    df['High'] = df[['High', 'Open', 'Close']].max(axis=1)
    df['Low'] = df[['Low', 'Open', 'Close']].min(axis=1)
    
    # 8. Xóa các cột tạm và trả về kết quả
    df = df.drop(columns=['Reference', 'Ceiling', 'Floor', 'Tick_Size'])
    return df

def fetch_price_history_from_api(symbol: str, last_known_date: pd.Timestamp = None):
    print(f"Fetching API for {symbol}. Last known date: {last_known_date}")
    url = "https://cafef.vn/du-lieu/Ajax/PageNew/DataHistory/PriceHistory.ashx"
    all_records = []
    page = 1
    page_size = 5000
    while True:
        try:
            resp = requests.get(
                url, 
                params={"symbol": symbol, "page": page, "PageSize": page_size}, 
                timeout=15
            )
            data = resp.json()
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            break
        if not data or "Data" not in data or not data["Data"]["Data"]:
            break
        records = data["Data"]["Data"]
        if last_known_date:
            temp_df = pd.DataFrame(records)
            temp_df['DateParsed'] = pd.to_datetime(temp_df['Ngay'], format="%d/%m/%Y", errors="coerce")
            if temp_df['DateParsed'].min() > last_known_date:
                all_records.extend(records)
            else:
                new_records = temp_df[temp_df['DateParsed'] > last_known_date]
                all_records.extend(new_records.drop(columns=['DateParsed']).to_dict('records'))
                break
        else:
            all_records.extend(records)
        if len(records) < page_size:
            break
        page += 1
    return pd.DataFrame(all_records)

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: 
        return df
    
    rename_map = {
        "Ngay": "Date", "GiaDongCua": "Close", "GiaMoCua": "Open",
        "GiaCaoNhat": "High", "GiaThapNhat": "Low", "KhoiLuongKhopLenh": "Volume",
    }
    df = df.rename(columns=rename_map)
    df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y", errors="coerce")
    
    for col in ["Close", "Open", "High", "Low"]:
        if col in df.columns:
            df[col] = (pd.to_numeric(df[col], errors="coerce") * 1000).round().astype(int)

    if "Volume" in df.columns:
        df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce", downcast="integer")
    
    df = df.dropna(subset=["Date", "Open", "High", "Low", "Close"], how="all")
    df = df.drop_duplicates(subset=['Date'], keep='first').reset_index(drop=True)
    
    return df


def update_local_data(symbol: str):

    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    # 1. Tìm sàn giao dịch của mã
    stock_map = get_stock_map()
    stock_info = stock_map[stock_map['symbol'] == symbol]
    if stock_info.empty:
        print(f"Cannot find exchange for {symbol}. Skipping limit rules.")
        exchange = "HOSE" # Mặc định
    else:
        exchange = stock_info.iloc[0]['exchange']
    
    file_path = os.path.join(DATA_DIR, f"{symbol}.csv")
    df_local = pd.DataFrame()
    last_known_date = None 

    # 2. Đọc dữ liệu cũ (nếu có)
    if os.path.exists(file_path):
        df_local = pd.read_csv(file_path, parse_dates=['Date'])
        if not df_local.empty:
            last_known_date = df_local['Date'].max()

    # 3. Lấy dữ liệu mới từ API
    df_new_raw = fetch_price_history_from_api(symbol, last_known_date=last_known_date)
    
    # 4. Kết hợp, xử lý, và áp dụng luật trần/sàn
    if not df_new_raw.empty:
        df_new = preprocess_data(df_new_raw)
        # Kết hợp dữ liệu cũ và mới
        df_combined = pd.concat([df_local, df_new], ignore_index=True)
        df_combined = df_combined.drop_duplicates(subset=['Date'], keep='last').sort_values('Date').reset_index(drop=True)
        
        # Áp dụng luật trần/sàn trên TOÀN BỘ dữ liệu để đảm bảo tính chính xác
        df_corrected = apply_exchange_limits(df_combined, exchange)
        
        # Ghi đè file với dữ liệu đã được làm sạch
        df_corrected.to_csv(file_path, index=False)
        print(f"Updated and corrected data for {symbol} ({exchange}). Total rows: {len(df_corrected)}")

# ======================
# --- PHẦN 2: GIAO DIỆN DASH VÀ CALLBACK ---
# ======================

# Hàm plot_stock_chart 

def plot_stock_chart(df: pd.DataFrame, symbol: str):
    from plotly.subplots import make_subplots

    df = df.reset_index(drop=True)

    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA100'] = df['Close'].rolling(window=100).mean()


    # Tạo subplot với secondary_y
    fig = make_subplots(
        rows=1, cols=1, 
        specs=[[{"secondary_y": True}]], 
        subplot_titles=(f'{symbol} - Giá và Khối lượng',)
    )
    
    # Thêm biểu đồ nến (trục Y chính)
    fig.add_trace(
        go.Candlestick(
            x=df.index,  
            open=df["Open"], 
            high=df["High"], 
            low=df["Low"], 
            close=df["Close"], 
            name="Giá", 
            customdata=df['Date'].dt.strftime('%Y-%m-%d'),
            increasing_line_color="#189E54", 
            decreasing_line_color="#C7211E",
            hovertext=[
                f"Date: {d.strftime('%Y-%m-%d')}<br>Open: {o:,.0f}<br>High: {h:,.0f}<br>Low: {l:,.0f}<br>Close: {c:,.0f}"
                for d, o, h, l, c in zip(df['Date'], df['Open'], df['High'], df['Low'], df['Close'])
            ],
            hoverinfo="text"
        ),
        secondary_y=False  # Dùng trục Y chính
    )
    
    fig.add_trace(go.Scatter(
        x=df.index, 
        y=df['MA50'], 
        mode='lines', 
        name='MA50',
        line=dict(color='orange', width=1.5)
    ))

    fig.add_trace(go.Scatter(
        x=df.index, 
        y=df['MA100'], 
        mode='lines', 
        name='MA100',
        line=dict(color='blue', width=1.5)
    ))

    # Thêm volume (trục Y phụ)
    if 'Volume' in df.columns and not df['Volume'].isna().all():
        colors_volume = [
            '#C7211E' if row['Close'] < row['Open'] else '#189E54' 
            for _, row in df.iterrows()
        ]
        
        fig.add_trace(
            go.Bar(
                x=df.index,  
                y=df["Volume"], 
                name="Khối lượng", 
                marker=dict(color=colors_volume, opacity=0.5),  # Opacity thấp để không che giá
                hovertemplate="Volume: %{y:,.0f}<extra></extra>"
            ),
            secondary_y=True  # Dùng trục Y phụ
        )
    
    num_ticks = 15
    tick_spacing = max(1, len(df) // num_ticks)
    tick_indices = df.index[::tick_spacing]
    tick_dates = df['Date'].dt.strftime('%Y-%m-%d')[::tick_spacing]

     # Tối ưu range cho trục Y chính (giá)
    min_price = df["Low"].min()
    max_price = df["High"].max()
    price_range = max_price - min_price
    
    padding = price_range * 0.1  # 10% padding
    y_max = max_price + padding
    y_min = min_price - padding

    # Tính range cho trục Y phụ (Volume)
    max_volume = df['Volume'].max() if 'Volume' in df.columns else 0
    volume_y_max = max_volume * 5  # Scale lên 10 lần để volume không đè lên giá

    # Cập nhật trục X
    fig.update_xaxes(
        tickvals=tick_indices,
        ticktext=tick_dates,
        showgrid=True,
        gridwidth=1,
        gridcolor='#E5E5E5',
        showline=True,
        linewidth=2,
        linecolor='black',
        hoverformat=''
    )
    
    # Cập nhật trục Y chính (Giá)
    fig.update_yaxes(
        title_text="Giá (VNĐ)",
        range=[y_min, y_max],
        showgrid=True,
        gridwidth=1,
        gridcolor='#E5E5E5',
        showline=True,
        linewidth=2,
        linecolor='black',
        side='left',
        secondary_y=False
    )
    
    # Cập nhật trục Y phụ (Volume) với range mới
    fig.update_yaxes(
        title_text="Khối lượng",
        range=[0, volume_y_max],  # Set range từ 0 đến max_volume * 10
        fixedrange=True,
        showgrid=False,  # Tắt lưới của trục phụ để không bị rối
        showline=True,
        linewidth=2,
        linecolor='black',
        side='right',
        secondary_y=True
    )
    
    # Layout
    fig.update_layout(
        height=700, 
        hovermode='x unified', 
        xaxis_rangeslider_visible=False,
        plot_bgcolor='white',
        paper_bgcolor='white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    return fig

# Khởi tạo ứng dụng và layout 
app = Dash(__name__)
server = app.server
app.layout = html.Div([
    html.H1("📊 Lịch sử giá cổ phiếu", style={'textAlign': 'center'}),
    html.Div([
        dcc.Dropdown(id='ticker-dropdown', options=get_ticker_list(), placeholder="Chọn một mã", style={'width': '350px'}),
        dcc.RadioItems(id='time-filter-radio', options=[{'label': '1M', 'value': '1M'}, {'label': '3M', 'value': '3M'}, {'label': '6M', 'value': '6M'}, {'label': '1Y', 'value': '1Y'}, {'label': '3Y', 'value': '3Y'}, {'label': 'All', 'value': 'All'}], value='1Y', labelStyle={'display': 'inline-block', 'marginRight': '10px'}),
    ], style={'display': 'flex', 'justifyContent': 'center', 'gap': '20px', 'marginBottom': '20px'}),
    dcc.Loading(id="loading-spinner", type="circle", children=html.Div(id='output-container'))
], style={'padding': '20px'})


# Callback
@app.callback(
    Output('output-container', 'children'),
    [Input('ticker-dropdown', 'value'), Input('time-filter-radio', 'value')]
)
def update_graph(symbol, time_filter):
    if not symbol:
        return html.Div("Vui lòng chọn một mã cổ phiếu để bắt đầu.", style={'textAlign': 'center', 'marginTop': '50px'})
    
    update_local_data(symbol)
    
    file_path = os.path.join(DATA_DIR, f"{symbol}.csv")
    if not os.path.exists(file_path):
        return html.Div(f"⚠️ Không tìm thấy file dữ liệu cho mã {symbol}.", style={'textAlign': 'center', 'color': 'red'})

    df_full = pd.read_csv(file_path, parse_dates=['Date'])

    today_date = pd.to_datetime(datetime.now().date())
    cutoff_date = today_date - timedelta(days=1)

    df_full = df_full[df_full['Date'] <= cutoff_date].copy()

    if df_full.empty:
        return html.Div(f"⚠️ Dữ liệu cho mã {symbol} rỗng.", style={'textAlign': 'center', 'color': 'orange'})

    today_marker = df_full['Date'].max()
    reference_date = pd.to_datetime(today_marker).normalize()
    start_date = None
    if time_filter == "1M": 
        start_date = reference_date - pd.DateOffset(months=1)
    elif time_filter == "3M": 
        start_date = reference_date - pd.DateOffset(months=3)
    elif time_filter == "6M": 
        start_date = reference_date - pd.DateOffset(months=6)
    elif time_filter == "1Y": 
        start_date = reference_date - pd.DateOffset(years=1)
    elif time_filter == "3Y": 
        start_date = reference_date - pd.DateOffset(years=3)

    df_filtered = df_full[df_full["Date"] >= start_date].copy() if start_date else df_full.copy()

    if df_filtered.empty:
        return html.Div(
            f"⚠️ Không có dữ liệu cho {symbol} trong khoảng thời gian đã chọn.", 
            style={'textAlign': 'center'})

    fig = plot_stock_chart(df_filtered, symbol)
    df_display = df_filtered.sort_values("Date", ascending=False)
    
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col in df_display.columns:
            df_display[col] = pd.to_numeric(df_display[col]).round().astype(int)
   

    df_display['Date'] = df_display['Date'].dt.strftime('%Y-%m-%d')

    desired_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    
    df_display = df_display[desired_columns]

    
    return html.Div([
        dcc.Graph(figure=fig),
        html.Details([
            html.Summary('Xem bảng dữ liệu chi tiết'),
            dash_table.DataTable(
                columns=[{"name": i, "id": i} for i in df_display.columns], 
                data=df_display.to_dict('records'), 
                page_size=365, sort_action="native", 
                filter_action="native", 
                style_table={'overflowX': 'auto'})
        ])
    ])


if __name__ == "__main__":
    app.run(debug=True)