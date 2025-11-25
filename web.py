import streamlit as st
import pandas as pd
import numpy as np
import torch
import joblib
import os
import pickle
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

from model import Transformer

# =============================================================================
st.set_page_config(page_title="Typhoon AI System", layout="wide", page_icon="🌪️")

st.markdown("""
    <style>
    .block-container {padding-top: 1rem;}
    div[data-testid="stMetricValue"] {font-size: 1.2rem;}
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_model_system():
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = Transformer(input_dim=15, output_dim=4, embed_dim=128, 
                        num_heads=8, num_encoder_layers=4, num_decoder_layers=4, dropout=0.0)
    
    ckpt_path = "./checkpoint/model.pth"
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    # Scaler & Test Data
    scaler = joblib.load("./dataset/scaler_y.pkl")
    with open("./dataset/test_dataset.pkl", 'rb') as f:
        X, y, sids = pickle.load(f)
    
    return model, scaler, X, y, sids, DEVICE

@st.cache_data
def load_csv_data():
    # Đường dẫn file CSV tạo ra từ bước processor.py
    path = "./data/TC_Filtered_Data.csv" 
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df['time'] = pd.to_datetime(df['time'])
    df['year'] = df['time'].dt.year
    return df

def show_forecast_page():
    try:
        model, scaler, X_test, y_test, sids_test, DEVICE = load_model_system()
    except Exception as e:
        st.error(f"System not ready. Missing files? Error: {e}")
        return

    # Helper Functions
    def get_storm_data(sid):
        idx = [i for i, s in enumerate(sids_test) if s == sid]
        x_stm = torch.tensor(X_test[idx], dtype=torch.float32).to(DEVICE)
        y_true = y_test[idx]
        with torch.no_grad():
            dec_in = x_stm[:, -1, :4].unsqueeze(1)
            preds = model(x_stm, dec_in).squeeze(1).cpu().numpy()
        return scaler.inverse_transform(preds), scaler.inverse_transform(y_true)

    def check_ri(wind):
        if len(wind) < 5: return False, 0.0
        diffs = wind[4:] - wind[:-4]
        return np.any(diffs >= 14.0), np.max(diffs) if len(diffs) > 0 else 0

    # UI Forecast
    st.title("🔮 AI Forecast System")
    
    col_sel, col_info = st.columns([1, 3])
    with col_sel:
        unique_sids = sorted(list(set(sids_test)))
        sel_sid = st.selectbox("Select Typhoon (Test Set):", unique_sids)
    
    preds, trues = get_storm_data(sel_sid)
    is_ri, max_inc = check_ri(preds[:, 2])

    # Top Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Max Wind", f"{trues[:, 2].max():.1f} m/s")
    m2.metric("Min Press", f"{trues[:, 3].min():.0f} hPa")
    err = np.mean(np.sqrt((preds[:,0]-trues[:,0])**2 + (preds[:,1]-trues[:,1])**2)) * 111
    m3.metric("Track Error", f"{err:.1f} km")
    if is_ri:
        m4.error(f"⚠️ RI PREDICTED (+{max_inc:.1f} m/s)")
    else:
        m4.success("✅ Steady State")

    # Map & Charts
    tab1, tab2 = st.tabs(["🗺️ Trajectory", "📈 Intensity"])
    
    with tab1:
        fig = go.Figure()
        fig.add_trace(go.Scattermapbox(lat=trues[:,0], lon=trues[:,1], mode='markers+lines', name='Real', marker=dict(color='black')))
        fig.add_trace(go.Scattermapbox(lat=preds[:,0], lon=preds[:,1], mode='markers+lines', name='AI', marker=dict(color='red')))
        fig.update_layout(mapbox_style="open-street-map", height=500, margin={"r":0,"t":0,"l":0,"b":0},
                          mapbox=dict(center=dict(lat=np.mean(trues[:,0]), lon=np.mean(trues[:,1])), zoom=3))
        st.plotly_chart(fig, use_container_width=True)
        
    with tab2:
        c1, c2 = st.columns(2)
        fig_w = go.Figure()
        fig_w.add_trace(go.Scatter(y=trues[:,2], name='Real Wind', line=dict(color='blue')))
        fig_w.add_trace(go.Scatter(y=preds[:,2], name='Pred Wind', line=dict(color='red', dash='dash')))
        fig_w.update_layout(title="Wind Speed (m/s)", height=300, margin=dict(l=0, r=0, t=30, b=0))
        c1.plotly_chart(fig_w, use_container_width=True)
        
        fig_p = go.Figure()
        fig_p.add_trace(go.Scatter(y=trues[:,3], name='Real Press', line=dict(color='green')))
        fig_p.add_trace(go.Scatter(y=preds[:,3], name='Pred Press', line=dict(color='orange', dash='dash')))
        fig_p.update_layout(title="Pressure (hPa)", height=300, margin=dict(l=0, r=0, t=30, b=0))
        c2.plotly_chart(fig_p, use_container_width=True)

def show_analysis_page():
    st.title("📊 Dataset Analytics Dashboard")
    
    df = load_csv_data()
    if df is None:
        st.error("CSV File not found. Please check path in code.")
        return

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Data Points", f"{len(df):,}")
    k2.metric("Total Storms", f"{df['sid'].nunique()}")
    k3.metric("Year Range", f"{df['year'].min()} - {df['year'].max()}")
    k4.metric("Avg Wind Speed", f"{df['umax'].mean():.1f} m/s")
    
    st.markdown("---")

    # --- TABS ---
    tabs = st.tabs(["Start", "Global Heatmap", "Correlations", "Distributions", "Trend over Time"])

    # Tab 0: Raw Data
    with tabs[0]:
        st.write("### Raw Data Preview")
        st.dataframe(df.head(100))
        st.write("### Statistical Description")
        st.dataframe(df.describe())

    # Tab 1: Global Heatmap (Vị trí bão)
    with tabs[1]:
        st.write("### 🌏 Typhoon Density Map")
        # Dùng Density Mapbox
        fig_dens = px.density_mapbox(
            df.sample(frac=0.1), # Lấy mẫu 10% để vẽ cho nhanh nếu data lớn
            lat='lat', lon='lon', z='umax', radius=10,
            center=dict(lat=20, lon=130), zoom=2,
            mapbox_style="carto-positron",
            title="Typhoon Intensity Heatmap (Sampled 10%)"
        )
        st.plotly_chart(fig_dens, use_container_width=True)

    # Tab 2: Correlations (Quan hệ Vật lý)
    with tabs[2]:
        st.write("### 🔗 Feature Correlations")
        st.info("Phân tích mối tương quan giữa các biến môi trường và cường độ bão.")
        
        # Chọn các cột số quan trọng
        corr_cols = ['umax', 'press', 'u24_past', 'mov_speed', 
                     'ws_200', 'owz_850', 'rh_700', 'u_st']
        
        corr_matrix = df[corr_cols].corr()
        
        # Vẽ Heatmap bằng Seaborn
        fig_corr, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        st.pyplot(fig_corr)
        
        st.write("""
        **Gợi ý phân tích:**
        - **umax vs press**: Tương quan âm mạnh (Gió tăng -> Áp suất giảm).
        - **umax vs ws_200**: Thường là tương quan âm (Gió cắt cao -> Bão yếu).
        """)

    # Tab 3: Distributions (Phân bố)
    with tabs[3]:
        st.write("### 📊 Distributions")
        d_col1, d_col2 = st.columns(2)
        
        with d_col1:
            st.write("**Wind Speed Distribution**")
            fig_hist = px.histogram(df, x="umax", nbins=50, title="Wind Speed (m/s)", color_discrete_sequence=['blue'])
            st.plotly_chart(fig_hist, use_container_width=True)
            
        with d_col2:
            st.write("**Wind Shear Distribution**")
            fig_hist2 = px.histogram(df, x="ws_200", nbins=50, title="Wind Shear (200-850hPa)", color_discrete_sequence=['orange'])
            st.plotly_chart(fig_hist2, use_container_width=True)

    # Tab 4: Trend over Time (Xu hướng năm)
    with tabs[4]:
        st.write("### 📅 Temporal Trends")
        
        # Group by Year
        year_stats = df.groupby('year').agg({
            'sid': 'nunique',       # Số lượng bão
            'umax': 'mean'          # Cường độ trung bình
        }).reset_index()
        
        fig_trend = px.bar(year_stats, x='year', y='sid', 
                           title="Number of Storms per Year",
                           labels={'sid': 'Storm Count'}, color='sid')
        # Thêm đường cường độ trung bình
        fig_trend.add_scatter(x=year_stats['year'], y=year_stats['umax'], mode='lines', 
                              name='Avg Intensity (m/s)', yaxis='y2')
        
        # Dual axis setup
        fig_trend.update_layout(
            yaxis2=dict(title="Avg Wind (m/s)", overlaying='y', side='right')
        )
        
        st.plotly_chart(fig_trend, use_container_width=True)

# =============================================================================
# MAIN NAVIGATION
# =============================================================================
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Forecast System", "Dataset Analytics"])

if page == "Forecast System":
    show_forecast_page()
else:
    show_analysis_page()