import streamlit as st
import pandas as pd
import numpy as np
import torch
import joblib
import os
import pickle
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import tempfile
from torchinfo import summary
from torchviz import make_dot 

from model import Transformer
from processor import haversine_dist

st.set_page_config(page_title="Typhoon AI System", layout="wide", page_icon="🌪️")

st.markdown("""
    <style>
    .block-container {padding-top: 1.5rem;}
    div[data-testid="stMetricValue"] {font-size: 1.3rem;}
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_model_system():
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = Transformer(
        input_dim=15, 
        output_dim=4, 
        embed_dim=256, 
        num_heads=8, 
        num_encoder_layers=4, 
        num_decoder_layers=4,
        dropout=0.0
    )
    
    ckpt_path = "./checkpoint/model.pth"
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    scaler = joblib.load("./dataset/scaler_y.pkl")
    with open("./dataset/test_dataset.pkl", 'rb') as f:
        data = pickle.load(f)
        if len(data) == 3: X, y, sids = data
        else: X, y = data; sids = ["UNKNOWN"] * len(X)
    
    return model, scaler, X, y, sids, DEVICE

@st.cache_data
def load_csv_data():
    path = "./data/TC_Filtered_Data.csv"
    if not os.path.exists(path): return None
    df = pd.read_csv(path)
    df['time'] = pd.to_datetime(df['time'])
    return df

@st.cache_data
def get_model_performance(_model, _scaler, X_test, y_test, device_str):
    device = torch.device(device_str)
    X_tensor = torch.tensor(X_test, dtype=torch.float32)
    
    batch_size = 128
    preds_list = []
    with torch.no_grad():
        for i in range(0, len(X_tensor), batch_size):
            batch_x = X_tensor[i:i+batch_size].to(device)
            dec_in = batch_x[:, -1, :4].unsqueeze(1)
            batch_pred = _model(batch_x, dec_in).squeeze(1)
            preds_list.append(batch_pred.cpu().numpy())
            
    preds = np.concatenate(preds_list, axis=0)
    
    # 2. Inverse Scale
    preds_real = _scaler.inverse_transform(preds)
    trues_real = _scaler.inverse_transform(y_test)
    
    targets = ['Latitude', 'Longitude', 'Wind Speed', 'Pressure']
    metrics_data = []
    
    for i, target in enumerate(targets):
        p_col = preds_real[:, i]
        t_col = trues_real[:, i]
        
        mae = mean_absolute_error(t_col, p_col)
        mse = mean_squared_error(t_col, p_col)
        rmse = np.sqrt(mse)
        r2 = r2_score(t_col, p_col)
        
        metrics_data.append([target, mae, mse, rmse, r2])
        
    df_perf = pd.DataFrame(metrics_data, columns=['Target Feature', 'MAE', 'MSE', 'RMSE', 'R² Score'])
    
    # 4. Tính Track Error (Km)
    dists = haversine_dist(trues_real[:,1], trues_real[:,0], preds_real[:,1], preds_real[:,0])
    track_stats = {
        "Mean": np.mean(dists),
        "Median": np.median(dists),
        "Max": np.max(dists),
        "Std": np.std(dists),
        "RMSE": np.sqrt(np.mean(dists**2)) 
    }
    
    return df_perf, track_stats, preds_real, trues_real

def show_forecast_page():
    try:
        model, scaler, X_test, y_test, sids_test, DEVICE = load_model_system()
        df_perf, track_stats, g_preds, g_trues = get_model_performance(model, scaler, X_test, y_test, str(DEVICE))
    except Exception as e:
        st.error(f"⚠️ System Error: {e}")
        return

    st.title("🔮 Typhoon Forecasting System")
    tab_single, tab_score = st.tabs(["🌪️ Single Storm Forecast", "🏆 Global Model Scoreboard"])

    # --- TAB 1: SINGLE STORM ---
    with tab_single:
        col_sel, col_kpi = st.columns([1, 3])
        with col_sel:
            unique_sids = sorted(list(set(sids_test)))
            sel_sid = st.selectbox("Select Test Storm:", unique_sids)

        # Lấy data bão
        indices = [i for i, s in enumerate(sids_test) if s == sel_sid]
        storm_trues = g_trues[indices]
        storm_preds = g_preds[indices]

        # Check RI (Đơn giản hóa cho UI)
        is_ri = False
        max_inc = 0
        if len(storm_preds) > 4:
            diffs = storm_preds[4:, 2] - storm_preds[:-4, 2]
            max_inc = np.max(diffs)
            if max_inc >= 14: is_ri = True

        with col_kpi:
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Max Wind", f"{storm_trues[:, 2].max():.1f} m/s")
            k2.metric("Min Press", f"{storm_trues[:, 3].min():.0f} hPa")
            current_err = np.mean(haversine_dist(storm_trues[:,1], storm_trues[:,0], storm_preds[:,1], storm_preds[:,0]))
            k3.metric("Avg Track Err", f"{current_err:.1f} km")
            
            if is_ri: k4.error(f"⚠️ RI PREDICTED (+{max_inc:.1f} m/s)")
            else: k4.success("✅ Intensity Stable")

        # Biểu đồ
        c1, c2 = st.columns([3, 2])
        with c1:
            fig_map = go.Figure()
            fig_map.add_trace(go.Scattermapbox(lat=storm_trues[:,0], lon=storm_trues[:,1], mode='markers+lines', name='Real', marker=dict(color='black', size=6)))
            fig_map.add_trace(go.Scattermapbox(lat=storm_preds[:,0], lon=storm_preds[:,1], mode='markers+lines', name='AI Pred', marker=dict(color='red', size=6)))
            fig_map.update_layout(mapbox_style="open-street-map", height=450, margin={"r":0,"t":0,"l":0,"b":0},
                                  mapbox=dict(center=dict(lat=np.mean(storm_trues[:,0]), lon=np.mean(storm_trues[:,1])), zoom=3))
            st.plotly_chart(fig_map, use_container_width=True)
        
        with c2:
            # Wind
            fig_w = go.Figure()
            fig_w.add_trace(go.Scatter(y=storm_trues[:,2], name='True Wind', line=dict(color='blue')))
            fig_w.add_trace(go.Scatter(y=storm_preds[:,2], name='Pred Wind', line=dict(color='red', dash='dash')))
            fig_w.update_layout(title="Wind Speed (m/s)", height=220, margin=dict(l=0,r=0,t=30,b=0))
            st.plotly_chart(fig_w, use_container_width=True)
            # Press
            fig_p = go.Figure()
            fig_p.add_trace(go.Scatter(y=storm_trues[:,3], name='True Press', line=dict(color='green')))
            fig_p.add_trace(go.Scatter(y=storm_preds[:,3], name='Pred Press', line=dict(color='orange', dash='dash')))
            fig_p.update_layout(title="Pressure (hPa)", height=220, margin=dict(l=0,r=0,t=30,b=0))
            st.plotly_chart(fig_p, use_container_width=True)

    # --- TAB 2: SCOREBOARD ---
    with tab_score:
        st.subheader("📊 Overall Model Performance")
        
        st.caption("Số liệu được tính toán trực tiếp từ toàn bộ tập dữ liệu kiểm tra (Test Set).")
       
        st.dataframe(
            df_perf.style.format({
                'MAE': '{:.3f}', 'MSE': '{:.3f}', 'RMSE': '{:.3f}', 'R² Score': '{:.4f}'
            }).background_gradient(cmap='Blues', subset=['MAE', 'RMSE'])  
                .background_gradient(cmap='Greens', subset=['R² Score']),  
            use_container_width=True,
            hide_index=True
        )
        
        col_note1, col_note2 = st.columns(2)
        col_note1.info("ℹ️ **MAE/RMSE:** Càng thấp càng tốt (0 là hoàn hảo).")
        col_note2.success("ℹ️ **R² Score:** Càng gần 1.0 càng tốt.")

        st.markdown("---")

        # --- Bảng 1: Track Error ---
        st.subheader("1. Track Error")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Mean Error", f"{track_stats['Mean']:.2f} km", delta_color="inverse")
        c2.metric("RMSE", f"{track_stats['RMSE']:.2f} km", delta_color="inverse")
        c3.metric("Median Error", f"{track_stats['Median']:.2f} km", delta_color="inverse")
        c4.metric("Max Error", f"{track_stats['Max']:.2f} km", delta_color="inverse")
        c5.metric("Std Dev", f"{track_stats['Std']:.2f} km")

        st.markdown("---")
        # --- Bảng 2: RI Skill (Số liệu tham khảo từ eval.py) ---
        st.subheader("2. Rapid Intensification (RI) Skill")
        r1, r2, r3 = st.columns(3)
        r1.metric("HSS Score", "0.7848", "Excellent")
        r2.metric("POD (Detection)", "66.7%", "2/3 Events")
        r3.metric("False Alarm", "0.0%", "Perfect")

def show_analysis_page():
    st.title("📊 Advanced Data Analytics & Scoring")
    
    df = load_csv_data()
    if df is None:
        st.error("CSV File not found.")
        return

    # --- KPI Summary ---
    with st.container():
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Data Points", f"{len(df):,}")
        k2.metric("Total Storms", f"{df['sid'].nunique()}")
        k3.metric("Avg Intensity", f"{df['umax'].mean():.1f} m/s")
        # Tính tỷ lệ RI trong data (số mẫu có u24_past >= 14)
        ri_count = len(df[df['u24_past'] >= 14])
        k4.metric("RI Events (Samples)", f"{ri_count} ({ri_count/len(df)*100:.1f}%)")
    
    st.markdown("---")

    # --- TABS ---
    tabs = st.tabs([
        "ℹ️ Features Info",
        "🔍 Deep Correlation", 
        "⭐ Feature Scores", 
        "🏆 Storm Rankings", 
        "🗺️ Density Map"
    ])
    
    with tabs[0]:
        st.subheader("📖 Features Dictionary")
        st.caption("Chi tiết về 15 đặc trưng đầu vào được sử dụng trong mô hình.")

        # Tạo dữ liệu cho bảng (15 dòng, 4 cột)
        features_data = [
            # --- TRACK GROUP ---
            ["lat", "Latitude", "Vĩ độ của tâm bão (Bắc/Nam)", "IBTrACS (Observed)"],
            ["lon", "Longitude", "Kinh độ của tâm bão (Đông/Tây)", "IBTrACS (Observed)"],
            ["umax", "Max Sustained Wind", "Tốc độ gió mạnh nhất gần tâm bão (m/s)", "IBTrACS (Observed)"],
            ["press", "Min Central Pressure", "Áp suất khí quyển thấp nhất tại tâm (hPa)", "IBTrACS (Observed)"],
            ["u24_past", "24h Intensity Change", "Sự thay đổi tốc độ gió so với 24h trước", r"$V_t - V_{t-24h}$"],
            
            # --- KINEMATICS GROUP ---
            ["mov_speed", "Translation Speed", "Tốc độ di chuyển tịnh tiến của bão (km/h)", r"$Distance(t, t-6h) / 6$"],
            ["mov_angle", "Heading Angle", "Góc hướng di chuyển so với phương Bắc", r"$\arctan(\Delta Lon, \Delta Lat)$"],
            
            # --- ENVIRONMENT GROUP (NCEP) ---
            ["u_st", "Zonal Steering Flow", "Dòng dẫn đường phương Đông-Tây (Trung bình lớp sâu)", r"$\frac{\sum U_p \cdot w_p}{\sum w_p}$ (850-200hPa)"],
            ["v_st", "Meridional Steering Flow", "Dòng dẫn đường phương Bắc-Nam (Trung bình lớp sâu)", r"$\frac{\sum V_p \cdot w_p}{\sum w_p}$ (850-200hPa)"],
            ["ws_200", "Deep-Layer Wind Shear", "Độ chênh lệch vector gió giữa tầng 200hPa và 850hPa", r"$\sqrt{(U_{200}-U_{850})^2 + (V_{200}-V_{850})^2}$"],
            ["owz_500", "Okubo-Weiss Zeta (500)", "Thông số biến dạng & độ xoáy tại tầng 500hPa (nhận diện lõi bão)", r"$S_n^2 + S_s^2 - \zeta^2$"],
            ["owz_850", "Okubo-Weiss Zeta (850)", "Thông số biến dạng & độ xoáy tại tầng 850hPa", r"$S_n^2 + S_s^2 - \zeta^2$"],
            ["rh_700", "Relative Humidity (700)", "Độ ẩm tương đối tại tầng trung (700hPa)", "NCEP Reanalysis"],
            ["rh_925", "Relative Humidity (925)", "Độ ẩm tương đối tại tầng thấp (925hPa)", "NCEP Reanalysis"],
            ["sph_925", "Specific Humidity (925)", "Độ ẩm riêng (lượng hơi nước thực tế) tại 925hPa", "NCEP Reanalysis"]
        ]

        # Tạo DataFrame
        df_info = pd.DataFrame(features_data, columns=["CSV Column", "Full Name", "Description", "Formula / Source"])
        
        # Hiển thị bảng
        st.table(df_info)
    
    with tabs[1]:
        st.subheader("📊 Interactive Correlation Analysis")
        st.caption("Khám phá mối quan hệ giữa các biến môi trường và cường độ bão.")

        c1, c2, c3 = st.columns([1, 1, 2])
        
        numeric_cols = [
            "lat",
            "lon",
            'umax', 
            'press', 
            'u24_past', 
            'mov_speed', 
            'mov_angle', 
            'ws_200', 
            'owz_500', 
            'owz_850', 
            'rh_700', 
            'rh_925', 
            'sph_925', 
            'u_st', 
            'v_st'
        ]
        
        with c1:
            x_axis = st.selectbox("Select X Axis:", numeric_cols, index=5) # Default ws_200
        with c2:
            y_axis = st.selectbox("Select Y Axis:", numeric_cols, index=2) # Default u24_past
        
        # Vẽ Scatter Plot với đường hồi quy (Trendline)
        with c3:
            st.write(f"**Analysis: {x_axis} vs {y_axis}**")
            
        # Lấy mẫu random 3000 điểm để vẽ cho nhanh (nếu data > 5000)
        plot_df = df.sample(3000) if len(df) > 5000 else df
        
        fig_scatter = px.scatter(
            plot_df, x=x_axis, y=y_axis, 
            color="umax", # Màu sắc thể hiện cường độ bão hiện tại
            trendline="ols", # Vẽ đường xu hướng (Regression Line)
            trendline_color_override="red",
            title=f"Correlation: {x_axis} vs {y_axis}",
            opacity=0.6,
            labels={x_axis: x_axis.upper(), y_axis: y_axis.upper()}
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Hiển thị Global Heatmap ở dưới
        with st.expander("View Full Correlation Matrix (Heatmap)"):
            corr_matrix = df[numeric_cols].corr()
            fig_heat = px.imshow(corr_matrix, text_auto=".3f", aspect="auto", color_continuous_scale="RdBu_r")
            st.plotly_chart(fig_heat, use_container_width=True)

    with tabs[2]:
        st.subheader("⭐ Feature Importance Scores")
        st.caption("Các yếu tố nào ảnh hưởng mạnh nhất đến sự thay đổi cường độ bão (u24_past)?")
        
        # Tính toán độ tương quan (Correlation Score) với Target là 'u24_past'
        target = 'u24_past'
        features = ['ws_200', 'owz_500', 'owz_850', 'rh_700', 'rh_925', 'sph_925', 'u_st', 'v_st', 'mov_speed']
        
        # Tính Correlation Pearson
        scores = df[features].corrwith(df[target]).sort_values(ascending=False)
        
        # Tạo DataFrame cho biểu đồ
        score_df = pd.DataFrame({'Feature': scores.index, 'Correlation Score': scores.values})
        
        # Vẽ Bar Chart
        fig_score = px.bar(
            score_df, x='Correlation Score', y='Feature', 
            orientation='h', 
            color='Correlation Score',
            color_continuous_scale='RdBu', 
            title=f"Correlation with Intensification ({target})",
            text_auto='.4f'
        )
        fig_score.add_vline(x=0, line_width=2, line_color="black")
        
        st.plotly_chart(fig_score, use_container_width=True)
        
        st.info("""
        **Giải thích điểm số (Scores):**
        * **Điểm Dương (+):** Yếu tố hỗ trợ bão mạnh lên (VD: `owz`, `rh` - độ ẩm cao, độ xoáy lớn).
        * **Điểm Âm (-):** Yếu tố kìm hãm bão (VD: `ws_200` - Gió cắt càng lớn, bão càng khó mạnh lên).
        """)

    with tabs[3]:
        st.subheader("🏆 Typhoon Hall of Fame")
        
        col_rank1, col_rank2 = st.columns(2)
        
        storm_stats = df.groupby('sid').agg({
            'umax': 'max',          # Gió mạnh nhất đạt được
            'u24_past': 'max',      # Tốc độ tăng cấp nhanh nhất
            'press': 'min',         # Áp suất thấp nhất
            'time': 'count'         # Thời gian tồn tại (số mẫu * 6h)
        }).reset_index()
        
        with col_rank1:
            st.markdown("#### 🌪️ Top 10 Strongest Storms")
            top_strong = storm_stats.sort_values('umax', ascending=False).head(10)
            st.dataframe(
                top_strong[['sid', 'umax', 'press']].style.background_gradient(subset=['umax'], cmap='Reds'),
                use_container_width=True
            )
            
        with col_rank2:
            st.markdown("#### 🚀 Top 10 Fastest Intensifying (RI)")
            top_fast = storm_stats.sort_values('u24_past', ascending=False).head(10)
            st.dataframe(
                top_fast[['sid', 'u24_past', 'umax']].style.background_gradient(subset=['u24_past'], cmap='Purples'),
                use_container_width=True
            )

    with tabs[4]:
        st.subheader("🌏 Global Activity Map")
        fig_dens = px.density_mapbox(
            df.sample(frac=0.2), # Sample 20%
            lat='lat', lon='lon', z='umax', radius=8,
            center=dict(lat=20, lon=130), zoom=2,
            mapbox_style="carto-positron",
            color_continuous_scale="turbo"
        )
        st.plotly_chart(fig_dens, use_container_width=True)

def show_model_artitechture():
    st.title("Model Architecture")
    
    # 1. Summary
    st.subheader("📊 Model Summary")
    
    try:
        model, _, _, _, _, DEVICE = load_model_system()
    except Exception as e:
        st.error(f"⚠️ System Error: {e}")
        return
    
    col1, col2, col3 , col4, col5, col6 = st.columns(6)
    with col1:
        total_params = sum(p.numel() for p in model.parameters())
        st.metric("Total Parameters", f"{total_params:,}")
    with col2:
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        st.metric("Trainable Parameters", f"{trainable_params:,}")
    with col3:
        model_size = total_params * 4 / (1024**2)  # Assuming float32
        st.metric("Model Size", f"{model_size:.3f} MB")
    with col4:
        st.metric("Loss", f"0.000973")
    with col5:
        st.metric("Val", f"0.000814")
    with col6:
        st.metric("Time", f"15p")
        
    st.image("./reports/loss_chart.png", caption="Training Loss", use_container_width=True)
    st.divider()

    
    # 2. Layer Details
    st.subheader("📋 Layer Details")
    
    layer_data = []
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Only leaf modules
            params = sum(p.numel() for p in module.parameters())
            layer_data.append({
                "Layer Name": name,
                "Type": module.__class__.__name__,
                "Parameters": f"{params:,}"
            })
    
    st.dataframe(layer_data, use_container_width=True)
    
    st.divider()
    
    # 4. Model Architecture Text
    st.subheader("📝 Model Structure")
    with st.expander("View Full Model Structure"):
        st.code(str(model), language="typescript")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["Forecast System", "Dataset Analytics", "Model Artitechture"])
if page == "Forecast System": 
    show_forecast_page()
elif page == "Dataset Analytics": 
    show_analysis_page()
else:
    show_model_artitechture()