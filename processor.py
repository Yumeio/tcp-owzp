import xarray as xr
import numpy as np
import pandas as pd
import os
import warnings

warnings.filterwarnings("ignore")

def compute_wind_shear(u, v):
    """
    Tính độ lớn gió cắt (Deep-layer Wind Shear) giữa 200hPa và 850hPa.
    Công thức: sqrt((u200-u850)^2 + (v200-v850)^2)
    """
    u200 = u.sel(level=200)
    v200 = v.sel(level=200)
    u850 = u.sel(level=850)
    v850 = v.sel(level=850)
    return np.sqrt((u200 - u850)**2 + (v200 - v850)**2)

def haversine_dist(lat1, lon1, lat2, lon2):
    """Tính khoảng cách (km) giữa 2 điểm."""
    R = 6371.0 # Bán kính trái đất (km)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

def compute_owz(u, v):
    """
    Tính thông số Okubo-Weiss Zeta (OWZ) để nhận diện tâm bão.
    OWZ = Sn^2 + Ss^2 - Zeta^2
    """
    u_vals = u.values
    v_vals = v.values

    # Tính gradient (axis=-1 là lon, axis=-2 là lat)
    du_dx = np.gradient(u_vals, axis=-1)
    du_dy = np.gradient(u_vals, axis=-2)
    dv_dx = np.gradient(v_vals, axis=-1)
    dv_dy = np.gradient(v_vals, axis=-2)

    Sn = du_dx - dv_dy  # Biến dạng trượt
    Ss = dv_dx + du_dy  # Biến dạng chuẩn
    Zeta = dv_dx - du_dy # Độ xoáy
    
    W_vals = Sn**2 + Ss**2 - Zeta**2
    return xr.DataArray(W_vals, coords=u.coords, dims=u.dims, name='owz')

def compute_steering_flow(u, v):
    """
    Tính dòng dẫn (Steering Flow) trung bình lớp sâu.
    Trọng số áp suất: 850(75), 700(175), 500(250), 200(150).
    """
    levels = [850, 700, 500, 200]
    weights = np.array([75, 175, 250, 150])
    total_weight = weights.sum()

    u_weighted = sum(u.sel(level=l) * w for l, w in zip(levels, weights))
    v_weighted = sum(v.sel(level=l) * w for l, w in zip(levels, weights))
    
    return u_weighted / total_weight, v_weighted / total_weight

def calculate_bearing(lat1, lon1, lat2, lon2):
    """Tính góc hướng di chuyển (Radian) so với phương Bắc."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    initial_bearing = np.arctan2(x, y)
    # Chuẩn hóa về [0, 2pi]
    return (initial_bearing + 2 * np.pi) % (2 * np.pi)

class TCDataProcessor:
    def __init__(self, ibtracs_path, reanalysis_path, save_path):
        """
        Khởi tạo bộ xử lý.
        :param ibtracs_path: Đường dẫn đến file IBTrACS (.nc)
        :param reanalysis_path: Thư mục chứa các file NCEP (.nc)
        :param save_path: Đường dẫn lưu file CSV kết quả
        """
        self.ibtracs_path = ibtracs_path
        self.reanalysis_path = reanalysis_path
        self.save_path = save_path
        
        # Cache để lưu dữ liệu môi trường của năm hiện tại
        # Giúp tránh việc load lại file .nc liên tục
        self.cache_year = None
        self.cache_ds = None

    def _load_reanalysis(self, year):
        """
        Load dữ liệu NCEP cho 1 năm cụ thể, tính toán sẵn các Feature vật lý trên toàn lưới.
        """
        # Nếu năm yêu cầu đã có trong cache thì trả về ngay
        if self.cache_year == year and self.cache_ds is not None:
            return self.cache_ds

        print(f"   [System] Loading & Computing Physics Features for Year: {year}...")
        
        try:
            # Tạo đường dẫn file (Giả định tên file chuẩn của NCEP)
            # Bạn có thể sửa tên file ở đây nếu file bạn tải về có tên khác
            f_u = os.path.join(self.reanalysis_path, f"uwnd.{year}.nc")
            f_v = os.path.join(self.reanalysis_path, f"vwnd.{year}.nc")
            f_rh = os.path.join(self.reanalysis_path, f"rhum.{year}.nc")
            f_sh = os.path.join(self.reanalysis_path, f"shum.{year}.nc")

            # Mở dataset
            ds_u = xr.open_dataset(f_u)
            ds_v = xr.open_dataset(f_v)
            ds_rh = xr.open_dataset(f_rh)
            ds_sh = xr.open_dataset(f_sh)

            # --- XỬ LÝ TẦNG ÁP SUẤT (PRESSURE LEVELS) ---
            # Các tầng cần cho Gió (để tính Wind Shear)
            wind_levels = [200, 500, 700, 850, 925]
            
            # Các tầng cần cho Độ ẩm (NCEP thường thiếu 200hPa) -> Check thực tế
            req_humid = [500, 700, 850, 925]
            avail_rh = [l for l in req_humid if l in ds_rh.level]
            avail_sh = [l for l in req_humid if l in ds_sh.level]

            # Gộp vào 1 Dataset chung
            ds = xr.Dataset({
                'u': ds_u['uwnd'].sel(level=wind_levels),
                'v': ds_v['vwnd'].sel(level=wind_levels),
                'rh': ds_rh['rhum'].sel(level=avail_rh),
                'sph': ds_sh['shum'].sel(level=avail_sh)
            })

            # --- TÍNH TOÁN CÁC BIẾN PHÁI SINH (VECTORIZED) ---
            # 1. Wind Shear
            ds['ws'] = compute_wind_shear(ds['u'], ds['v'])
            
            # 2. Steering Flow (Dòng dẫn)
            ds['u_st'], ds['v_st'] = compute_steering_flow(ds['u'], ds['v'])
            
            # 3. Okubo-Weiss Zeta (OWZ) - Tính cho tầng 500 và 850
            ds['owz_500'] = compute_owz(ds['u'].sel(level=500), ds['v'].sel(level=500))
            ds['owz_850'] = compute_owz(ds['u'].sel(level=850), ds['v'].sel(level=850))

            # Lưu vào cache
            self.cache_year = year
            self.cache_ds = ds
            return ds

        except FileNotFoundError as e:
            print(f"   [Error] File not found for year {year}. Skipping. Details: {e}")
            return None
        except Exception as e:
            print(f"   [Error] Processing failed for year {year}: {e}")
            return None

    def _extract_environment(self, ds, lat, lon, time):
        """
        Trích xuất giá trị môi trường tại một điểm (lat, lon, time).
        Sử dụng phương pháp 'nearest' (gần nhất).
        """
        try:
            # Chọn điểm lưới gần nhất với tâm bão
            # Lưu ý: NCEP lưới 2.5 độ, nếu muốn chính xác hơn có thể dùng .interp (nội suy)
            # nhưng .sel(method='nearest') nhanh hơn nhiều.
            pt = ds.sel(time=time, lat=lat, lon=lon, method='nearest')
            
            # Trả về dictionary các features
            return {
                'ws_200': float(pt['ws']),
                'u_st': float(pt['u_st']),
                'v_st': float(pt['v_st']),
                'owz_500': float(pt['owz_500']),
                'owz_850': float(pt['owz_850']),
                # Với độ ẩm, phải gọi đúng tầng đã load được
                'rh_700': float(pt['rh'].sel(level=700)) if 700 in pt['rh'].level else np.nan,
                'rh_925': float(pt['rh'].sel(level=925)) if 925 in pt['rh'].level else np.nan,
                'sph_925': float(pt['sph'].sel(level=925)) if 925 in pt['sph'].level else np.nan,
            }
        except Exception as e:
            # Nếu thời gian bão không khớp với file NCEP (ví dụ lệch múi giờ quá xa)
            return None

    def process(self, start_year, end_year):
        """
        Hàm chính: Chạy toàn bộ quy trình từ lọc bão đến xuất CSV.
        """
        print(f"--- STARTING PROCESSING (Years: {start_year}-{end_year}) ---")
        
        # 1. Đọc dữ liệu bão IBTrACS
        if not os.path.exists(self.ibtracs_path):
            raise FileNotFoundError(f"IBTrACS file not found: {self.ibtracs_path}")
            
        ds_ib = xr.open_dataset(self.ibtracs_path)
        years_all = ds_ib['season'].values
        
        # Lấy index các cơn bão trong khoảng thời gian yêu cầu
        storm_indices = np.where((years_all >= start_year) & (years_all <= end_year))[0]
        print(f"Found {len(storm_indices)} potential storms in IBTrACS.")

        final_data = []
        processed_count = 0

        # 2. Duyệt qua từng cơn bão
        for idx in storm_indices:
            # --- EXTRACT RAW TRACK DATA ---
            sid = ds_ib['sid'][idx].item().decode() if isinstance(ds_ib['sid'][idx].item(), bytes) else str(ds_ib['sid'][idx].item())
            
            # Chuyển đổi đơn vị và xử lý toạ độ
            w_raw = ds_ib['cma_wind'][idx].values * 0.5144  # knots -> m/s
            p_raw = ds_ib['cma_pres'][idx].values
            lat_raw = ds_ib['cma_lat'][idx].values
            lon_raw = np.mod(ds_ib['cma_lon'][idx].values + 360, 360) # 0-360
            times_raw_bytes = ds_ib['iso_time'][idx].values
            # Chuyển đổi toàn bộ mảng bytes sang string
            times_raw = [t.decode('utf-8') if isinstance(t, bytes) else str(t) for t in times_raw_bytes]
            times_raw = np.array(times_raw) # Chuyển lại thành numpy array để slice dễ hơn

            # --- FILTER 1: INTENSITY (Cường độ > 17 m/s) ---
            valid_mask = (w_raw > 17) & (~np.isnan(w_raw))
            valid_indices = np.where(valid_mask)[0]
            
            if len(valid_indices) == 0:
                continue # Bỏ qua áp thấp nhiệt đới yếu

            # Cắt đoạn dữ liệu hợp lệ (từ lúc bắt đầu > 17m/s đến lúc suy yếu)
            first = valid_indices[0]
        #     # Tìm điểm suy yếu đầu tiên sau khi đã mạnh lên
            weaken_indices = np.where((w_raw[first:] < 17) | (np.isnan(w_raw[first:])))[0]
            last = first + weaken_indices[0] if len(weaken_indices) > 0 else len(w_raw)

        #     # Slice dữ liệu
            track_times = times_raw[first:last]
            track_lats = lat_raw[first:last]
            track_lons = lon_raw[first:last]
            track_ws = w_raw[first:last]
            track_ps = p_raw[first:last]
            
            if len(track_times) == 0 and len(track_times) < 2 : continue

            speeds = np.zeros(len(track_times))
            angles = np.zeros(len(track_times))

            # Tính cho từng bước (từ điểm thứ 2 trở đi)
            # Điểm đầu tiên (index 0) sẽ có speed=0, angle=0 (hoặc lấy bằng điểm số 1)
            for k in range(1, len(track_times)):
                dist = haversine_dist(track_lats[k-1], track_lons[k-1], track_lats[k], track_lons[k])
                # Giả định dữ liệu 6-hourly (chia 6 giờ). Nếu giờ lẻ cần tính dt chính xác.
                # Ở đây ta lấy gần đúng cho đơn giản, hoặc tính dt từ track_times
                dt = 6.0 
                speeds[k] = dist / dt # km/h
                angles[k] = calculate_bearing(track_lats[k-1], track_lons[k-1], track_lats[k], track_lons[k])
            
            # Gán giá trị điểm đầu bằng điểm thứ 2 để không bị số 0
            speeds[0] = speeds[1]
            angles[0] = angles[1]

            u24_past = np.zeros(len(track_times))
            for k in range(len(track_times)):
                if k >= 4:
                    u24_past[k] = track_ws[k] - track_ws[k-4]
                else:
                    # Với 4 điểm đầu tiên, chưa đủ 24h, ta gán bằng 0 hoặc 
                    # lấy hiệu số với điểm đầu tiên (tuỳ chọn). Ở đây để 0 cho an toàn.
                    u24_past[k] = 0.0

        #     # --- PREPARE REANALYSIS DATA ---
        #     # Lấy năm của cơn bão (dựa vào điểm đầu tiên)
            current_year = pd.to_datetime(track_times[0]).year
            
        #     # Load dữ liệu môi trường (sẽ dùng cache nếu đã load)
            env_ds = self._load_reanalysis(current_year)
            if env_ds is None: continue # Skip nếu thiếu file NCEP

        #     # --- LOOP THROUGH TRACK POINTS ---
            storm_points = []
            for j in range(len(track_times)):
                t = track_times[j]
                la = track_lats[j]
                lo = track_lons[j]

                # --- FILTER 2: SPATIAL (Lọc không gian: 0-30N, 100-150E) ---
                if not (0 <= la <= 30 and 100 <= lo <= 150):
                    continue

                # --- FILTER 3: 6-HOURLY (Chỉ lấy các mốc 00, 06, 12, 18) ---
                dt_time = pd.to_datetime(t)
                if dt_time.hour not in [0, 6, 12, 18]:
                    continue

                # Trích xuất đặc trưng môi trường
                env_features = self._extract_environment(env_ds, la, lo, t)

                if env_features:
                    # Tạo dòng dữ liệu hoàn chỉnh
                    row = {
                        'sid': sid,
                        'time': t,
                        'lat': la,
                        'lon': lo,
                        'umax': track_ws[j],
                        'mov_angle': (angles[j]),
                        'mov_speed': speeds[j],
                        'u24_past': u24_past[j],
                        'press': track_ps[j],
                        # Thêm các cột môi trường
                        **env_features
                    }
                    storm_points.append(row)

            # Nếu bão có điểm nào thỏa mãn thì thêm vào list tổng
            if len(storm_points) > 0:
                final_data.extend(storm_points)
                processed_count += 1
                # In tiến độ cứ mỗi 50 cơn bão
                if processed_count % 50 == 0:
                    print(f"   Processed {processed_count} storms...")

        # --- SAVE RESULT ---
        if final_data:
            df = pd.DataFrame(final_data)
            # Sắp xếp lại cột cho đẹp (tuỳ chọn)
            cols = ['sid', 'time', 'lat', 'lon', 'umax', 'press', 
                    'u24_past',
                    'mov_angle', 'mov_speed',
                    'u_st', 'v_st', 'ws_200', 'owz_500', 'owz_850', 
                    'rh_700', 'rh_925', 'sph_925']
            # Chỉ lấy các cột tồn tại
            cols = [c for c in cols if c in df.columns]
            df = df[cols]
            
            df.to_csv(self.save_path, index=False)
            print(f"\n[SUCCESS] Saved {len(df)} records to: {self.save_path}")
        else:
            print("\n[WARNING] No data found matching criteria!")
