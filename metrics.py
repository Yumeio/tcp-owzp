import numpy as np
import xarray as xr
import pandas as pd

class RIMetrics:
    def __init__(self, ibtracs_path):
        self.ibtracs_path = ibtracs_path
        self.ty_ids, self.ty_names = self._load_ibtracs_names()

    def _load_ibtracs_names(self):
        """Load tên bão từ file NetCDF"""
        try:
            ds = xr.open_dataset(self.ibtracs_path)
            ids = [id.decode('utf-8') for id in ds['sid'].values]
            names = [name.decode('utf-8') for name in ds['name'].values]
            return ids, names
        except Exception as e:
            print(f"[Warning] Cannot load IBTrACS names: {e}")
            return [], []

    def get_storm_name(self, sid):
        """Lấy tên bão kèm năm (VD: DAMREY (2017))"""
        if sid in self.ty_ids:
            idx = self.ty_ids.index(sid)
            name = self.ty_names[idx]
            year = sid[:4]
            return f"{name} ({year})"
        return f"UNKNOWN ({sid})"

    def calculate_hss(self, tp, tn, fp, fn):
        """Tính Heidke Skill Score"""
        N = tp + tn + fp + fn
        if N == 0: return np.nan
        
        acc = (tp + tn) / N
        sf = ((tp + fn)/N * (tp + fp)/N) + ((tn + fn)/N * (tn + fp)/N)
        
        if 1 - sf == 0: return np.nan
        return (acc - sf) / (1 - sf)

    def calculate_advanced_metrics(self, tp, fn, fp, tn):
        """Tính POD, FARate, FARatio"""
        pod = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        farate = fp / (fp + tn) if (fp + tn) > 0 else 0.0 # False Alarm Rate
        faratio = fp / (tp + fp) if (tp + fp) > 0 else 0.0 # False Alarm Ratio
        return pod, farate, faratio

    def evaluate_ri(self, preds, trues, sids):
        """
        Đánh giá khả năng dự báo Tăng cường độ nhanh (RI).
        RI Definition: Tăng >= 14 m/s (approx 30kt) trong 24h.
        """
        # 1. Gom dữ liệu theo từng cơn bão
        trues_by_sid = {}
        preds_by_sid = {}

        # preds/trues shape: [N, 4] -> [Lat, Lon, Umax, Press]
        # Ta quan tâm cột Umax (index 2)
        for sid, t_val, p_val in zip(sids, trues, preds):
            if sid not in trues_by_sid:
                trues_by_sid[sid] = []
                preds_by_sid[sid] = []
            trues_by_sid[sid].append(t_val)
            preds_by_sid[sid].append(p_val)

        # Chuyển sang numpy
        for k in trues_by_sid:
            trues_by_sid[k] = np.array(trues_by_sid[k])
            preds_by_sid[k] = np.array(preds_by_sid[k])

        # 2. Detect RI Events
        ri_trues_sids = set()
        ri_preds_sids = set()
        all_sids = set(trues_by_sid.keys())

        THRESHOLD = 14.0 # 14 m/s trong 24h (4 bước thời gian 6h)

        for sid in all_sids:
            t_vals = trues_by_sid[sid][:, 2] # Cột Umax
            p_vals = preds_by_sid[sid][:, 2] # Cột Umax dự báo

            # Check RI Thực tế
            is_true_ri = False
            for i in range(4, len(t_vals)):
                # Kiểm tra độ lệch so với 24h trước (index i-4)
                if (t_vals[i] - t_vals[i-4]) >= THRESHOLD:
                    is_true_ri = True; break
            if is_true_ri: ri_trues_sids.add(sid)

            # Check RI Dự báo
            is_pred_ri = False
            for i in range(4, len(p_vals)):
                if (p_vals[i] - p_vals[i-4]) >= THRESHOLD:
                    is_pred_ri = True; break
            if is_pred_ri: ri_preds_sids.add(sid)

        # 3. Tính Metrics
        tp = len(ri_trues_sids & ri_preds_sids)
        fn = len(ri_trues_sids - ri_preds_sids)
        fp = len(ri_preds_sids - ri_trues_sids)
        tn = len(all_sids - (ri_trues_sids | ri_preds_sids))

        hss = self.calculate_hss(tp, tn, fp, fn)
        pod, farate, faratio = self.calculate_advanced_metrics(tp, fn, fp, tn)

        # 4. Tìm các ca RI bị bỏ sót (Missed RI)
        missed_sids = list(ri_trues_sids - ri_preds_sids)
        missed_names = [self.get_storm_name(s) for s in missed_sids]
        
        # Tìm các ca báo động giả (False Alarm)
        false_alarm_sids = list(ri_preds_sids - ri_trues_sids)
        false_alarm_names = [self.get_storm_name(s) for s in false_alarm_sids]

        return {
            "TP": tp, "FN": fn, "FP": fp, "TN": tn,
            "HSS": hss, "POD": pod, "FARate": farate, "FARatio": faratio,
            "Missed_RI": missed_names,
            "False_Alarm_RI": false_alarm_names,
            "Total_RI_True": len(ri_trues_sids),
            "Total_RI_Pred": len(ri_preds_sids)
        }