import torch
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import pickle
import os
import joblib
import random

from model import Transformer
from dataset import TCDataset

CFG = {
    "data_dir": "./dataset",
    "checkpoint": "./checkpoint/model.pth",
    "scaler_path": "./dataset/scaler_y.pkl",
    "save_dir": "./reports/",
    
    "out_dim": 4, "emb": 128, "n_head": 8, "layer": 4, "drop": 0.0,
    "device": torch.device('cuda' if torch.cuda.is_available() else 'cpu')
}

os.makedirs(CFG['save_dir'], exist_ok=True)


def get_predictions():
    print(">>> Running Inference...")
    
    ds = TCDataset(os.path.join(CFG['data_dir'], 'test_dataset.pkl'))
    input_dim = ds.X.shape[2]
    
    model = Transformer(input_dim, CFG["out_dim"], CFG["emb"], CFG["n_head"], 
                        CFG["layer"], CFG["layer"], CFG["drop"]).to(CFG['device'])
    model.load_state_dict(torch.load(CFG['checkpoint'], map_location=CFG['device']))
    model.eval()

    preds, trues = [], []
    batch_size = 256
    
    with torch.no_grad():
        for i in range(0, len(ds.X), batch_size):
            x_batch = ds.X[i:i+batch_size].to(CFG['device'])
            
            # Decoder Input Strategy
            dec_in = x_batch[:, -1, :CFG['out_dim']].unsqueeze(1)
            pred = model(x_batch, dec_in).squeeze(1)
            
            preds.append(pred.cpu().numpy())
            trues.append(ds.y[i:i+batch_size].numpy())

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    
    scaler = joblib.load(CFG['scaler_path'])
    preds_real = scaler.inverse_transform(preds)
    trues_real = scaler.inverse_transform(trues)
    
    # 4. Gom nhóm theo SID (Reconstruct Tracks)
    # Dictionary structure: tracks[sid] = {'lat_t': [], 'lon_t': [], 'lat_p': [], 'lon_p': []}
    tracks = {}
    
    for idx, sid in enumerate(ds.sids):
        if sid not in tracks:
            tracks[sid] = {'true': [], 'pred': []}
        
        # Lưu toạ độ (Lat, Lon) - Giả sử cột 0 là Lat, cột 1 là Lon
        # True Track
        tracks[sid]['true'].append([trues_real[idx, 0], trues_real[idx, 1]])
        # Pred Track
        tracks[sid]['pred'].append([preds_real[idx, 0], preds_real[idx, 1]])

    # Chuyển sang numpy array cho dễ vẽ
    for sid in tracks:
        tracks[sid]['true'] = np.array(tracks[sid]['true'])
        tracks[sid]['pred'] = np.array(tracks[sid]['pred'])
        
    return tracks

# =============================================================================
def plot_storm_track(sid, data, save_path):
    """Vẽ 1 cơn bão cụ thể lên bản đồ"""
    true_track = data['true']
    pred_track = data['pred']
    
    # Tạo Plot với Cartopy
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    
    # Thêm nền bản đồ
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_geometries(cfeature.BORDERS.geometries(), ccrs.PlateCarree(), linestyle=':', linewidth=0.5)
    ax.add_feature(cfeature.LAND, facecolor='#f0f0f0')
    ax.add_feature(cfeature.OCEAN, facecolor='#e0f7fa')
    
    # Vẽ lưới kinh vĩ độ
    gl = ax.gridlines(draw_labels=True, linestyle='--', alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    # --- VẼ TRACK ---
    # 1. Đường thực tế (Màu đen/xanh đậm)
    ax.plot(true_track[:, 1], true_track[:, 0], 
            color='black', linewidth=2, label='Best Track (Actual)', 
            transform=ccrs.PlateCarree())
    # Đánh dấu điểm khởi đầu thực tế
    ax.scatter(true_track[0, 1], true_track[0, 0], color='black', marker='o', s=50)

    # 2. Đường dự báo (Màu đỏ)
    ax.plot(pred_track[:, 1], pred_track[:, 0], 
            color='red', linewidth=2, linestyle='--', label='AI Prediction', 
            transform=ccrs.PlateCarree())
    
    # Thiết lập phạm vi bản đồ (Zoom vào khu vực bão hoạt động)
    buffer = 5 # độ mở rộng
    min_lon = min(true_track[:, 1].min(), pred_track[:, 1].min()) - buffer
    max_lon = max(true_track[:, 1].max(), pred_track[:, 1].max()) + buffer
    min_lat = min(true_track[:, 0].min(), pred_track[:, 0].min()) - buffer
    max_lat = max(true_track[:, 0].max(), pred_track[:, 0].max()) + buffer
    
    ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())
    
    plt.title(f"Typhoon Track Prediction: {sid}", fontsize=14, fontweight='bold')
    plt.legend(loc='lower right')
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {save_path}")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    all_tracks = get_predictions()
    print(f"Total storms in Test set: {len(all_tracks)}")
    
    set_seed(42)
    sids_to_plot = random.sample(list(all_tracks.keys()), min(5, len(all_tracks)))
    print(f"Plotting {len(sids_to_plot)} storms...")
    
    for sid in sids_to_plot:
        if sid in all_tracks:
            save_name = os.path.join(CFG['save_dir'], f"track_{sid}.png")
            plot_storm_track(sid, all_tracks[sid], save_name)
        else:
            print(f"SID {sid} not found in test set.")
            
    print("Done.")