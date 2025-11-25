import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle, os, time
import matplotlib.pyplot as plt 

from dataset import TCDataset
from model import Transformer

CFG = {
    "data_dir": "./ds",
    "save_path": "./checkpoint/model.pth",
    "plot_path": "./checkpoint/loss_chart.png", 
    "out_dim": 4,
    'emb': 128,
    "n_head": 8,
    "layer": 4,
    "drop": 0.1,
    "bs": 128,
    "lr": 1e-4,
    "epoch": 200,
    "patience": 10, 
    "dev": torch.device('cuda' if torch.cuda.is_available() else 'cpu')
}

# Tạo thư mục checkpoint nếu chưa có
os.makedirs(os.path.dirname(CFG['save_path']), exist_ok=True)

def save_plot(train_hist, val_hist, save_path):
    """Hàm vẽ và lưu biểu đồ Loss"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_hist, label='Train Loss', color='blue')
    plt.plot(val_hist, label='Val Loss', color='orange')
    plt.title('Model Training Progress')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print(f"   [Plot] Saved loss chart to {save_path}")

def train():
    print(f"--- Running on: {CFG['dev']} ---")
    print("Loading Data...")
    loaders = {}
    for phase in ['train', 'val']:
        path = os.path.join(CFG['data_dir'], f'{phase}_dataset.pkl')
        ds = TCDataset(path)
        # num_workers=0 để an toàn nhất, tăng lên nếu chạy Linux
        loaders[phase] = DataLoader(ds, batch_size=CFG['bs'], shuffle=(phase=='train'), num_workers=0)

    sample_x, _, _ = next(iter(loaders['train']))
    input_dim = sample_x.shape[2]
    print(f"Features: {input_dim}, Target: {CFG['out_dim']}")

    model = Transformer(
        input_dim=input_dim,
        output_dim=CFG["out_dim"],
        embed_dim=CFG["emb"],
        num_heads=CFG["n_head"],
        dropout=CFG["drop"],
        num_decoder_layers=CFG["layer"],
        num_encoder_layers=CFG["layer"]
    )
    model.to(CFG['dev']) 

    opt = optim.AdamW(model.parameters(), lr=CFG['lr'], eps=1e-7)
    crit = nn.MSELoss()
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', patience=3)

    best_loss = float('inf')
    early_stop = 0
    
    # <--- 2. Khởi tạo list lưu lịch sử
    history = {'train': [], 'val': []} 

    print("Start Training...")

    try:
        for ep in range(CFG['epoch']):
            start_time = time.time()
            
            # --- TRAIN ---
            model.train()
            t_loss = 0.0
            for x, y, _ in loaders['train']:
                x, y = x.to(CFG['dev']), y.to(CFG['dev'])
                dec_in = x[:, -1, :CFG['out_dim']].unsqueeze(1)

                opt.zero_grad()
                pred = model(x, dec_in).squeeze(1)
                loss = crit(pred, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                
                t_loss += loss.item() 

            # --- VAL ---
            model.eval()
            v_loss = 0.0
            with torch.no_grad():
                for x, y, _ in loaders['val']:
                    x, y = x.to(CFG['dev']), y.to(CFG['dev'])
                    dec_in = x[:, -1, :CFG['out_dim']].unsqueeze(1)
                    pred = model(x, dec_in).squeeze(1)
                    v_loss += crit(pred, y).item() # Dùng .item()

            avg_t = t_loss / len(loaders['train'])
            avg_v = v_loss / len(loaders['val'])
            
            # <--- 3. Lưu lịch sử
            history['train'].append(avg_t)
            history['val'].append(avg_v)
            
            sched.step(avg_v)
            
            # Tính thời gian
            mins = int((time.time() - start_time) / 60)
            secs = int((time.time() - start_time) % 60)
            
            print(f"Ep {ep+1:02d} | Time: {mins}m{secs}s | T: {avg_t:.6f} | V: {avg_v:.6f}")

            if avg_v < best_loss:
                best_loss = avg_v
                early_stop = 0
                torch.save(model.state_dict(), CFG['save_path'])
                print("   > Saved Best Model.")
            else:
                early_stop += 1
                if early_stop >= CFG['patience']:
                    print(">>> Early Stopping Triggered.")
                    break
                    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving plot...")

    # <--- 4. Vẽ biểu đồ khi kết thúc (hoặc ngắt giữa chừng)
    save_plot(history['train'], history['val'], CFG['plot_path'])
    print("Done.")

if __name__ == "__main__":
    train()