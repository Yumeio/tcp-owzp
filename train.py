import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle, os, time

CFG = {
    "data_dir": "./dataset",
    "save_path": "./checkpoint/model.pth",
    "out_dim": 4,
    'emb': 128,
    "n_head": 8,
    "layer": 4,
    "drop": 0.0,
    "bs": 128,
    "lr": 1e-4,
    "epoch": 100,
    "patience": 100,
    "dev": torch.device('cuda' if torch.cuda.is_available() else 'cpu')
}

class TCDataset(Dataset):
    def __init__(self, path):
        with open(path, 'rb') as f: self.X, self.y = pickle.load(f)
        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.float32)
        if self.y.dim() == 3: self.y = self.y.squeeze(1)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

def train():
    print("Loading Data...")
    loaders = {}
    for phase in ['train', 'val']:
        ds = TCDataset(os.path.join(CFG['data_dir'], f'{phase}_dataset.pkl'))
        loaders[phase] = DataLoader(ds, batch_size=CFG['bs'], shuffle=(phase=='train'))

    # Auto-detect Input Dim
    sample_x, _ = next(iter(loaders['train']))
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
    model.to(CFG['dev']) # <--- Added this line

    opt = optim.AdamW(model.parameters(), lr=CFG['lr'], eps=1e-7)
    crit = nn.MSELoss()
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', patience=3)

    best_loss = float('inf')
    early_stop = 0
    print("Start Training...")

    # Create checkpoint directory if it doesn't exist
    os.makedirs(os.path.dirname(CFG['save_path']), exist_ok=True)

    for ep in range(CFG['epoch']):
        model.train()
        t_loss = 0
        for x, y in loaders['train']:
            x, y = x.to(CFG['dev']), y.to(CFG['dev'])

            # Decoder Input: Last step of Encoder input (Only Target cols)
            # 4 cột đầu là Lat, Lon, Umax, Press (theo builder.py)
            dec_in = x[:, -1, :CFG['out_dim']].unsqueeze(1)

            opt.zero_grad()
            pred = model(x, dec_in).squeeze(1)
            loss = crit(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            t_loss += loss.item()

        # Val
        model.eval()
        v_loss = 0
        with torch.no_grad():
            for x, y in loaders['val']:
                x, y = x.to(CFG['dev']), y.to(CFG['dev'])
                dec_in = x[:, -1, :CFG['out_dim']].unsqueeze(1)
                pred = model(x, dec_in).squeeze(1)
                v_loss += crit(pred, y)

        avg_t, avg_v = t_loss/len(loaders['train']), v_loss.item()/len(loaders['val'])
        sched.step(avg_v)
        print(f"Ep {ep+1} | T: {avg_t:.6f} | V: {avg_v:.6f}")

        if avg_v < best_loss:
            best_loss = avg_v; early_stop = 0
            torch.save(model.state_dict(), CFG['save_path'])
            print("  > Saved Best.")
        else:
            early_stop += 1
            if early_stop >= CFG['patience']: print("Early Stop.");
            break

if __name__ == "__main__":
    train()