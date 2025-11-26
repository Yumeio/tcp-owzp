import torch
import numpy as np
import pickle, os, joblib
from torch.utils.data import DataLoader, Dataset

from model import Transformer
from dataset import TCDataset
from metrics import RIMetrics  

CFG = {
    "data_dir": "./dataset",
    "checkpoint": "./checkpoint/model.pth",
    "scaler_path": "./dataset/scaler_y.pkl",
    "ibtracs_path": "./data/IBTrACS.WP.v04r01.nc", # Đường dẫn IBTrACS
    "out_dim": 4, 
    "emb": 256, 
    "n_head": 8, 
    "layer": 4, 
    "drop": 0.0, 
    "bs": 128,
    "device": torch.device('cuda' if torch.cuda.is_available() else 'cpu')
}

def main():
    print(f"--- Eval on {CFG['device']} ---")
    
    # 1. Load Data
    ds = TCDataset(os.path.join(CFG['data_dir'], 'test_dataset.pkl'))
    loader = DataLoader(ds, batch_size=CFG['bs'], shuffle=False)
    input_dim = ds.X.shape[2]

    # 2. Load Model
    model = Transformer(input_dim, CFG["out_dim"], CFG["emb"], CFG["n_head"], 
                        CFG["layer"], CFG["layer"], CFG["drop"]).to(CFG['device'])
    model.load_state_dict(torch.load(CFG['checkpoint'], map_location=CFG['device']))
    model.eval()

    # 3. Inference
    preds, trues, sids_all = [], [], []
    with torch.no_grad():
        for x, y, sids in loader:
            x = x.to(CFG['device'])
            dec_in = x[:, -1, :CFG['out_dim']].unsqueeze(1)
            pred = model(x, dec_in).squeeze(1)
            
            preds.append(pred.cpu().numpy())
            trues.append(y.cpu().numpy())
            sids_all.extend(sids) # Gom sids lại

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)

    # 4. Inverse Scale
    scaler = joblib.load(CFG['scaler_path'])
    preds_real = scaler.inverse_transform(preds)
    trues_real = scaler.inverse_transform(trues)

    # 5. --- TÍNH TOÁN RI METRICS (PHẦN MỚI) ---
    print("\n>>> Calculating RI Metrics...")
    ri_calculator = RIMetrics(CFG['ibtracs_path'])
    
    metrics = ri_calculator.evaluate_ri(preds_real, trues_real, sids_all)

    # 6. In kết quả đẹp
    print("\n" + "="*50)
    print(f"{'RAPID INTENSIFICATION (RI) REPORT':^50}")
    print("="*50)
    print(f"Total Storms Tested: {metrics['TP'] + metrics['FN'] + metrics['FP'] + metrics['TN']}")
    print(f"True RI Events:      {metrics['Total_RI_True']}")
    print(f"Predicted RI Events: {metrics['Total_RI_Pred']}")
    print("-" * 50)
    print(f"TP (Correct RI):     {metrics['TP']}")
    print(f"TN (Correct Non-RI): {metrics['TN']}")
    print(f"FP (False Alarm):    {metrics['FP']}")
    print(f"FN (Missed RI):      {metrics['FN']}")
    print("-" * 50)
    print(f"HSS Score:           {metrics['HSS']:.4f}")
    print(f"POD (Detection Rate):{metrics['POD']:.4f}")
    print(f"FAR (False Alarm):   {metrics['FARatio']:.4f}")
    print("="*50)

    print("\n[Missed RI Storms (Real RI but Model missed)]:")
    print(metrics['Missed_RI'][:10]) # In 10 cái đầu
    
    print("\n[False Alarm Storms (Model predicted RI but Wrong)]:")
    print(metrics['False_Alarm_RI'][:10])

if __name__ == "__main__":
    main()