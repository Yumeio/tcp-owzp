import os
import pickle
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

CONFIG = {
    "csv_file": "./data/",
    "output_dir": "./dataset",
    "seq_length": 5
}

os.makedirs(CONFIG['output_dir'], exist_ok=True)

def create_data(
    df, 
    scaler_x, 
    scaler_y,
    is_train = False
): 
    feats = ['lat', 'lon', 'umax', 'press',
        'u24_past', 'mov_speed', 'mov_angle',
        'u_st', 'v_st',
        'ws_200', 'owz_500', 'owz_850', 'rh_700', 'rh_925', 'sph_925'
    ]
    
    targets = ['lat', 'lon', 'umax', 'press']
    
    if any(c not in df.columns for c in feats): raise ValueError("Missing columns!")
    
    data_x, data_y = df[feats].values, df[targets].values
    
    if is_train:
        scaler_x, scaler_y = MinMaxScaler(), MinMaxScaler()
        scaler_x.fit(data_x); scaler_y.fit(data_y)
        joblib.dump(scaler_x, os.path.join(CONFIG['output_dir'], 'scaler_x.pkl'))
        joblib.dump(scaler_y, os.path.join(CONFIG['output_dir'], 'scaler_y.pkl'))
        
    # Transform normalization [0, 1]
    df_s = pd.DataFrame(scaler_x.transform(data_x), columns=feats)
    y_s = scaler_y.transform(data_y)
    for i, c in enumerate(targets): df_s[f"TG_{c}"] = y_s[:, i]
    
    X, y = [], []
    # Group by SID using original DF indices
    for sid, idxs in df.groupby('sid', sort=False).indices.items():
        grp = df_s.iloc[idxs].reset_index(drop=True)
        if len(grp) <= CONFIG['seq_length']: continue
        
        arr_x = grp[feats].values
        arr_y = grp[[f"TG_{c}" for c in targets]].values
        
        for i in range(len(grp) - CONFIG['']):
            X.append(arr_x[i : i+CONFIG['seq_length']])
            y.append(arr_y[i + CONFIG['seq_length']]) # Predict t+1

    return np.array(X), np.array(y), scaler_x, scaler_y

def create_dataset(
    train_split = (1980, 1996),
    val_split = (1996, 1998),
    test_split = (1999, 2000)
):
    if not os.path.exists(CONFIG['csv_file']): return print("CSV Missing")
    
    df = pd.read_csv(CONFIG['csv_file'])
    df['year'] = pd.to_datetime(df['time']).dt.year
    
    train = df[(df.y >= train_split[0]) & (df.y <= train_split[1])]
    val   = df[(df.y >= val_split[0]) & (df.y <= val_split[1])]
    test  = df[(df.y >= test_split[0]) & (df.y <= test_split[1])]
    
    print("Building Train...")
    Xt, yt, sx, sy = create_data(train, None, None, True)
    print("Building Val...")
    Xv, yv, _, _ = create_data(val, sx, sy, False)
    print("Building Test...")
    Xte, yte, _, _ = create_data(test, sx, sy, False)
    
    for name, data in zip(['train', 'val', 'test'], [(Xt, yt), (Xv, yv), (Xte, yte)]):
        with open(os.path.join(CONFIG['output_dir'], f'{name}_dataset.pkl'), 'wb') as f:
            pickle.dump(data, f)
            
    print(f"Done. Train shape: {Xt.shape}")