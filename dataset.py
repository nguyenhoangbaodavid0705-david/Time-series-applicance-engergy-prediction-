import torch
from torch.utils.data import Dataset
import pandas as pd
from sklearn.preprocessing import RobustScaler

class AppliancesDataset(Dataset):
    def __init__(self, df, target_col='Appliances', seq_len=48, pred_len=1):
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        # Feature engineering: Trích xuất thông tin thời gian
        df['date'] = pd.to_datetime(df['date'])
        df['hour'] = df['date'].dt.hour
        df['dayofweek'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        
        drop_cols = ['date', target_col]
        feature_cols = [c for c in df.columns if c not in drop_cols]
        
        self.scaler_X = RobustScaler()
        self.scaler_y = RobustScaler()
        
        scaled_features = self.scaler_X.fit_transform(df[feature_cols])
        scaled_target = self.scaler_y.fit_transform(df[[target_col]])
        
        self.data_x = scaled_features
        self.data_y = scaled_target
        
    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        x = self.data_x[idx : idx + self.seq_len]
        y = self.data_y[idx + self.seq_len : idx + self.seq_len + self.pred_len]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32).squeeze(-1)