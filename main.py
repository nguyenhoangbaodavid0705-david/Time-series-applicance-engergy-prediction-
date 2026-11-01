import torch
import pandas as pd
from torch.utils.data import DataLoader, random_split

from dataset import AppliancesDataset
from model import TimeSeriesTransformer, LSTMBaseline
from train_and_evaluate import train_model, evaluate_predictions

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--> Dang su dung thiet bi: {device}")

    # Đọc file dữ liệu (nằm cùng thư mục)
    data_path = 'energydata_complete.csv'
    print(f"--> Dang tai du lieu tu {data_path}...")
    df = pd.read_csv(data_path)

    seq_len = 48  
    pred_len = 1  
    
    dataset = AppliancesDataset(df, target_col='Appliances', seq_len=seq_len, pred_len=pred_len)

    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    num_features = dataset.data_x.shape[1]

    # Huấn luyện Transformer Encoder
    print("\n==========================================")
    print("      TRAINING TRANSFORMER ENCODER        ")
    print("==========================================")
    transformer_model = TimeSeriesTransformer(num_features=num_features, d_model=64, n_heads=4, num_layers=2)
    train_model(transformer_model, train_loader, val_loader, epochs=10, lr=1e-3, device=device)
    
    print("\n--> Danh gia Transformer tren tap Test:")
    evaluate_predictions(transformer_model, test_loader, dataset.scaler_y, device=device)

    # Huấn luyện LSTM Baseline
    print("\n==========================================")
    print("          TRAINING LSTM BASELINE          ")
    print("==========================================")
    lstm_model = LSTMBaseline(num_features=num_features, hidden_dim=64, num_layers=2)
    train_model(lstm_model, train_loader, val_loader, epochs=10, lr=1e-3, device=device)

    print("\n--> Danh gia LSTM tren tap Test:")
    evaluate_predictions(lstm_model, test_loader, dataset.scaler_y, device=device)

if __name__ == '__main__':
    main()