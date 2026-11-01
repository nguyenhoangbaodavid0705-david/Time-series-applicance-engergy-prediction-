import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

def train_model(model, train_loader, val_loader, epochs=10, lr=1e-3, device='cpu'):
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3)
    
    model.to(device)
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            preds = model(x)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                preds = model(x)
                loss = criterion(preds, y)
                val_loss += loss.item() * x.size(0)
        val_loss /= len(val_loader.dataset)
        
        scheduler.step(val_loss)
        print(f"Epoch {epoch+1}/{epochs} | Train Loss (MSE): {train_loss:.4f} | Val Loss (MSE): {val_loss:.4f}")

def evaluate_predictions(model, test_loader, scaler_y, device='cpu'):
    model.eval()
    preds_list, true_list = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            preds = model(x).cpu().numpy()
            preds_list.append(preds)
            true_list.append(y.numpy())
            
    preds_arr = scaler_y.inverse_transform(np.concatenate(preds_list))
    true_arr = scaler_y.inverse_transform(np.concatenate(true_list))
    
    rmse = np.sqrt(mean_squared_error(true_arr, preds_arr))
    mae = mean_absolute_error(true_arr, preds_arr)
    r2 = r2_score(true_arr, preds_arr)
    
    print(f"Test Results -> RMSE: {rmse:.2f} | MAE: {mae:.2f} | R2 Score: {r2:.3f}")
    return preds_arr, true_arr