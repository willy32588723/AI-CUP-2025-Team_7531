import argparse
import torch
import torch.optim as optim
import pandas as pd
from torch_optimizer import Lookahead
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from model import MultiBranchInceptionModel, smooth_focal_loss
from dataset import get_dataloader, get_test_dataloader
import torch.nn as nn
from tqdm import tqdm
import numpy as np

class EarlyStopping:
    def __init__(self, patience=5, delta=1e-4):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def __call__(self, score):
        if self.best_score is None or score > self.best_score + self.delta:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

def get_class_weights(df):
    from sklearn.utils.class_weight import compute_class_weight
    weights = {}
    for key, colname in [
        ("gender", "gender"),
        ("hand", "hold racket handed"),
        ("players", "play years"),
        ("level", "level")
    ]:
        labels = df[colname]
        classes = np.unique(labels)
        w = compute_class_weight(class_weight="balanced", classes=classes, y=labels)
        weights[key] = torch.tensor(w, dtype=torch.float32)
    return weights

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    preds, targets = {k: [] for k in ['gender', 'hand', 'players', 'level']}, {k: [] for k in ['gender', 'hand', 'players', 'level']}
    pred_labels, true_labels = {k: [] for k in preds}, {k: [] for k in preds}
    for data, stat, gender, hand, players, level, mode in tqdm(loader, leave=False):
        data = data.to(device)
        stat = stat.to(device)
        gender = gender.to(device)
        hand = hand.to(device)
        players = players.to(device)
        level = level.to(device)
        mode = mode.to(device)
        optimizer.zero_grad()
        outputs = model(data, stat, mode)
        loss = sum([
            criterion['gender'](outputs['gender'], gender),
            criterion['hand'](outputs['hand'], hand),
            criterion['players'](outputs['players'], players),
            criterion['level'](outputs['level'], level)
        ])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        for key in preds:
            if key in ['gender', 'hand']:
                sigmoid_out = torch.sigmoid(outputs[key]).detach().cpu()
                preds[key].extend(sigmoid_out.numpy())
                pred_labels[key].extend((sigmoid_out > 0.5).int().numpy())
            else:
                softmax_out = outputs[key].softmax(dim=1).detach().cpu()
                preds[key].extend(softmax_out.numpy().tolist())
                pred_labels[key].extend(outputs[key].argmax(dim=1).cpu().numpy())
        true_labels['gender'].extend(gender.cpu().numpy())
        true_labels['hand'].extend(hand.cpu().numpy())
        true_labels['players'].extend(players.cpu().numpy())
        true_labels['level'].extend(level.cpu().numpy())
    aucs = {}
    for k in preds:
        y_true = np.array(true_labels[k])
        y_score = np.array(preds[k])
        if k in ['players', 'level']:
            aucs[k] = roc_auc_score(y_true, y_score, multi_class='ovr', average='micro')
        else:
            aucs[k] = roc_auc_score(y_true, y_score)
    accs = {k: accuracy_score(np.array(true_labels[k]), np.array(pred_labels[k])) for k in preds}
    return total_loss / len(loader), accs, aucs, np.mean(list(aucs.values()))

@torch.no_grad()
def evaluate_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    preds, targets = {k: [] for k in ['gender', 'hand', 'players', 'level']}, {k: [] for k in ['gender', 'hand', 'players', 'level']}
    pred_labels, true_labels = {k: [] for k in preds}, {k: [] for k in preds}
    
    for data, stat, gender, hand, players, level, mode in tqdm(loader, leave=False):
        data = data.to(device)
        stat = stat.to(device)
        gender = gender.to(device)
        hand = hand.to(device)
        players = players.to(device)
        level = level.to(device)
        mode = mode.to(device)
        outputs = model(data, stat, mode)
        loss = sum([
            criterion['gender'](outputs['gender'], gender),
            criterion['hand'](outputs['hand'], hand),
            criterion['players'](outputs['players'], players),
            criterion['level'](outputs['level'], level)
        ])
        total_loss += loss.item()

        for key in preds:
            if key in ['gender', 'hand']:
                sigmoid_out = torch.sigmoid(outputs[key]).detach().cpu()
                preds[key].extend(sigmoid_out.numpy())
                pred_labels[key].extend((sigmoid_out > 0.5).int().numpy())
            else:
                softmax_out = outputs[key].softmax(dim=1).detach().cpu()
                preds[key].extend(softmax_out.numpy().tolist())
                pred_labels[key].extend(outputs[key].argmax(dim=1).cpu().numpy())
        
        true_labels['gender'].extend(gender.cpu().numpy())
        true_labels['hand'].extend(hand.cpu().numpy())
        true_labels['players'].extend(players.cpu().numpy())
        true_labels['level'].extend(level.cpu().numpy())

    aucs = {}
    for k in preds:
        y_true = np.array(true_labels[k])
        y_score = np.array(preds[k])
        if k in ['players', 'level']:
            aucs[k] = roc_auc_score(y_true, y_score, multi_class='ovr', average='micro')
        else:
            aucs[k] = roc_auc_score(y_true, y_score)

    accs = {
        k: accuracy_score(np.array(true_labels[k]), np.array(pred_labels[k])) for k in preds
    }

    return total_loss / len(loader), accs, aucs, np.mean(list(aucs.values()))

def predict_test(model, test_loader, device):
    model.eval()
    predictions = {}
    with torch.no_grad():
        for ids, data, stat, modes in tqdm(test_loader, desc="Testing"):
            data = data.to(device)
            stat = stat.to(device)
            modes = modes.to(device)
            outputs = model(data, stat, modes)
            gender_probs = torch.sigmoid(outputs['gender']).cpu().numpy()
            hand_probs = torch.sigmoid(outputs['hand']).cpu().numpy()
            players_probs = outputs['players'].softmax(dim=1).cpu().numpy()
            level_probs = outputs['level'].softmax(dim=1).cpu().numpy()
            for i, uid in enumerate(ids):
                row = [
                    float(1.0 - gender_probs[i]),
                    float(1.0 - hand_probs[i]),
                    *players_probs[i],
                    *level_probs[i],
                ]
                predictions[uid] = row
    return predictions

def save_predictions_to_csv(predictions, filename="submission.csv"):
    rows = []
    for uid in predictions:
        prob = predictions[uid]
        row = {
            "unique_id": uid,
            "gender": round(prob[0], 4),
            "hold racket handed": round(prob[1], 4),
            "play years_0": round(prob[2], 4),
            "play years_1": round(prob[3], 4),
            "play years_2": round(prob[4], 4),
            "level_2": round(prob[5], 4),
            "level_3": round(prob[6], 4),
            "level_4": round(prob[7], 4),
            "level_5": round(prob[8], 4),
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(filename, index=False, float_format="%.4f")
    print(f"預測結果已儲存到 {filename}（保留 4 位小數）")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_info', type=str ,default='39_Training_Dataset/train_info_augmented.csv')
    parser.add_argument('--train_data', type=str, default='39_Training_Dataset/train_data_augmented_level_year')
    parser.add_argument('--test_info', type=str ,default='39_Test_Dataset/test_info.csv')
    parser.add_argument('--test_data', type=str, default='39_Test_Dataset/test_data')
    parser.add_argument('--mode', choices=['train_full', 'train_val', 'test'], default='train_val')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda',)
    parser.add_argument('--model', type=str, default="best_inception_model.pt")
    args = parser.parse_args()
    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    full_df = pd.read_csv(args.train_info)
    train_loader = get_dataloader(args.train_info, args.train_data, args.batch)
    model = MultiBranchInceptionModel().to(device)
    class_weights = get_class_weights(full_df)
    criterion = {
        'gender': nn.BCEWithLogitsLoss(pos_weight=class_weights['gender'][1].to(device)),
        'hand': nn.BCEWithLogitsLoss(pos_weight=class_weights['hand'][1].to(device)),
        'players': lambda input, target: smooth_focal_loss(input, target, alpha=class_weights['players'].to(device)),
        'level':   lambda input, target: smooth_focal_loss(input, target, alpha=class_weights['level'].to(device)),
    }
    best_auc = 0
    base_optim = optim.AdamW(model.parameters(), lr=args.lr)
    optimizer = Lookahead(base_optim)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', 0.5, 5)
    if args.mode == "train_full":
        early_stopping = EarlyStopping(patience=10)
        for epoch in range(args.epochs):
            lr = optimizer.param_groups[0]['lr']
            train_loss, train_acc, train_aucs, train_avg_auc = train_one_epoch(model, train_loader, criterion, optimizer, device, lr)
            print(f"Epoch {epoch+1:02d} 🔁 LR: {lr:.2e}")
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc}")
            print(f"Train AUCs: {train_aucs} | Train Avg AUC: {train_avg_auc:.4f}")
            if train_avg_auc > best_auc:
                best_auc = train_avg_auc
                torch.save(model.state_dict(), args.model)
                print("Model saved.")
            scheduler.step(train_avg_auc)
            early_stopping(train_avg_auc)
            if early_stopping.early_stop:
                print(f"⏹️ Early stopping triggered at epoch {epoch+1}")
                break
    if args.mode == "train_val":
        train_df, val_df = train_test_split(
            full_df, test_size=0.1, random_state=42, 
            stratify=full_df[['gender', 'hold racket handed', 'play years', 'level']]
        )
        train_loader = get_dataloader(train_df, args.train_data, args.batch, shuffle=True)
        val_loader   = get_dataloader(val_df,   args.train_data, args.batch, shuffle=False)
        model = MultiBranchInceptionModel().to(device)
        class_weights = get_class_weights(train_df)
        criterion = {
            'gender': nn.BCEWithLogitsLoss(pos_weight=class_weights['gender'][1].to(device)),
            'hand': nn.BCEWithLogitsLoss(pos_weight=class_weights['hand'][1].to(device)),
            'players': lambda input, target: smooth_focal_loss(input, target, alpha=class_weights['players'].to(device)),
            'level':   lambda input, target: smooth_focal_loss(input, target, alpha=class_weights['level'].to(device)),
        }
        best_auc = 0
        base_optim = optim.AdamW(model.parameters(), lr=args.lr)
        optimizer = Lookahead(base_optim)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', 0.5, 5)
        early_stopping = EarlyStopping(patience=10)
        for epoch in range(args.epochs):
            lr = optimizer.param_groups[0]['lr']
            train_loss, train_acc, train_aucs, train_avg_auc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc, val_aucs, val_avg_auc = evaluate_one_epoch(model, val_loader, criterion, device)
            print(f"Epoch {epoch+1:02d} 🔁 LR: {lr:.2e}")
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_aucs} | Train Avg AUC: {train_avg_auc:.4f}")
            print(f"Valid Loss: {val_loss:.4f} | Valid Acc: {val_aucs} | Valid Avg AUC: {val_avg_auc:.4f}")
            if val_avg_auc > best_auc:
                best_auc = val_avg_auc
                torch.save(model.state_dict(), args.model)
                print("Model saved.")
            scheduler.step(val_avg_auc)
            early_stopping(val_avg_auc)
            if early_stopping.early_stop:
                print(f"⏹️ Early stopping triggered at epoch {epoch+1}")
                break
    elif args.mode == "test":
        # 載入測試資料
        test_loader = get_test_dataloader(args.test_info, args.test_data, batch_size=args.batch)
        # 載入最佳模型
        model = MultiBranchInceptionModel().to(device)
        model.load_state_dict(torch.load(args.model, map_location=device))

        # 預測
        predictions = predict_test(model, test_loader, device)

        # 儲存 CSV
        save_predictions_to_csv(predictions, filename="submission.csv")

if __name__ == '__main__':
    main()
