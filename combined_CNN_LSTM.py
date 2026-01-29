# ============================================================
# CNN-LSTM Video Forgery Detection (HOG + LBP + Optical Flow)
# FULL SINGLE FILE - GPU ENABLED
# ============================================================

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from skimage.feature import hog, local_binary_pattern
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from tqdm import tqdm
import joblib

# ============================================================
# CUDA CHECK
# ============================================================

print("CUDA Available:", torch.cuda.is_available())
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# ============================================================
# CONFIGURATION
# ============================================================

VIDEO_DIR = r"C:\Users\UIET\Desktop\OPTICAL\Dataset_original"
MODEL_PATH = "cnn_lstm_hog_lbp_opticalflow.pkl"

N_FRAMES = 20
LBP_POINTS = 8
LBP_RADIUS = 1

MAG_BINS = 8
ANG_BINS = 8
MAG_RANGE = (0, 10)
ANG_RANGE = (0, 2 * np.pi)

EPOCHS = 30
BATCH_SIZE = 32
LR = 1e-3

# ============================================================
# OPTICAL FLOW FEATURES (MEAN ONLY)
# ============================================================

def extract_optical_flow_features(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, prev = cap.read()
    if not ret:
        cap.release()
        return np.zeros(MAG_BINS + ANG_BINS)

    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    feats = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        mag_hist = cv2.normalize(
            cv2.calcHist([mag.astype(np.float32)], [0], None, [MAG_BINS], MAG_RANGE),
            None
        ).flatten()

        ang_hist = cv2.normalize(
            cv2.calcHist([ang.astype(np.float32)], [0], None, [ANG_BINS], ANG_RANGE),
            None
        ).flatten()

        feats.append(np.concatenate([mag_hist, ang_hist]))
        prev_gray = gray

    cap.release()
    return np.mean(feats, axis=0) if len(feats) else np.zeros(MAG_BINS + ANG_BINS)

# ============================================================
# HOG + LBP
# ============================================================

def extract_hog_features(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return hog(gray, pixels_per_cell=(16,16),
               cells_per_block=(2,2),
               orientations=9,
               block_norm="L2-Hys")

def extract_lbp_features(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, LBP_POINTS, LBP_RADIUS, method="uniform")
    hist, _ = np.histogram(lbp.ravel(),
                           bins=np.arange(0, LBP_POINTS+3),
                           range=(0, LBP_POINTS+2))
    hist = hist.astype(float)
    return hist / (hist.sum() + 1e-6)

# ============================================================
# COMBINED VIDEO FEATURE
# ============================================================

def extract_video_features(video_path):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return None

    idxs = np.linspace(0, total-1, N_FRAMES, dtype=int)
    hogs, lbps = [], []

    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            hogs.append(extract_hog_features(frame))
            lbps.append(extract_lbp_features(frame))

    cap.release()

    hog_mean = np.mean(hogs, axis=0)
    lbp_mean = np.mean(lbps, axis=0)
    flow = extract_optical_flow_features(video_path)

    return np.concatenate([hog_mean, lbp_mean, flow])

# ============================================================
# DATASET BUILDING
# ============================================================

def build_dataset(root):
    X, y = [], []
    classes = sorted([c for c in os.listdir(root) if not c.startswith(".")])
    print("Classes:", classes)

    for label, cls in enumerate(classes):
        path = os.path.join(root, cls)
        for root2, _, files in os.walk(path):
            for f in tqdm(files, desc=f"Processing {cls}"):
                if f.endswith((".mp4",".avi",".mov",".mkv")):
                    feat = extract_video_features(os.path.join(root2, f))
                    if feat is not None:
                        X.append(feat)
                        y.append(label)

    return np.array(X), np.array(y), classes

# ============================================================
# DATASET CLASS
# ============================================================

class VideoDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ============================================================
# CNN + LSTM MODEL
# ============================================================

class CNN_LSTM(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 64, 3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool1d(2)

        self.lstm = nn.LSTM(128, 128, batch_first=True)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.permute(0, 2, 1)
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])

# ============================================================
# TRAINING
# ============================================================

def train_and_evaluate(X, y, classes):
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    train_dl = DataLoader(VideoDataset(Xtr, ytr), batch_size=BATCH_SIZE, shuffle=True)
    test_dl = DataLoader(VideoDataset(Xte, yte), batch_size=BATCH_SIZE)

    model = CNN_LSTM(X.shape[1], len(classes)).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    for e in range(EPOCHS):
        model.train()
        total = 0
        for xb, yb in train_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item()
        print(f"Epoch [{e+1}/{EPOCHS}] Loss: {total/len(train_dl):.4f}")

    model.eval()
    yt, yp = [], []
    with torch.no_grad():
        for xb, yb in test_dl:
            preds = torch.argmax(model(xb.to(DEVICE)), 1).cpu().numpy()
            yp.extend(preds)
            yt.extend(yb.numpy())

    print("\nAccuracy:", accuracy_score(yt, yp))
    print(classification_report(yt, yp, target_names=classes))

    cm = confusion_matrix(yt, yp)
    ConfusionMatrixDisplay(cm, classes).plot(cmap="Blues")
    plt.show()

    return model, scaler

# ============================================================
# MAIN
# ============================================================

def main():
    X, y, classes = build_dataset(VIDEO_DIR)
    model, scaler = train_and_evaluate(X, y, classes)
    joblib.dump({'model': model, 'scaler': scaler, 'classes': classes}, MODEL_PATH)
    print("Model saved:", MODEL_PATH)

if __name__ == "__main__":
    main()
