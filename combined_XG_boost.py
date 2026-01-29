# ============================================================
# Combined HOG + LBP + Optical Flow with XGBoost
# ============================================================

import os
import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tqdm import tqdm
import joblib
import torch
from xgboost import XGBClassifier

# ============================================================
# CUDA CHECK (INFO ONLY)
# ============================================================

print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA Device Count:", torch.cuda.device_count())
    print("CUDA Device Name:", torch.cuda.get_device_name(0))

# ============================================================
# CONFIGURATION
# ============================================================

VIDEO_DIR = r"C:\Users\UIET\Desktop\OPTICAL\Dataset_original"
MODEL_PATH = "xgboost_combined_hog_lbp_opticalflow_mean.pkl"

N_FRAMES = 20
LBP_POINTS = 8
LBP_RADIUS = 1

# Optical flow histogram config
MAG_BINS = 8
ANG_BINS = 8
MAG_RANGE = (0, 10)
ANG_RANGE = (0, 2 * np.pi)

# ============================================================
# OPTICAL FLOW FEATURE EXTRACTION (MEAN ONLY)
# ============================================================

def extract_optical_flow_features(video_path,
                                  mag_bins=MAG_BINS,
                                  ang_bins=ANG_BINS,
                                  mag_range=MAG_RANGE,
                                  ang_range=ANG_RANGE):

    cap = cv2.VideoCapture(video_path)
    ret, prev_frame = cap.read()
    if not ret:
        cap.release()
        return np.zeros(mag_bins + ang_bins)

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    features = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray, None,
            0.5, 3, 15, 3, 5, 1.2, 0
        )

        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        mag_hist = cv2.calcHist(
            [mag.astype("float32")], [0], None,
            [mag_bins], [mag_range[0], mag_range[1]]
        )
        ang_hist = cv2.calcHist(
            [ang.astype("float32")], [0], None,
            [ang_bins], [ang_range[0], ang_range[1]]
        )

        mag_hist = cv2.normalize(mag_hist, mag_hist).flatten()
        ang_hist = cv2.normalize(ang_hist, ang_hist).flatten()

        features.append(np.concatenate((mag_hist, ang_hist)))
        prev_gray = gray

    cap.release()

    if len(features) == 0:
        return np.zeros(mag_bins + ang_bins)

    return np.mean(features, axis=0)

# ============================================================
# HOG + LBP FEATURE EXTRACTION
# ============================================================

def extract_hog_features(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return hog(
        gray,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        orientations=9,
        block_norm="L2-Hys"
    )

def extract_lbp_features(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(
        gray,
        P=LBP_POINTS,
        R=LBP_RADIUS,
        method="uniform"
    )
    hist, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, LBP_POINTS + 3),
        range=(0, LBP_POINTS + 2)
    )
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    return hist

# ============================================================
# COMBINED VIDEO FEATURE EXTRACTION
# ============================================================

def extract_video_features(video_path, n_frames=N_FRAMES):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return None

    frame_indices = np.linspace(0, total_frames - 1, n_frames, dtype=int)

    hog_features = []
    lbp_features = []

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        hog_features.append(extract_hog_features(frame))
        lbp_features.append(extract_lbp_features(frame))

    cap.release()

    if len(hog_features) == 0:
        return None

    hog_mean = np.mean(hog_features, axis=0)
    lbp_mean = np.mean(lbp_features, axis=0)
    flow_feat = extract_optical_flow_features(video_path)

    return np.concatenate([hog_mean, lbp_mean, flow_feat])

# ============================================================
# DATASET BUILDING
# ============================================================

def build_dataset(video_root):
    X, y = [], []
    classes = [c for c in sorted(os.listdir(video_root)) if not c.startswith(".")]
    print(f"Detected classes: {classes}")

    for label, class_name in enumerate(classes):
        class_dir = os.path.join(video_root, class_name)
        if not os.path.isdir(class_dir):
            continue

        count = 0
        for root, _, files in os.walk(class_dir):
            videos = [f for f in files if f.endswith((".mp4", ".avi", ".mov", ".mkv"))]
            for vid in tqdm(videos, desc=f"Processing {class_name}"):
                video_path = os.path.join(root, vid)
                features = extract_video_features(video_path)
                if features is not None:
                    X.append(features)
                    y.append(label)
                    count += 1

        print(f"✅ Processed {count} videos from '{class_name}'")

    return np.array(X), np.array(y), classes

# ============================================================
# TRAINING AND EVALUATION (XGBOOST)
# ============================================================

def train_and_evaluate(X, y, classes):
    print("\n🚀 Starting training and evaluation (XGBoost)...")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = XGBClassifier(
        n_estimators=400,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=len(classes),
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"\n🎯 Accuracy: {acc * 100:.2f}%")
    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=classes))

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.show()

    return model, scaler

# ============================================================
# MAIN
# ============================================================

def main():
    print("🔍 Building dataset...")
    X, y, classes = build_dataset(VIDEO_DIR)

    if len(X) == 0:
        print("⚠️ No videos found! Check dataset path.")
        return

    print(f"✅ Feature matrix shape: {X.shape}")

    model, scaler = train_and_evaluate(X, y, classes)

    joblib.dump(
        {"model": model, "scaler": scaler, "classes": classes},
        MODEL_PATH
    )
    print(f"\n💾 Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()
