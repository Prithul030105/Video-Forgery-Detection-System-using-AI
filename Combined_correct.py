import os
import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tqdm import tqdm
import joblib

# ==============================
# CONFIGURATION
# ==============================

TRAIN_DIR = r"C:\Users\UIET\Desktop\OPTICAL\forgery_kaggle\Training"
TEST_DIR = r"C:\Users\UIET\Desktop\OPTICAL\forgery_kaggle\Testing"
MODEL_PATH = "svm_combined_hog_lbp_opticalflow_final.pkl"

N_FRAMES = 20
LBP_POINTS = 8
LBP_RADIUS = 1

MAG_BINS = 8
ANG_BINS = 8
MAG_RANGE = (0, 10)
ANG_RANGE = (0, 2 * np.pi)

USE_GPU = cv2.cuda.getCudaEnabledDeviceCount() > 0
print(f"⚙️ CUDA GPU Available: {USE_GPU}")

# ==============================
# FEATURE EXTRACTION FUNCTIONS
# ==============================

def extract_optical_flow_features(video_path, mag_bins=MAG_BINS, ang_bins=ANG_BINS,
                                  mag_range=MAG_RANGE, ang_range=ANG_RANGE):
    """Extract GPU-accelerated optical flow mean and std histogram features."""
    cap = cv2.VideoCapture(video_path)
    ret, prev_frame = cap.read()
    if not ret:
        cap.release()
        return np.zeros(mag_bins + ang_bins)
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    if USE_GPU:
        gpu_prev = cv2.cuda_GpuMat()
        gpu_next = cv2.cuda_GpuMat()
        gpu_prev.upload(prev_gray)
        flow_calc = cv2.cuda_FarnebackOpticalFlow.create(
            5, 0.5, False, 15, 3, 5, 1.2, 0
        )

    features = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if USE_GPU:
            gpu_next.upload(gray)
            flow = flow_calc.calc(gpu_prev, gpu_next, None)
            flow = flow.download()
            gpu_prev = gpu_next.clone()
        else:
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None,
                                                0.5, 3, 15, 3, 5, 1.2, 0)
            prev_gray = gray

        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        mag_hist = cv2.calcHist([mag.astype('float32')], [0], None,
                                [mag_bins], [mag_range[0], mag_range[1]])
        ang_hist = cv2.calcHist([ang.astype('float32')], [0], None,
                                [ang_bins], [ang_range[0], ang_range[1]])
        mag_hist = cv2.normalize(mag_hist, mag_hist).flatten()
        ang_hist = cv2.normalize(ang_hist, ang_hist).flatten()
        features.append(np.concatenate((mag_hist, ang_hist)))

    cap.release()
    if len(features) == 0:
        return np.zeros(mag_bins + ang_bins)
    feats = np.vstack(features)
    return np.concatenate([np.mean(feats, axis=0), np.std(feats, axis=0)])


def extract_hog_features(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return hog(gray, pixels_per_cell=(16, 16), cells_per_block=(2, 2),
               orientations=9, block_norm='L2-Hys')


def extract_lbp_features(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, P=LBP_POINTS, R=LBP_RADIUS, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, LBP_POINTS + 3),
                           range=(0, LBP_POINTS + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    return hist


def extract_video_features(video_path, n_frames=N_FRAMES):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return np.zeros(200)
    frame_indices = np.linspace(0, total_frames - 1, n_frames, dtype=int)
    hog_features, lbp_features = [], []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        hog_features.append(extract_hog_features(frame))
        lbp_features.append(extract_lbp_features(frame))
    cap.release()
    hog_mean = np.mean(hog_features, axis=0) if len(hog_features) else np.zeros(100)
    lbp_mean = np.mean(lbp_features, axis=0) if len(lbp_features) else np.zeros(LBP_POINTS + 2)
    flow_feat = extract_optical_flow_features(video_path)
    return np.concatenate([hog_mean, lbp_mean, flow_feat])


# ==============================
# DATASET BUILDING (RECURSIVE)
# ==============================

def build_dataset(video_root):
    X, y = [], []
    all_video_paths = []
    class_names = []

    print(f"\n📂 Scanning all subfolders under: {video_root}")

    for root, dirs, files in os.walk(video_root):
        videos = [f for f in files if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))]
        if len(videos) > 0:
            label_name = os.path.basename(os.path.dirname(root)) if "forged" in root else os.path.basename(root)
            if label_name not in class_names:
                class_names.append(label_name)
            label = class_names.index(label_name)

            for vid in tqdm(videos, desc=f"Processing {label_name}"):
                video_path = os.path.join(root, vid)
                features = extract_video_features(video_path)
                X.append(features)
                y.append(label)
                all_video_paths.append(video_path)

    print(f"✅ Total videos processed from {video_root}: {len(X)}")
    print(f"🧾 Classes found: {class_names}")
    return np.array(X), np.array(y), class_names


# ==============================
# TRAINING & EVALUATION
# ==============================

def train_and_evaluate(X_train, y_train, X_test, y_test, classes):
    print("\n🚀 Starting training with GridSearchCV...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    param_grid = {
        'C': [1, 10, 100],
        'gamma': ['scale', 0.01, 0.001],
        'kernel': ['rbf']
    }

    grid = GridSearchCV(
        SVC(class_weight='balanced', probability=True),
        param_grid,
        refit=True,
        verbose=1,
        cv=3,
        n_jobs=-1
    )
    grid.fit(X_train_scaled, y_train)
    best_model = grid.best_estimator_
    print("\n✅ Best Parameters:", grid.best_params_)

    y_pred = best_model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n🎯 Accuracy: {acc * 100:.2f}%")
    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=classes))
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap='Blues')
    plt.title("Confusion Matrix")
    plt.show()

    return best_model, scaler


# ==============================
# MAIN
# ==============================

def main():
    print("🔍 Loading training dataset...")
    X_train, y_train, classes = build_dataset(TRAIN_DIR)
    print(f"✅ Training data shape: {X_train.shape}")

    print("\n🔍 Loading testing dataset...")
    X_test, y_test, _ = build_dataset(TEST_DIR)
    print(f"✅ Testing data shape: {X_test.shape}")

    model, scaler = train_and_evaluate(X_train, y_train, X_test, y_test, classes)

    joblib.dump({'model': model, 'scaler': scaler, 'classes': classes}, MODEL_PATH)
    print(f"\n💾 Model saved successfully to {MODEL_PATH}")


if __name__ == "__main__":
    main()
