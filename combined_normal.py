# import os
# import cv2
# import numpy as np
# from skimage.feature import hog, local_binary_pattern
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
# import matplotlib.pyplot as plt
# from tqdm import tqdm
# import joblib
# import torch

# # ============================================================
# # CUDA CHECK
# # ============================================================
# import torch
# print(torch.cuda.is_available())  # Should print True
# print(torch.cuda.device_count())  # Should be > 0
# print(torch.cuda.get_device_name(0))  # Print GPU name

# # Try importing RAPIDS cuML (GPU SVM)
# use_gpu_svm = False
# try:
#     from cuml.svm import SVC
#     from cuml.preprocessing import StandardScaler
#     use_gpu_svm = True
#     print("⚡ Using GPU-accelerated cuML SVM and StandardScaler.")
# except ImportError:
#     from sklearn.svm import SVC
#     from sklearn.preprocessing import StandardScaler
#     print("💡 cuML not found — falling back to CPU-based scikit-learn SVM.")

# # ============================================================
# # CONFIGURATION
# # ============================================================
# VIDEO_DIR = r"C:\Users\UIET\Desktop\OPTICAL\forgery_kaggle\Training"
# MODEL_PATH = "svm_combined_hog_lbp_opticalflow_optimized.pkl"

# N_FRAMES = 20             # number of frames per video for HOG+LBP
# LBP_POINTS = 8
# LBP_RADIUS = 1

# # Optical flow histogram config
# MAG_BINS = 8
# ANG_BINS = 8
# MAG_RANGE = (0, 10)
# ANG_RANGE = (0, 2 * np.pi)

# # ============================================================
# # OPTICAL FLOW FEATURE EXTRACTION (GPU-enabled)
# # ============================================================

# def extract_optical_flow_features(video_path, mag_bins=MAG_BINS, ang_bins=ANG_BINS,
#                                   mag_range=MAG_RANGE, ang_range=ANG_RANGE):
#     """
#     Extract optical flow mean and std histogram features from a video.
#     GPU accelerated if OpenCV CUDA is available.
#     Returns a 32-D vector (mean + std of magnitude & angle histograms).
#     """
#     cap = cv2.VideoCapture(video_path)
#     ret, prev_frame = cap.read()
#     if not ret:
#         cap.release()
#         return np.zeros(mag_bins + ang_bins)  # fallback zero features

#     prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
#     features = []

#     # Check for CUDA optical flow support
#     use_gpu = False
#     if cv2.cuda.getCudaEnabledDeviceCount() > 0:
#         try:
#             gpu_flow = cv2.cuda_FarnebackOpticalFlow.create()
#             use_gpu = True
#             print(f"⚡ Using GPU optical flow for: {os.path.basename(video_path)}")
#         except Exception:
#             use_gpu = False

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#         if use_gpu:
#             gpu_prev = cv2.cuda_GpuMat()
#             gpu_next = cv2.cuda_GpuMat()
#             gpu_prev.upload(prev_gray)
#             gpu_next.upload(gray)

#             flow_gpu = gpu_flow.calc(gpu_prev, gpu_next, None)
#             flow = flow_gpu.download()
#         else:
#             flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None,
#                                                 0.5, 3, 15, 3, 5, 1.2, 0)

#         mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

#         mag_hist = cv2.calcHist([mag.astype('float32')], [0], None,
#                                 [mag_bins], [mag_range[0], mag_range[1]])
#         ang_hist = cv2.calcHist([ang.astype('float32')], [0], None,
#                                 [ang_bins], [ang_range[0], ang_range[1]])

#         mag_hist = cv2.normalize(mag_hist, mag_hist).flatten()
#         ang_hist = cv2.normalize(ang_hist, ang_hist).flatten()

#         features.append(np.concatenate((mag_hist, ang_hist)))
#         prev_gray = gray

#     cap.release()
#     if len(features) == 0:
#         return np.zeros(mag_bins + ang_bins)

#     feats = np.vstack(features)
#     mean_feat = np.mean(feats, axis=0)
#     std_feat = np.std(feats, axis=0)
#     combined_feat = np.concatenate([mean_feat, std_feat])
#     return combined_feat  # 32-D vector

# # ============================================================
# # HOG + LBP FEATURE EXTRACTION
# # ============================================================

# def extract_hog_features(frame):
#     """Extract HOG descriptor from a frame."""
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     features = hog(gray, pixels_per_cell=(16, 16),
#                    cells_per_block=(2, 2),
#                    orientations=9, block_norm='L2-Hys')
#     return features


# def extract_lbp_features(frame):
#     """Extract normalized LBP histogram."""
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     lbp = local_binary_pattern(gray, P=LBP_POINTS, R=LBP_RADIUS, method="uniform")
#     hist, _ = np.histogram(lbp.ravel(),
#                            bins=np.arange(0, LBP_POINTS + 3),
#                            range=(0, LBP_POINTS + 2))
#     hist = hist.astype("float")
#     hist /= (hist.sum() + 1e-6)
#     return hist


# def extract_video_features(video_path, n_frames=N_FRAMES):
#     """Extract combined HOG + LBP + Optical Flow features for one video."""
#     cap = cv2.VideoCapture(video_path)
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     if total_frames <= 0:
#         cap.release()
#         return np.zeros(200)

#     frame_indices = np.linspace(0, total_frames - 1, n_frames, dtype=int)

#     hog_features = []
#     lbp_features = []

#     for idx in frame_indices:
#         cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
#         ret, frame = cap.read()
#         if not ret:
#             continue
#         hog_features.append(extract_hog_features(frame))
#         lbp_features.append(extract_lbp_features(frame))

#     cap.release()

#     # compute averages for HOG + LBP
#     if len(hog_features) == 0:
#         hog_mean = np.zeros(100)
#     else:
#         hog_mean = np.mean(hog_features, axis=0)

#     if len(lbp_features) == 0:
#         lbp_mean = np.zeros(LBP_POINTS + 2)
#     else:
#         lbp_mean = np.mean(lbp_features, axis=0)

#     # Optical flow features (32D)
#     flow_feat = extract_optical_flow_features(video_path)

#     combined = np.concatenate([hog_mean, lbp_mean, flow_feat])
#     return combined

# # ============================================================
# # DATASET BUILDING
# # ============================================================

# def build_dataset(video_root):
#     """Iterate over class folders and extract features."""
#     X, y = [], []
#     classes = [c for c in sorted(os.listdir(video_root)) if not c.startswith('.')]
#     print(f"Detected classes: {classes}")

#     for label, class_name in enumerate(classes):
#         class_dir = os.path.join(video_root, class_name)
#         if not os.path.isdir(class_dir):
#             continue

#         count = 0
#         for root, _, files in os.walk(class_dir):
#             videos = [f for f in files if f.endswith((".mp4", ".avi", ".mov", ".mkv"))]
#             for vid in tqdm(videos, desc=f"Processing {class_name}"):
#                 video_path = os.path.join(root, vid)
#                 features = extract_video_features(video_path)
#                 X.append(features)
#                 y.append(label)
#                 count += 1
#         print(f"✅ Processed {count} videos from '{class_name}'")

#     X = np.array(X)
#     y = np.array(y)
#     return X, y, classes

# # ============================================================
# # TRAINING AND EVALUATION (GPU-READY)
# # ============================================================

# def train_and_evaluate(X, y, classes):
#     """Train an optimized SVM model and evaluate."""
#     print("\n🚀 Starting training and evaluation...")

#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)

#     X_train, X_test, y_train, y_test = train_test_split(
#         X_scaled, y, test_size=0.2, random_state=42, stratify=y)

#     if use_gpu_svm:
#         # cuML SVM doesn't support GridSearchCV directly
#         print("⚡ Training GPU SVM (cuML)...")
#         model = SVC(kernel='rbf', C=10, gamma='scale', probability=True)
#         model.fit(X_train, y_train)
#         y_pred = model.predict(X_test)
#     else:
#         # CPU-based SVM with GridSearchCV
#         print("⚙️ Running GridSearchCV (CPU SVM)...")
#         param_grid = {
#             'C': [1, 10, 100],
#             'gamma': ['scale', 0.01, 0.001],
#             'kernel': ['rbf']
#         }
#         grid = GridSearchCV(
#             SVC(class_weight='balanced', probability=True),
#             param_grid,
#             refit=True,
#             verbose=1,
#             cv=3,
#             n_jobs=-1
#         )
#         grid.fit(X_train, y_train)
#         model = grid.best_estimator_
#         print("\n✅ Best Parameters:", grid.best_params_)
#         y_pred = model.predict(X_test)

#     acc = accuracy_score(y_test, y_pred)
#     print(f"\n🎯 Accuracy: {acc * 100:.2f}%")
#     print("\n📊 Classification Report:")
#     print(classification_report(y_test, y_pred, target_names=classes))

#     cm = confusion_matrix(y_test, y_pred)
#     disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
#     disp.plot(cmap='Blues')
#     plt.title("Confusion Matrix")
#     plt.show()

#     return model, scaler

# # ============================================================
# # MAIN
# # ============================================================

# def main():
#     print("🔍 Building dataset...")
#     X, y, classes = build_dataset(VIDEO_DIR)
#     if len(X) == 0:
#         print("⚠️ No videos found! Please check dataset path.")
#         return

#     print(f"✅ Feature matrix shape: {X.shape}")
#     model, scaler = train_and_evaluate(X, y, classes)

#     # Save everything
#     output = {'model': model, 'scaler': scaler, 'classes': classes}
#     joblib.dump(output, MODEL_PATH)
#     print(f"\n💾 Model saved to {MODEL_PATH}")


# if __name__ == "__main__":
#     main()









# Combined all without mean+std
import os
import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from tqdm import tqdm
import joblib
import torch

# ============================================================
# CUDA CHECK
# ============================================================
print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA Device Count:", torch.cuda.device_count())
    print("CUDA Device Name:", torch.cuda.get_device_name(0))

# Try importing RAPIDS cuML (GPU SVM)
use_gpu_svm = False
try:
    from cuml.svm import SVC
    from cuml.preprocessing import StandardScaler
    use_gpu_svm = True
    print("⚡ Using GPU-accelerated cuML SVM and StandardScaler.")
except ImportError:
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    print("💡 cuML not found — using CPU-based scikit-learn SVM.")

# ============================================================
# CONFIGURATION
# ============================================================
VIDEO_DIR = r"C:\Users\UIET\Desktop\OPTICAL\augmented_data\augmented_data"
MODEL_PATH = "svm_combined_hog_lbp_opticalflow_mean_augmented.pkl"

N_FRAMES = 20
LBP_POINTS = 8
LBP_RADIUS = 1

# Optical flow histogram config
MAG_BINS = 8
ANG_BINS = 8
MAG_RANGE = (0, 10)
ANG_RANGE = (0, 2 * np.pi)

# ============================================================
# OPTICAL FLOW FEATURE EXTRACTION (using Code 1 logic)
# ============================================================

def extract_optical_flow_features(video_path, mag_bins=MAG_BINS, ang_bins=ANG_BINS,
                                  mag_range=MAG_RANGE, ang_range=ANG_RANGE):
    """
    Extract optical flow features (histograms of magnitude and angle)
    between consecutive frames in a video.
    Returns a mean histogram vector (mag_bins + ang_bins dimensions).
    """
    cap = cv2.VideoCapture(video_path)
    ret, prev_frame = cap.read()
    if not ret:
        print(f"Cannot open video {video_path}")
        cap.release()
        return np.zeros(mag_bins + ang_bins)

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    features = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray,
                                            None,
                                            0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        mag_hist = cv2.calcHist([mag.astype('float32')], [0], None,
                                 [mag_bins], [mag_range[0], mag_range[1]])
        ang_hist = cv2.calcHist([ang.astype('float32')], [0], None,
                                 [ang_bins], [ang_range[0], ang_range[1]])

        mag_hist = cv2.normalize(mag_hist, mag_hist).flatten()
        ang_hist = cv2.normalize(ang_hist, ang_hist).flatten()

        feat = np.concatenate((mag_hist, ang_hist))
        features.append(feat)

        prev_gray = gray

    cap.release()
    if len(features) == 0:
        return np.zeros(mag_bins + ang_bins)

    # Use mean histogram only (no std)
    mean_feat = np.mean(features, axis=0)
    return mean_feat  # 16-D vector

# ============================================================
# HOG + LBP FEATURE EXTRACTION
# ============================================================

def extract_hog_features(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    features = hog(gray, pixels_per_cell=(16, 16),
                   cells_per_block=(2, 2),
                   orientations=9, block_norm='L2-Hys')
    return features

def extract_lbp_features(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, P=LBP_POINTS, R=LBP_RADIUS, method="uniform")
    hist, _ = np.histogram(lbp.ravel(),
                           bins=np.arange(0, LBP_POINTS + 3),
                           range=(0, LBP_POINTS + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    return hist

# ============================================================
# COMBINED VIDEO FEATURE EXTRACTION
# ============================================================

def extract_video_features(video_path, n_frames=N_FRAMES):
    """Extract combined HOG + LBP + Optical Flow features for one video."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return np.zeros(200)

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

    # Compute averages
    hog_mean = np.mean(hog_features, axis=0) if len(hog_features) else np.zeros(100)
    lbp_mean = np.mean(lbp_features, axis=0) if len(lbp_features) else np.zeros(LBP_POINTS + 2)

    # Optical flow (mean-only 16D)
    flow_feat = extract_optical_flow_features(video_path)

    # Combine all features
    combined = np.concatenate([hog_mean, lbp_mean, flow_feat])
    return combined

# ============================================================
# DATASET BUILDING
# ============================================================

def build_dataset(video_root):
    X, y = [], []
    classes = [c for c in sorted(os.listdir(video_root)) if not c.startswith('.')]
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
                X.append(features)
                y.append(label)
                count += 1
        print(f"✅ Processed {count} videos from '{class_name}'")

    X = np.array(X)
    y = np.array(y)
    return X, y, classes

# ============================================================
# TRAINING AND EVALUATION
# ============================================================

def train_and_evaluate(X, y, classes):
    print("\n🚀 Starting training and evaluation...")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y)

    if use_gpu_svm:
        print("⚡ Training GPU SVM (cuML)...")
        model = SVC(kernel='rbf', C=10, gamma='scale', probability=True)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    else:
        print("⚙️ Running GridSearchCV (CPU SVM)...")
        param_grid = {'C': [1, 10, 100],
                      'gamma': ['scale', 0.01, 0.001],
                      'kernel': ['rbf']}
        grid = GridSearchCV(
            SVC(class_weight='balanced', probability=True),
            param_grid, refit=True, verbose=1, cv=3, n_jobs=-1)
        grid.fit(X_train, y_train)
        model = grid.best_estimator_
        print("\n✅ Best Parameters:", grid.best_params_)
        y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"\n🎯 Accuracy: {acc * 100:.2f}%")
    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=classes))

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap='Blues')
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
        print("⚠️ No videos found! Please check dataset path.")
        return

    print(f"✅ Feature matrix shape: {X.shape}")
    model, scaler = train_and_evaluate(X, y, classes)

    # Save everything
    output = {'model': model, 'scaler': scaler, 'classes': classes}
    joblib.dump(output, MODEL_PATH)
    print(f"\n💾 Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()
