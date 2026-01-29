# import os
# import cv2
# import numpy as np
# from sklearn.svm import SVC
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# from sklearn.preprocessing import StandardScaler
# import joblib
# import matplotlib.pyplot as plt
# import seaborn as sns
# from tqdm import tqdm

# import torch
# print("CUDA Available:", torch.cuda.is_available())
# print("CUDA Device Count:", torch.cuda.device_count())
# if torch.cuda.is_available():
#     print("CUDA Device Name:", torch.cuda.get_device_name(0))


# def extract_optical_flow_features(video_path, mag_bins=8, ang_bins=8,
#                                   mag_range=(0, 10), ang_range=(0, 2 * np.pi)):
#     """
#     Extract optical flow features (histograms of magnitude and angle)
#     between consecutive frames in a video.
#     Returns an array of shape (num_frame_pairs, mag_bins + ang_bins).
#     """
#     cap = cv2.VideoCapture(video_path)
#     ret, prev_frame = cap.read()
#     if not ret:
#         print(f"Cannot open video {video_path}")
#         cap.release()
#         return np.empty((0, mag_bins + ang_bins))
    
#     prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
#     features = []

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
        
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         flow = cv2.calcOpticalFlowFarneback(prev_gray, gray,
#                                             None,
#                                             0.5, 3, 15, 3, 5, 1.2, 0)
#         mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

#         mag_hist = cv2.calcHist([mag.astype('float32')], [0], None,
#                                  [mag_bins], [mag_range[0], mag_range[1]])
#         ang_hist = cv2.calcHist([ang.astype('float32')], [0], None,
#                                  [ang_bins], [ang_range[0], ang_range[1]])

#         mag_hist = cv2.normalize(mag_hist, mag_hist).flatten()
#         ang_hist = cv2.normalize(ang_hist, ang_hist).flatten()

#         feat = np.concatenate((mag_hist, ang_hist))
#         features.append(feat)

#         prev_gray = gray

#     cap.release()
#     if len(features) == 0:
#         return np.empty((0, mag_bins + ang_bins))
#     return np.vstack(features)


# def build_dataset(root_dir):
#     """
#     Walk the dataset folder and build feature and label arrays.
#     root_dir is the path to the dataset directory.
#     Returns X (features), y (labels).
#     """
#     X = []
#     y = []
#     video_paths = []

#     # Define label mapping
#     label_map = {
#         # "original": 0,  # Uncomment if original videos are present
#         "duplication": 1,
#         "insertion": 2,
#         "deletion": 3
#     }

#     print("Building dataset from:", root_dir)

#     for tamper_type, label in label_map.items():
#         tamper_dir = os.path.join(root_dir, tamper_type)
#         if not os.path.isdir(tamper_dir):
#             print(f"Warning: Folder {tamper_dir} does not exist.")
#             continue

#         for sub in os.listdir(tamper_dir):
#             subdir_path = os.path.join(tamper_dir, sub)
#             if not os.path.isdir(subdir_path):
#                 continue
#             for fname in os.listdir(subdir_path):
#                 if fname.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
#                     vpath = os.path.join(subdir_path, fname)
#                     video_paths.append((vpath, label))

#     print(f"Found {len(video_paths)} videos. Extracting features...")

#     for vpath, label in tqdm(video_paths):
#         feats = extract_optical_flow_features(vpath)
#         if feats.shape[0] == 0:
#             print(f"Skipping {vpath} (no features)")
#             continue
#         mean_feat = np.mean(feats, axis=0)
#         X.append(mean_feat)
#         y.append(label)

#     X = np.array(X)
#     y = np.array(y)

#     print("Dataset built successfully.")
#     print("X shape:", X.shape, "y shape:", y.shape)
#     return X, y


# def train_and_evaluate(X, y, test_size=0.2, random_state=42):
#     """
#     Train an SVM classifier and evaluate it.
#     Returns trained model, scaler, and test data for further analysis.
#     """
#     # Normalize features
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)

#     X_train, X_test, y_train, y_test = train_test_split(
#         X_scaled, y, test_size=test_size, random_state=random_state, stratify=y)

#     clf = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
#     clf.fit(X_train, y_train)

#     y_pred = clf.predict(X_test)

#     print("\n✅ Evaluation Results")
#     print("Accuracy:", accuracy_score(y_test, y_pred))
#     print("Classification Report:\n", classification_report(y_test, y_pred))

#     # Confusion Matrix
#     cm = confusion_matrix(y_test, y_pred)
#     plt.figure(figsize=(6, 5))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
#     plt.xlabel("Predicted")
#     plt.ylabel("True")
#     plt.title("Confusion Matrix")
#     plt.show()

#     return clf, scaler, (X_test, y_test, y_pred)


# def main():
#     # Change this to your dataset path
#     root_dir = r"C:\Users\UIET\Downloads\Dataset_original\Dataset_original"

#     X, y = build_dataset(root_dir)
#     if X.shape[0] == 0:
#         print("❌ No data to train on. Exiting.")
#         return

#     clf, scaler, results = train_and_evaluate(X, y)

#     # Save model and scaler
#     output = {
#         'model': clf,
#         'scaler': scaler
#     }
#     model_path = "svm_optical_flow_tamper_model.pkl"
#     joblib.dump(output, model_path)
#     print("\n💾 Model and scaler saved to:", model_path)


# if __name__ == "__main__":
#     main()

import os
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
print("CUDA Available:", torch.cuda.is_available())
print("CUDA Device Count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("CUDA Device Name:", torch.cuda.get_device_name(0))


def extract_optical_flow_features(video_path, mag_bins=8, ang_bins=8,
                                  mag_range=(0, 10), ang_range=(0, 2 * np.pi)):
    """
    Extract optical flow features (histograms of magnitude and angle)
    between consecutive frames in a video.
    Returns an array of shape (num_frame_pairs, mag_bins + ang_bins).
    """
    cap = cv2.VideoCapture(video_path)
    ret, prev_frame = cap.read()
    if not ret:
        print(f"Cannot open video {video_path}")
        cap.release()
        return np.empty((0, mag_bins + ang_bins))

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
        return np.empty((0, mag_bins + ang_bins))
    return np.vstack(features)


def build_dataset(root_dir):
    """
    Walk the dataset folder and collect video paths and labels.
    Returns: list of (video_path, label)
    """
    video_paths = []

    # Define label mapping
    label_map = {
        "duplication": 1,
        "insertion": 2,
        "deletion": 3
    }

    print("Building dataset from:", root_dir)

    for tamper_type, label in label_map.items():
        tamper_dir = os.path.join(root_dir, tamper_type)
        if not os.path.isdir(tamper_dir):
            print(f"Warning: Folder {tamper_dir} does not exist.")
            continue

        for sub in os.listdir(tamper_dir):
            subdir_path = os.path.join(tamper_dir, sub)
            if not os.path.isdir(subdir_path):
                continue
            for fname in os.listdir(subdir_path):
                if fname.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    vpath = os.path.join(subdir_path, fname)
                    video_paths.append((vpath, label))

    print(f"Found {len(video_paths)} videos.")
    return video_paths


def extract_dataset_features(video_paths):
    """
    Extract enhanced features (mean + std) from all video paths.
    Returns X (features), y (labels)
    """
    X = []
    y = []

    print("Extracting features (mean + std) from videos...")

    for vpath, label in tqdm(video_paths):
        feats = extract_optical_flow_features(vpath)
        if feats.shape[0] == 0:
            print(f"Skipping {vpath} (no features)")
            continue
        mean_feat = np.mean(feats, axis=0)
        std_feat = np.std(feats, axis=0)
        combined_feat = np.concatenate([mean_feat, std_feat])  # 32-dim
        X.append(combined_feat)
        y.append(label)

    X = np.array(X)
    y = np.array(y)

    print("✅ Feature extraction completed.")
    print("X shape:", X.shape, "y shape:", y.shape)
    return X, y


def train_and_evaluate(X, y, test_size=0.2, random_state=42):
    """
    Train an SVM classifier using GridSearchCV and evaluate it.
    Returns trained model, scaler, and test data for further analysis.
    """
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state, stratify=y)

    # Grid search parameters
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 0.01, 0.001, 0.0001],
        'kernel': ['rbf']
    }

    grid = GridSearchCV(
        SVC(class_weight='balanced', probability=True),
        param_grid,
        refit=True,
        verbose=1,
        cv=5,
        n_jobs=-1
    )

    print("\n🔍 Starting hyperparameter tuning with GridSearchCV...")
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    print("\n✅ Best Parameters Found:", grid.best_params_)

    # Evaluate
    y_pred = best_model.predict(X_test)

    print("\n📊 Evaluation Results")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.title("Confusion Matrix")
    plt.show()

    return best_model, scaler, (X_test, y_test, y_pred)


def main():
    # 👇 Change this to your dataset path
    root_dir = r"C:\Users\UIET\Desktop\OPTICAL\augmented_data\augmented_data"

    video_paths = build_dataset(root_dir)
    if len(video_paths) == 0:
        print("❌ No video paths found. Exiting.")
        return

    X, y = extract_dataset_features(video_paths)
    if X.shape[0] == 0:
        print("❌ No valid features extracted. Exiting.")
        return

    clf, scaler, results = train_and_evaluate(X, y)

    # Save model and scaler
    output = {
        'model': clf,
        'scaler': scaler
    }
    model_path = "svm_optical_flow_tamper_model_optimized.pkl"
    joblib.dump(output, model_path)
    print("\n💾 Model and scaler saved to:", model_path)


if __name__ == "__main__":
    main()
