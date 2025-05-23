# -*- coding: utf-8 -*-
"""
Created on Tue May  6 18:01:18 2025

@author: XIE
"""

# Part 1 数据的初步处理和分类，并储存.pkl文件方便之后的读取
"""
import os
import pandas as pd
import pyarrow.parquet as pq
import joblib

base_path = "D:/My File/SMU/QF632/Final Work/train.parquet"

data_partitions = {
    "train": [0, 1, 2, 3, 4],
    "val": [5],
    "test": [6, 7],
    "future": [8],
    "future_eval": [9],
}


def load_partition(partition_ids):
    dfs = []
    for pid in partition_ids:
        folder = os.path.join(base_path, f"partition_id={pid}")
        for file in os.listdir(folder):
            if file.endswith(".parquet"):
                file_path = os.path.join(folder, file)
                pq_file = pq.ParquetFile(file_path)
                for i in range(pq_file.num_row_groups):
                    df = pq_file.read_row_group(i).to_pandas()
                    df["partition_id"] = pid
                    dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

df_train = load_partition(data_partitions["train"])
df_val = load_partition(data_partitions["val"])
df_test = load_partition(data_partitions["test"])
df_future = load_partition(data_partitions["future"])
df_future_eval = load_partition(data_partitions["future_eval"])

feature_cols = [col for col in df_train.columns if col.startswith("feature_")]

def convert_features(df):
    return df[feature_cols].astype("float32")

X_train_raw = convert_features(df_train)
y_train_raw = (df_train["responder_6"] > 0).astype(int)

X_val_raw = convert_features(df_val)
y_val_raw = (df_val["responder_6"] > 0).astype(int)

X_test_raw = convert_features(df_test)
y_test_raw = (df_test["responder_6"] > 0).astype(int)

df_future_clean = df_future.drop(columns=[col for col in df_future.columns if col.startswith("responder_")])
X_future_raw = convert_features(df_future_clean)
df_future_eval_clean = df_future_eval[["responder_6", "symbol_id", "time_id"]].copy()

joblib.dump((X_train_raw, y_train_raw), "Xy_train_raw.pkl")
joblib.dump((X_val_raw, y_val_raw), "Xy_val_raw.pkl")
joblib.dump((X_test_raw, y_test_raw), "Xy_test_raw.pkl")
joblib.dump(X_future_raw, "X_future_raw.pkl") 
joblib.dump(df_future_eval_clean, "df_future_eval_raw.pkl") 
"""
#  数据已清洗并保存为 .pkl 文件，此段已不再运行，仅供参考


# Part2 对于LR模型的更详细的清洗
# 再次清洗数据，着重在清理一些不必要的特征值，全部完成后储存新的.pkl （只需要看训练集）
"""
import joblib

X_train, y_train = joblib.load("Xy_train_raw.pkl")

nan_ratio = X_train.isna().mean().sort_values(ascending=False)

nan_ratio_top20 = nan_ratio.head(20)

print("The features with the highest missing ratios are as follows (top 20):")
for i, (feature, ratio) in enumerate(nan_ratio_top20.items(), 1):
    print(f"{i:>2}. {feature:<12} - Missing ratio: {ratio:.2%}")
"""
# 缺失比例最高的特征如下（前 20）:
# 1. feature_21   - 缺失比例: 48.94%
# 2. feature_31   - 缺失比例: 48.94%
# 3. feature_27   - 缺失比例: 48.94%
# 4. feature_26   - 缺失比例: 48.94%
# 5. feature_00   - 缺失比例: 18.91%
# 6. feature_02   - 缺失比例: 18.91%
# 7. feature_03   - 缺失比例: 18.91%
# 8. feature_04   - 缺失比例: 18.91%
# 9. feature_01   - 缺失比例: 18.91%
# 10. feature_42   - 缺失比例: 12.91%
# 11. feature_39   - 缺失比例: 12.91%
# 12. feature_50   - 缺失比例: 12.63%
# 13. feature_53   - 缺失比例: 12.63%
# 14. feature_41   - 缺失比例: 3.15%
# 15. feature_44   - 缺失比例: 3.15%
# 16. feature_52   - 缺失比例: 2.86%
# 17. feature_55   - 缺失比例: 2.86%
# 18. feature_15   - 缺失比例: 2.72%
# 19. feature_65   - 缺失比例: 1.86%
# 20. feature_46   - 缺失比例: 1.86%

# 21，31，27，26 缺失过多直接删除。
# 我们保存新的删去这四个特征值后的pkl，之后的代码会使用这个新的pkl。
"""
import joblib

X_train, y_train = joblib.load("Xy_train_raw.pkl")
X_val, y_val = joblib.load("Xy_val_raw.pkl")
X_test, y_test = joblib.load("Xy_test_raw.pkl")
X_future = joblib.load("X_future_raw.pkl") 
df_future_eval = joblib.load("df_future_eval_raw.pkl")

cols_to_drop = ["feature_21", "feature_31", "feature_27", "feature_26"]

X_train_clean = X_train.drop(columns=cols_to_drop)
X_val_clean = X_val.drop(columns=cols_to_drop)
X_test_clean = X_test.drop(columns=cols_to_drop)
X_future_clean = X_future.drop(columns=cols_to_drop)

joblib.dump((X_train_clean, y_train), "Xy_train_clean.pkl")
joblib.dump((X_val_clean, y_val), "Xy_val_clean.pkl")
joblib.dump((X_test_clean, y_test), "Xy_test_clean.pkl")
joblib.dump(X_future_clean, "X_future_clean.pkl")
joblib.dump(df_future_eval, "df_future_eval_clean.pkl") 
"""

# 填补缺失值 + L1正则 Logistic Regression, 这一步中删去的特征值只代表LR模型。
"""
import joblib
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

X_train_clean, y_train = joblib.load("Xy_train_clean.pkl")
X_val_clean, y_val = joblib.load("Xy_val_clean.pkl")

# 均值填补 NaN
imputer = SimpleImputer(strategy="mean")
X_train_imp = imputer.fit_transform(X_train_clean)
X_val_imp = imputer.transform(X_val_clean)

# L1 正则 Logistic Regression 训练（强惩罚，筛特征）
lr_l1 = LogisticRegression(penalty='l1', solver='saga', C=0.01, max_iter=1000)
lr_l1.fit(X_train_imp, y_train)

coef = lr_l1.coef_[0]
selected_mask = coef != 0
feature_names = list(X_train_clean.columns)
selected_features = [f for f, keep in zip(feature_names, selected_mask) if keep]

print(f"The number of features retained after L1 regularization: {len(selected_features)}")
for i, f in enumerate(selected_features, 1):
    print(f"{i:>2}: {f}")
"""

# L1 正则后保留的特征数量（The number of features retained after L1 regularization）: 75
# 在c=0.01 的情况下依旧没有排除任何特征值，那我们接下来就直接使用当前数据来建模做预测。
# 多做一步，我们需要用均值填补 NaN，并且保存新的pkl来建模（LR）
"""
import joblib
from sklearn.impute import SimpleImputer

X_train_clean, y_train = joblib.load("Xy_train_clean.pkl")
X_val_clean, y_val = joblib.load("Xy_val_clean.pkl")
X_test_clean, y_test = joblib.load("Xy_test_clean.pkl")
X_future_clean = joblib.load("X_future_clean.pkl")
df_future_eval_clean = joblib.load("df_future_eval_clean.pkl")

# 均值填补 NaN
imputer = SimpleImputer(strategy="mean")
X_train_ready = imputer.fit_transform(X_train_clean)
X_val_ready = imputer.transform(X_val_clean)
X_test_ready = imputer.transform(X_test_clean)
X_future_ready = imputer.transform(X_future_clean)

joblib.dump((X_train_ready, y_train), "Xy_train_ready.pkl")
joblib.dump((X_val_ready, y_val), "Xy_val_ready.pkl")
joblib.dump((X_test_ready, y_test), "Xy_test_ready.pkl")
joblib.dump(X_future_ready, "X_future_ready.pkl")  
joblib.dump(df_future_eval_clean, "df_future_eval_ready.pkl")  
"""
# 现在，我们确定下来了LR模型的最终数据，并保存为.pkl

# Part 3 LR建模
"""
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns

X_train, y_train = joblib.load("Xy_train_ready.pkl")
X_val, y_val = joblib.load("Xy_val_ready.pkl")
X_test, y_test = joblib.load("Xy_test_ready.pkl")
X_future = joblib.load("X_future_ready.pkl")
df_future_eval = joblib.load("df_future_eval_ready.pkl")  

# 训练 Logistic Regression 模型（标准 L2 正则）
lr = LogisticRegression(solver='lbfgs', max_iter=1000)
lr.fit(X_train, y_train)

# ---------- VALIDATION ----------
y_val_pred = lr.predict(X_val)
y_val_proba = lr.predict_proba(X_val)[:, 1]

print("[Val] Classification Report:")
print(classification_report(y_val, y_val_pred, digits=4))
print("[Val] AUC:", roc_auc_score(y_val, y_val_proba))

# Confusion Matrix
cm = confusion_matrix(y_val, y_val_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - Validation Set")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(y_val, y_val_proba)
plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_val, y_val_proba):.4f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Validation Set")
plt.legend()
plt.grid()
plt.show()

# PR Curve
precision, recall, _ = precision_recall_curve(y_val, y_val_proba)
plt.plot(recall, precision)
plt.title("Precision-Recall Curve - Validation Set")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.grid()
plt.show()

# ---------- TEST ----------
y_test_proba = lr.predict_proba(X_test)[:, 1]
print("[TEST] AUC:", roc_auc_score(y_test, y_test_proba))

# ---------- FUTURE ---------- 
# 我们预测的responder_6会以0，1的二分类形式表示出来，因此我们画图对比的时候把df_future_eval中的responder_6也换为0，1
y_future_pred_proba = lr.predict_proba(X_future)[:, 1]

df_plot = df_future_eval.iloc[:len(y_future_pred_proba)].copy()
df_plot["pred_proba"] = y_future_pred_proba
df_plot["label_bin"] = (df_plot["responder_6"] > 0).astype(int)

symbol_id_to_plot = 0
df_symbol = df_plot[df_plot["symbol_id"] == symbol_id_to_plot].sort_values("time_id")
df_symbol = df_symbol.drop_duplicates(subset="time_id", keep="first")  # 避免重叠

plt.figure(figsize=(150, 5))

plt.plot(df_symbol["time_id"], df_symbol["pred_proba"], label="Predicted Prob", linewidth=2, color='tab:blue')

df_up = df_symbol[df_symbol["label_bin"] == 1]
plt.scatter(df_up["time_id"], df_up["label_bin"], color='green', s=5, alpha=0.6, label="responder_6 > 0")

df_down = df_symbol[df_symbol["label_bin"] == 0]
plt.scatter(df_down["time_id"], df_down["label_bin"], color='red', s=5, alpha=0.4, label="responder_6 ≤ 0")

plt.title(" Logistic Regression Prediction vs Actual")
plt.xlabel("time_id")
plt.ylabel("Probability / Direction")
plt.ylim(-0.1, 1.1)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
"""

# Part 4 XGBoots数据清理+建模
# 使用之前的clean的.pkl数据作为初始数据
"""
import joblib
import numpy as np
from xgboost import XGBClassifier

X_train_clean, y_train = joblib.load("Xy_train_clean.pkl")

model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train_clean, y_train)

importances = model.feature_importances_
feature_names = X_train_clean.columns

sorted_idx = np.argsort(importances)[::-1]
top_features = [(feature_names[i], importances[i]) for i in sorted_idx if importances[i] > 0]

print("XGBoost features'importances:")
for i, (name, score) in enumerate(top_features[:75], 1):
    print(f"{i:>2}. {name:<12} - importance: {score:.4f}")
"""
# 我们发现XGBoots中，这所有的75个特征值都是可以保留的(最尾端的两个也是视情况保留，由于模型适应性强，在此我们决定保留)
# 因此可以使用ready.pkl的数据
# XGBoots Full Features(GPU)
"""
import joblib
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score

X_train, y_train = joblib.load("Xy_train_ready.pkl")
X_val, y_val = joblib.load("Xy_val_ready.pkl")
X_test, y_test = joblib.load("Xy_test_ready.pkl")
X_future = joblib.load("X_future_ready.pkl")
df_future_eval = joblib.load("df_future_eval_ready.pkl")  

xgb = XGBClassifier(
    tree_method="gpu_hist", 
    predictor="gpu_predictor",
    use_label_encoder=False,
    eval_metric="logloss"
)
xgb.fit(X_train, y_train)

joblib.dump(xgb, "xgb_model_gpu.pkl")

# ---------- VALIDATION ----------
y_val_proba = xgb.predict_proba(X_val)[:, 1]
y_val_pred = (y_val_proba >= 0.5).astype(int)

print("[Val] Classification Report:")
print(classification_report(y_val, y_val_pred, digits=4))
print("[Val] AUC:", roc_auc_score(y_val, y_val_proba))

# ---------- TEST ----------
y_test_proba = xgb.predict_proba(X_test)[:, 1]
print("[TEST] AUC:", roc_auc_score(y_test, y_test_proba))

# ---------- FUTURE ----------
y_future_pred_proba = xgb.predict_proba(X_future)[:, 1]
df_plot = df_future_eval.iloc[:len(y_future_pred_proba)].copy()
df_plot["pred_proba"] = y_future_pred_proba
df_plot["label_bin"] = (df_plot["responder_6"] > 0).astype(int)

symbol_id_to_plot = 0
df_symbol = df_plot[df_plot["symbol_id"] == symbol_id_to_plot].sort_values("time_id")
df_symbol = df_symbol.drop_duplicates(subset="time_id", keep="first")

plt.figure(figsize=(150, 5))

plt.plot(df_symbol["time_id"], df_symbol["pred_proba"], label="Predicted Prob", linewidth=2, color='tab:blue')

df_up = df_symbol[df_symbol["label_bin"] == 1]
plt.scatter(df_up["time_id"], df_up["label_bin"], color='green', s=5, alpha=0.6, label="responder_6 > 0")

df_down = df_symbol[df_symbol["label_bin"] == 0]
plt.scatter(df_down["time_id"], df_down["label_bin"], color='red', s=5, alpha=0.4, label="responder_6 ≤ 0")

plt.title(" XGBoots Prediction vs Actual(Full Features)")
plt.xlabel("time_id")
plt.ylabel("Probability / Direction")
plt.ylim(-0.1, 1.1)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

"""
# [Val] Classification Report:
#              precision    recall  f1-score   support

#           0     0.5783    0.6479    0.6111   2856415
#           1     0.5318    0.4584    0.4924   2491785

#    accuracy                         0.5596   5348200
#   macro avg     0.5550    0.5531    0.5517   5348200
# weighted avg     0.5566    0.5596    0.5558   5348200

# [Val] AUC: 0.5760809452025897
# [TEST] AUC: 0.561606945142231


# 有一定提升，接下来尝试减少特征值，看看会有什么变化
# 保留重要性 ≥ 0.006 的特征 前65

# XGBoots 65 Features(GPU)
"""
import joblib

X_train_clean, y_train = joblib.load("Xy_train_clean.pkl")
X_val_clean, y_val = joblib.load("Xy_val_clean.pkl")
X_test_clean, y_test = joblib.load("Xy_test_clean.pkl")
X_future_clean = joblib.load("X_future_clean.pkl")
df_future_eval_clean = joblib.load("df_future_eval_clean.pkl")

selected_features = [
    'feature_06', 'feature_60', 'feature_59', 'feature_07', 'feature_54',
    'feature_51', 'feature_68', 'feature_04', 'feature_30', 'feature_55',
    'feature_75', 'feature_52', 'feature_15', 'feature_71', 'feature_69',
    'feature_14', 'feature_50', 'feature_01', 'feature_02', 'feature_53',
    'feature_56', 'feature_58', 'feature_48', 'feature_36', 'feature_10',
    'feature_23', 'feature_76', 'feature_19', 'feature_18', 'feature_29',
    'feature_05', 'feature_17', 'feature_47', 'feature_66', 'feature_09',
    'feature_00', 'feature_72', 'feature_22', 'feature_11', 'feature_20',
    'feature_73', 'feature_61', 'feature_28', 'feature_08', 'feature_40',
    'feature_13', 'feature_44', 'feature_24', 'feature_65', 'feature_67',
    'feature_37', 'feature_03', 'feature_49', 'feature_38', 'feature_25',
    'feature_39', 'feature_78', 'feature_70', 'feature_46', 'feature_12',
    'feature_41', 'feature_62', 'feature_57', 'feature_42', 'feature_45'
]

joblib.dump((X_train_clean[selected_features], y_train), "Xy_train_65.pkl")
joblib.dump((X_val_clean[selected_features], y_val), "Xy_val_65.pkl")
joblib.dump((X_test_clean[selected_features], y_test), "Xy_test_65.pkl")
joblib.dump(X_future_clean[selected_features], "X_future_65.pkl")
joblib.dump(df_future_eval_clean, "df_future_eval_65.pkl")

import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score

X_train, y_train = joblib.load("Xy_train_65.pkl")
X_val, y_val = joblib.load("Xy_val_65.pkl")
X_test, y_test = joblib.load("Xy_test_65.pkl")
X_future = joblib.load("X_future_65.pkl")
df_future_eval = joblib.load("df_future_eval_65.pkl")

xgb_65 = XGBClassifier(
    tree_method="gpu_hist", 
    predictor="gpu_predictor",
    use_label_encoder=False,
    eval_metric="logloss"
)
xgb_65.fit(X_train, y_train)

joblib.dump(xgb_65, "xgb_model_65.pkl")

# ---------- VALIDATION ----------
y_val_proba = xgb_65.predict_proba(X_val)[:, 1]
y_val_pred = (y_val_proba >= 0.5).astype(int)

print("[Val] Classification Report:")
print(classification_report(y_val, y_val_pred, digits=4))
print("[Val] AUC:", roc_auc_score(y_val, y_val_proba))

# ---------- TEST ----------
y_test_proba = xgb_65.predict_proba(X_test)[:, 1]
print("[TEST] AUC:", roc_auc_score(y_test, y_test_proba))

# ---------- FUTURE ----------
y_future_pred_proba = xgb_65.predict_proba(X_future)[:, 1]
df_plot = df_future_eval.iloc[:len(y_future_pred_proba)].copy()
df_plot["pred_proba"] = y_future_pred_proba
df_plot["label_bin"] = (df_plot["responder_6"] > 0).astype(int)

symbol_id_to_plot = 0
df_symbol = df_plot[df_plot["symbol_id"] == symbol_id_to_plot].sort_values("time_id")
df_symbol = df_symbol.drop_duplicates(subset="time_id", keep="first")

plt.figure(figsize=(150, 5))

plt.plot(df_symbol["time_id"], df_symbol["pred_proba"], label="Predicted Prob", linewidth=2, color='tab:blue')

df_up = df_symbol[df_symbol["label_bin"] == 1]
plt.scatter(df_up["time_id"], df_up["label_bin"], color='green', s=5, alpha=0.6, label="responder_6 > 0")

df_down = df_symbol[df_symbol["label_bin"] == 0]
plt.scatter(df_down["time_id"], df_down["label_bin"], color='red', s=5, alpha=0.4, label="responder_6 ≤ 0")

plt.title(" XGBoots Prediction vs Actual(65 Features)")
plt.xlabel("time_id")
plt.ylabel("Probability / Direction")
plt.ylim(-0.1, 1.1)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

"""
# 提升很小很小，我称之为没啥提升
#[Val] Classification Report:
#             precision    recall  f1-score   support

#           0     0.5785    0.6472    0.6109   2856415
#           1     0.5318    0.4594    0.4930   2491785

#    accuracy                         0.5597   5348200
#   macro avg     0.5551    0.5533    0.5519   5348200
# weighted avg     0.5567    0.5597    0.5559   5348200

# [Val] AUC: 0.5761781268468457
# [TEST] AUC: 0.5624955605090792

# 尝试一下保留大于0.01的（强烈保留）前28的
# XGBoots 28 Features(GPU)
import joblib

X_train_clean, y_train = joblib.load("Xy_train_clean.pkl")
X_val_clean, y_val = joblib.load("Xy_val_clean.pkl")
X_test_clean, y_test = joblib.load("Xy_test_clean.pkl")
X_future_clean = joblib.load("X_future_clean.pkl")
df_future_eval_clean = joblib.load("df_future_eval_clean.pkl")

selected_features_28 = [
    'feature_06', 'feature_60', 'feature_59', 'feature_07', 'feature_54',
    'feature_51', 'feature_68', 'feature_04', 'feature_30', 'feature_55',
    'feature_75', 'feature_52', 'feature_15', 'feature_71', 'feature_69',
    'feature_14', 'feature_50', 'feature_01', 'feature_02', 'feature_53',
    'feature_56', 'feature_58', 'feature_48', 'feature_36', 'feature_10',
    'feature_23', 'feature_76', 'feature_19'
]


X_train_sel = X_train_clean[selected_features_28]
X_val_sel = X_val_clean[selected_features_28]
X_test_sel = X_test_clean[selected_features_28]
X_future_sel = X_future_clean[selected_features_28]

joblib.dump((X_train_sel, y_train), "Xy_train_28.pkl")
joblib.dump((X_val_sel, y_val), "Xy_val_28.pkl")
joblib.dump((X_test_sel, y_test), "Xy_test_28.pkl")
joblib.dump(X_future_sel, "X_future_28.pkl")
joblib.dump(df_future_eval_clean, "df_future_eval_28.pkl")

import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score

X_train, y_train = joblib.load("Xy_train_28.pkl")
X_val, y_val = joblib.load("Xy_val_28.pkl")
X_test, y_test = joblib.load("Xy_test_28.pkl")
X_future = joblib.load("X_future_28.pkl")
df_future_eval = joblib.load("df_future_eval_28.pkl")

xgb_28 = XGBClassifier(
    tree_method="gpu_hist", 
    predictor="gpu_predictor",
    use_label_encoder=False,
    eval_metric="logloss"
)
xgb_28.fit(X_train_sel, y_train)


joblib.dump(xgb_28, "xgb_model_28.pkl")

# ---------- VALIDATION ----------
y_val_proba = xgb_28.predict_proba(X_val_sel)[:, 1]
y_val_pred = (y_val_proba >= 0.5).astype(int)
print("[Val] Classification Report:")
print(classification_report(y_val, y_val_pred, digits=4))
print("[Val] AUC:", roc_auc_score(y_val, y_val_proba))

# ---------- TEST ----------
y_test_proba = xgb_28.predict_proba(X_test_sel)[:, 1]
print("[TEST] AUC:", roc_auc_score(y_test, y_test_proba))

# ---------- FUTURE ----------
y_future_proba = xgb_28.predict_proba(X_future_sel)[:, 1]
df_plot = df_future_eval_clean.iloc[:len(y_future_proba)].copy()
df_plot["pred_proba"] = y_future_proba
df_plot["label_bin"] = (df_plot["responder_6"] > 0).astype(int)

symbol_id_to_plot = 0
df_symbol = df_plot[df_plot["symbol_id"] == symbol_id_to_plot].sort_values("time_id")
df_symbol = df_symbol.drop_duplicates(subset="time_id", keep="first")

plt.figure(figsize=(150, 5))

plt.plot(df_symbol["time_id"], df_symbol["pred_proba"], label="Predicted Prob", linewidth=2, color='tab:blue')

df_up = df_symbol[df_symbol["label_bin"] == 1]
plt.scatter(df_up["time_id"], df_up["label_bin"], color='green', s=5, alpha=0.6, label="responder_6 > 0")

df_down = df_symbol[df_symbol["label_bin"] == 0]
plt.scatter(df_down["time_id"], df_down["label_bin"], color='red', s=5, alpha=0.4, label="responder_6 ≤ 0")

plt.title(" XGBoots Prediction vs Actual(28 Features)")
plt.xlabel("time_id")
plt.ylabel("Probability / Direction")
plt.ylim(-0.1, 1.1)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# 我愿称之为没啥变化
# [Val] Classification Report:
#              precision    recall  f1-score   support

#           0     0.5779    0.6497    0.6117   2856415
#           1     0.5317    0.4560    0.4910   2491785

#    accuracy                         0.5594   5348200
#   macro avg     0.5548    0.5528    0.5513   5348200
# weighted avg     0.5564    0.5594    0.5554   5348200

# [Val] AUC: 0.5756430997223801
# [TEST] AUC: 0.5623682031790292

