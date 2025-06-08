import os
import pandas as pd
import numpy as np
import time
import seaborn as sns
from sklearn.feature_selection import RFE
import pyarrow.parquet as pq
import lightgbm as lgb
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve, auc
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import HistGradientBoostingRegressor


base_path = "D:/MQF/MQF632 Financial Data Science/final group work/archive/train.parquet"

"""
#为了方便代码的合并，在LR模型正则前使用同样的数据处理，删除四个大量缺失的特征，但是其余特征尝试不同的缺失值填补手段
#同样这段代码后续不再运行，跑完清空内存重启项目
base_path = "D:/MQF/MQF632 Financial Data Science/final group work/archive/train.parquet"

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

                parquet_file = pq.ParquetFile(file_path)
                for i in range(parquet_file.num_row_groups):
                    df = parquet_file.read_row_group(i).to_pandas()
                    df["partition_id"] = pid
                    dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


df_train = load_partition(data_partitions["train"])
df_val = load_partition(data_partitions["val"])
df_test = load_partition(data_partitions["test"])
df_future = load_partition(data_partitions["future"])
df_future_eval = load_partition(data_partitions["future_eval"])

df_future_clean = df_future.drop(columns=[col for col in df_future.columns if col.startswith("responder_")], errors="ignore")

df_future_eval_clean = df_future_eval[["responder_6", "partition_id"] +
                                      [col for col in df_future_eval.columns if "ts_id" in col or "time_id" in col or "symbol_id" in col]]

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

X_train, y_train = joblib.load("Xy_train_raw.pkl")

nan_ratio = X_train.isna().mean().sort_values(ascending=False)

nan_ratio_top20 = nan_ratio.head(20)

print("The features with the highest missing ratios are as follows (top 20):")
for i, (feature, ratio) in enumerate(nan_ratio_top20.items(), 1):
    print(f"{i:>2}. {feature:<12} - Missing ratio: {ratio:.2%}")

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

"""
# 使用MICE方法处理缺失值，但是跑得太慢了所以先取样1%跑一遍流程，再用晚上跑10%之类的更多的量试试
# 记录总开始时间
# 首先跑出来MICE的数据集备用，目前是用了20%的采样比例
total_start = time.time()

# 加载预处理数据
X_train_clean, y_train = joblib.load("Xy_train_clean.pkl")
X_val_clean, y_val = joblib.load("Xy_val_clean.pkl")
X_test_clean, y_test = joblib.load("Xy_test_clean.pkl")
X_future_clean = joblib.load("X_future_clean.pkl")
df_future_eval = joblib.load("df_future_eval_clean.pkl")


# 优化参数配置
mice_params = {
    'estimator': HistGradientBoostingRegressor(
        max_iter=50,
        max_depth=3,
        early_stopping=True,
        random_state=42
    ),
    'max_iter': 5,            
    'tol': 1e-2,              
    'n_nearest_features': 20, 
    'initial_strategy': 'mean',
    'random_state': 42,
    # 移除了n_jobs参数
}


# TrackedMICE
class TrackedMICE(IterativeImputer):
    def _setup_pbar(self, n_features):
        self._pbar = tqdm(total=self.max_iter * n_features, desc="MICE迭代进度")

    def _print_verbose_msg(self, iteration, n_imputed, n_missing):
        # tqdm 更新进度
        if hasattr(self, "_pbar") and self._pbar:
            self._pbar.update(n_imputed)

    def fit_transform(self, X, y=None):
        self._setup_pbar(X.shape[1])  
        with joblib.parallel_backend('threading', n_jobs=-1):
            result = super().fit_transform(X, y)
        self._pbar.close()
        return result


def process_mice(data, imputer):
    chunks = np.array_split(data, indices_or_sections=10)  # 分块处理
    results = []
    indices = []

    for chunk in tqdm(chunks, desc="数据分块处理"):
        result = imputer.transform(chunk)
        results.append(result)
        indices.extend(chunk.index)

    final = np.vstack(results)
    return pd.DataFrame(final, index=indices, columns=data.columns)


def run_mice_pipeline(sample_ratio=0.2):
    # 加载原始数据（保留原始索引）
    X_train_clean, y_train = joblib.load("Xy_train_clean.pkl")
    X_val_clean = joblib.load("Xy_val_clean.pkl")[0]
    X_test_clean = joblib.load("Xy_test_clean.pkl")[0]
    X_future_clean = joblib.load("X_future_clean.pkl")

    original_indices = {
        'train': X_train_clean.index.copy(),
        'val': X_val_clean.index.copy(),
        'test': X_test_clean.index.copy(),
        'future': X_future_clean.index.copy()
    }

    sample_idx = np.random.choice(X_train_clean.index,
                                  size=int(len(X_train_clean) * sample_ratio),
                                  replace=False)
    X_train_sampled = X_train_clean.loc[sample_idx]

    print(f"\n=== 优化MICE处理 ===")
    print(f"采样数据形状: {X_train_sampled.shape}")
    print(f"使用估计器: {mice_params['estimator'].__class__.__name__}")

    # 初始化带进度条的MICE
    mice_imputer = TrackedMICE(**mice_params)

    # 阶段1：在采样数据上训练（保留索引）
    start_time = time.time()

    # 转换时保留索引
    X_train_sampled_imp = pd.DataFrame(
        mice_imputer.fit_transform(X_train_sampled),
        index=X_train_sampled.index,
        columns=X_train_sampled.columns
    )

    # 阶段2：全量数据转换（带索引管理）
    datasets = {
        'train': (X_train_clean, original_indices['train']),
        'val': (X_val_clean, original_indices['val']),
        'test': (X_test_clean, original_indices['test']),
        'future': (X_future_clean, original_indices['future'])
    }

    imputed_data = {}
    for name, (data, original_idx) in datasets.items():
        print(f"\n转换数据集: {name}")

        # 分块处理并保留索引
        processed = process_mice_with_index(data, mice_imputer, original_idx)

        # 索引验证
        if not processed.index.equals(original_idx):
            missing = original_idx.difference(processed.index)
            extra = processed.index.difference(original_idx)
            raise ValueError(
                f"{name}数据集索引改变！\n"
                f"丢失索引数: {len(missing)}\n"
                f"多余索引数: {len(extra)}\n"
                f"示例丢失索引: {missing[:5] if len(missing) > 0 else '无'}\n"
                f"示例多余索引: {extra[:5] if len(extra) > 0 else '无'}"
            )

        imputed_data[name] = processed

    # 保存结果（保留索引）
    save_paths = {}
    for name, data in imputed_data.items():
        path = f"X_{name}_mice.pkl"
        data.to_pickle(path)  # 使用DataFrame原生保存方法保留索引
        save_paths[name] = path

    total_time = time.time() - start_time
    print(f"\n=== 处理完成 总耗时: {total_time // 60:.0f}分 {total_time % 60:.2f}秒 ===")
    print("保存文件:")
    for name, path in save_paths.items():
        print(f"- {name}: {path}")


# 新增分块处理函数（带索引保留）
def process_mice_with_index(data, imputer, original_index):
    # 备份原始索引和列名
    columns = data.columns
    index = data.index

    # 分块处理
    chunks = np.array_split(data.values, indices_or_sections=10)
    processed_chunks = []
    for chunk in tqdm(chunks, desc=f"处理分块"):
        processed = imputer.transform(chunk)
        processed_chunks.append(processed)

    # 重建DataFrame并验证
    full_processed = np.vstack(processed_chunks)
    df_processed = pd.DataFrame(full_processed,
                                index=index,
                                columns=columns)

    # 最终维度验证
    if df_processed.shape != data.shape:
        raise ValueError(
            f"形状改变！原始: {data.shape} 处理后: {df_processed.shape}"
        )

    return df_processed

# 执行处理流程（调整sample_ratio控制采样比例）
run_mice_pipeline(sample_ratio=0.2)


# === MICE ===
# sample shape: (3365013, 75)
# using: HistGradientBoostingRegressor
#
# dataset: train
# process chunking: 100%|██████████| 10/10 [02:42<00:00, 16.25s/it]
#
# dataset: val
# process chunking: 100%|██████████| 10/10 [00:29<00:00,  2.95s/it]
#
# dataset: test
# process chunking: 100%|██████████| 10/10 [01:11<00:00,  7.14s/it]
#
# dataset: future
# process chunking: 100%|██████████| 10/10 [00:32<00:00,  3.21s/it]
#
# === time: 28 minute 27.74 sec ===
# save:
# - train: X_train_mice.pkl
# - val: X_val_mice.pkl
# - test: X_test_mice.pkl
# - future: X_future_mice.pkl
"""
"""
# 加载预处理数据
X_train_clean, y_train = joblib.load("Xy_train_clean.pkl")
X_val_clean, y_val = joblib.load("Xy_val_clean.pkl")
X_test_clean, y_test = joblib.load("Xy_test_clean.pkl")
X_future_clean = joblib.load("X_future_clean.pkl")
df_future_eval = joblib.load("df_future_eval_clean.pkl")
# lgb模型三个版本比较
###########lgb1
# 因为lgb可以直接处理缺失值，所以先用最开始的数据单跑lgb模型作为基准模型
# 创建数据集
print("\n=== 开始模型训练 ===")
train_start = time.time()

lgb_params = {
    'use_missing': True,          # 显式处理缺失值
    'zero_as_missing': False,     # 区分0和真正的缺失
    # 基础参数
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'device': 'gpu',  # 必须启用GPU加速

    # 树结构
    'num_leaves': 511,
    'max_depth': -1,
    'min_data_in_leaf': 500,
    'extra_trees': True,

    # 正则化
    'lambda_l1': 0.05,
    'lambda_l2': 0.05,
    'feature_fraction': 0.7,
    'bagging_fraction': 0.7,

    # 学习策略
    'learning_rate': 0.02,
    'max_bin': 255
}

train_data = lgb.Dataset(X_train_clean, label=y_train)
val_data = lgb.Dataset(X_val_clean, label=y_val, reference=train_data)

# 模型训练
lgb_model = lgb.train(
    params=lgb_params,
    train_set=train_data,
    valid_sets=[val_data],
    num_boost_round=1000
)

print(f"模型训练完成，耗时: {time.time()-train_start:.2f}秒")

from sklearn.metrics import roc_curve, auc

def plot_roc_curve(y_true, y_score, title, save_path):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.4f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


print("\n=== lgb_clean ===")
def evaluate_model(model, X, y, dataset_name):
    start = time.time()
    proba = model.predict(X)
    auc = roc_auc_score(y, proba)
    print(f"{dataset_name} AUC: {auc:.4f} | time: {time.time()-start:.2f}秒")
    return proba

from sklearn.metrics import roc_auc_score

lgb_val_proba = evaluate_model(lgb_model, X_val_clean, y_val, "test")
lgb_val_pred = (lgb_val_proba >= 0.5).astype(int)
print("[Val] Classification Report:")
print(classification_report(y_val, lgb_val_pred, digits=4))
print("[Val] AUC:", roc_auc_score(y_val, lgb_val_proba))

lgb_test_proba = evaluate_model(lgb_model, X_test_clean, y_test, "test")
print("[TEST] AUC:", roc_auc_score(y_test, lgb_test_proba))

plot_roc_curve(y_val, lgb_val_proba,
               title="ROC Curve - Validation Set",
               save_path="lgb_roc_curve_clean.png")

# Confusion Matrix
cm = confusion_matrix(y_val, lgb_val_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - Validation Set lgb_clean")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# PR Curve
precision, recall, _ = precision_recall_curve(y_val, lgb_val_proba)
plt.plot(recall, precision)
plt.title("Precision-Recall Curve - Validation Set lgb_clean")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.grid()
plt.show()

# ========== Future  ========== #
print("\n===  Future  ===")

y_future_pred_proba = lgb_model.predict(X_future_clean)

df_plot = df_future_eval.iloc[:len(y_future_pred_proba)].copy()
df_plot["pred_proba"] = y_future_pred_proba
df_plot["label_bin"] = (df_plot["responder_6"] > 0).astype(int)

symbol_id_to_plot = 0
df_symbol = df_plot[df_plot["symbol_id"] == symbol_id_to_plot].sort_values("time_id")
df_symbol = df_symbol.drop_duplicates(subset="time_id")

plt.figure(figsize=(150, 5))


plt.plot(df_symbol["time_id"], df_symbol["pred_proba"],
         label="Predicted Probability (LGB)", linewidth=2, color='tab:blue')

df_up = df_symbol[df_symbol["label_bin"] == 1]
plt.scatter(df_up["time_id"], df_up["label_bin"], color='green', s=15, alpha=0.6, label="responder_6 > 0")
df_down = df_symbol[df_symbol["label_bin"] == 0]
plt.scatter(df_down["time_id"], df_down["label_bin"], color='red', s=15, alpha=0.4, label="responder_6 ≤ 0")

plt.title(f"LGB Prediction vs Actual (symbol_id={symbol_id_to_plot})")
plt.xlabel("time_id")
plt.ylabel("Probability / Direction")
plt.ylim(-0.1, 1.1)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === lgb_clean ===
# test AUC: 0.5830 | time: 100.73秒
# [Val] Classification Report:
#               precision    recall  f1-score   support
# 
#            0     0.5830    0.6446    0.6123   2856415
#            1     0.5364    0.4714    0.5018   2491785
# 
#     accuracy                         0.5639   5348200
#    macro avg     0.5597    0.5580    0.5570   5348200
# weighted avg     0.5613    0.5639    0.5608   5348200
# 
# [Val] AUC: 0.5830372749303108
# test AUC: 0.5675 | time: 237.07秒
# [TEST] AUC: 0.5674896500636926
"""

"""
###########lgb2
# 下面一步要跑进行了特征筛选版本，开始挣扎
# 使用lgb内嵌自动计算特征的重要程度，然后选择重要分数大于等于总体中位数的特征值
print("\n=== 开始特征筛选 (RFE) ===")
rfe_start = time.time()

# 创建基础模型
base_model = lgb.LGBMClassifier(**lgb_params)

# 设置RFE参数 - 修改为实际筛选
n_features_to_select = 40  # 目标特征数量调整为40
step = 0.2  # 每次迭代移除20%的特征，加速过程

# 初始化RFE
selector = RFE(
    estimator=base_model,
    n_features_to_select=n_features_to_select,
    step=step,
    verbose=1
)

# 执行RFE
selector = selector.fit(X_train_clean, y_train)

# 获取选择的特征索引
selected_features = selector.support_
selected_indices = np.where(selected_features)[0]

# 筛选特征
X_train_selected = X_train_clean.iloc[:, selected_indices]
X_val_selected = X_val_clean.iloc[:, selected_indices]
X_test_selected = X_test_clean.iloc[:, selected_indices]
X_future_selected = X_future_clean.iloc[:, selected_indices]

# ============ 新增：获取并分析特征重要性 ============
# 训练完整模型获取重要性分数
full_model = lgb.LGBMClassifier(**lgb_params).fit(X_train_clean, y_train)
importances = full_model.feature_importances_

# 创建特征重要性DataFrame
feature_importance = pd.DataFrame({
    'Feature': X_train_clean.columns,
    'Importance': importances,
    'Selected': selected_features,
    'Ranking': selector.ranking_
})

# 计算重要性统计
importance_stats = {
    'Max': np.max(importances),
    'Min': np.min(importances),
    'Mean': np.mean(importances),
    'Median': np.median(importances),
    'Std': np.std(importances)
}

median_importance = np.median(importances)
important_feature_mask = feature_importance['Importance'] >= median_importance
important_features = feature_importance[important_feature_mask]['Feature'].values

# 打印保留的特征信息
print(f"\nBased on median({median_importance:.2f})Number of features retained after filtering: {len(important_features)}")
print("Examples of reserved features:", important_features[:])

# 筛选各个数据集中的特征列
X_train_filtered = X_train_clean[important_features]
X_val_filtered = X_val_clean[important_features]
X_test_filtered = X_test_clean[important_features]
X_future_filtered = X_future_clean[important_features]

# 保存筛选后的数据
joblib.dump((X_train_filtered, y_train), "Xy_train_filtered.pkl")
joblib.dump((X_val_filtered, y_val), "Xy_val_filtered.pkl")
joblib.dump((X_test_filtered, y_test), "Xy_test_filtered.pkl")
joblib.dump(X_future_filtered, "X_future_filtered.pkl")

print("\n已保存基于特征重要性中位数筛选后的数据。")

print("\n=== 开始模型训练 ===")
train_start = time.time()

train_data = lgb.Dataset(X_train_filtered, label=y_train)
val_data = lgb.Dataset(X_val_filtered, label=y_val, reference=train_data)

# 模型训练
lgb_model = lgb.train(
    params=lgb_params,
    train_set=train_data,
    valid_sets=[val_data],
    num_boost_round=1000
)

print(f"模型训练完成，耗时: {time.time()-train_start:.2f}秒")

from sklearn.metrics import roc_curve, auc

def plot_roc_curve(y_true, y_score, title, save_path):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.4f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# 模型评估
print("\n=== lgb_filtered ===")
def evaluate_model(model, X, y, dataset_name):
    start = time.time()
    proba = model.predict(X)
    auc = roc_auc_score(y, proba)
    print(f"{dataset_name} AUC: {auc:.4f} | time: {time.time()-start:.2f}秒")
    return proba

from sklearn.metrics import roc_auc_score

lgb_val_proba = evaluate_model(lgb_model, X_val_filtered, y_val, "test")
lgb_val_pred = (lgb_val_proba >= 0.5).astype(int)
print("[Val] Classification Report:")
print(classification_report(y_val, lgb_val_pred, digits=4))
print("[Val] AUC:", roc_auc_score(y_val, lgb_val_proba))

lgb_test_proba = evaluate_model(lgb_model, X_test_filtered, y_test, "test")
print("[TEST] AUC:", roc_auc_score(y_test, lgb_test_proba))
# 验证集 ROC 曲线
plot_roc_curve(y_val, lgb_val_proba,
               title="ROC Curve - Validation Set",
               save_path="lgb_roc_curve_filtered.png")

# Confusion Matrix
cm = confusion_matrix(y_val, lgb_val_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - Validation Set lgb_filtered")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# PR Curve
precision, recall, _ = precision_recall_curve(y_val, lgb_val_proba)
plt.plot(recall, precision)
plt.title("Precision-Recall Curve - Validation Set lgb_filtered")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.grid()
plt.show()

# === lgb_filtered ===
# test AUC: 0.5766 | time: 94.32秒
# [Val] Classification Report:
#               precision    recall  f1-score   support
# 
#            0     0.5785    0.6467    0.6107   2856415
#            1     0.5318    0.4600    0.4933   2491785
# 
#     accuracy                         0.5597   5348200
#    macro avg     0.5551    0.5533    0.5520   5348200
# weighted avg     0.5567    0.5597    0.5560   5348200
# 
# [Val] AUC: 0.5765979914375632
# test AUC: 0.5616 | time: 214.41秒
# [TEST] AUC: 0.5615735208682778

# Based on median(573.00)Number of features retained after filtering: 38
# Examples of reserved features: ['feature_00' 'feature_01' 'feature_02' 'feature_03' 'feature_04'
#  'feature_05' 'feature_07' 'feature_08' 'feature_14' 'feature_15'
#  'feature_17' 'feature_19' 'feature_20' 'feature_22' 'feature_23'
#  'feature_24' 'feature_25' 'feature_28' 'feature_29' 'feature_30'
#  'feature_36' 'feature_39' 'feature_47' 'feature_49' 'feature_50'
#  'feature_51' 'feature_52' 'feature_53' 'feature_54' 'feature_55'
#  'feature_58' 'feature_59' 'feature_60' 'feature_66' 'feature_68'
#  'feature_69' 'feature_71' 'feature_72']

###########lgb3
# 使用MICE之后的数据集+特征工程+处理异常值
# 属于是能尝试的全试一下看看了

#先读取MICE数据
X_train_mice = joblib.load("X_train_mice.pkl")
X_val_mice = joblib.load("X_val_mice.pkl")
X_test_mice = joblib.load("X_test_mice.pkl")
X_future_mice = joblib.load("X_future_mice.pkl")

# 筛选特征值
X_train_filtered = X_train_mice[important_features]
X_val_filtered = X_val_mice[important_features]
X_test_filtered = X_test_mice[important_features]
X_future_filtered = X_future_mice[important_features]

# 进行极端值处理
from scipy.stats import zscore
import pandas as pd
import numpy as np


def handle_outliers_zscore(df, threshold=3.0):
    df_out = df.copy()
    z_scores = np.abs(zscore(df_out))
    medians = df_out.median()

    # 替换异常值为中位数
    for col_idx, col in enumerate(df_out.columns):
        outlier_mask = z_scores[:, col_idx] > threshold
        df_out.loc[outlier_mask, col] = medians[col]
    return df_out


# 对所有筛选后的数据集进行异常值处理
X_train_filtered = handle_outliers_zscore(X_train_filtered)
X_val_filtered = handle_outliers_zscore(X_val_filtered)
X_test_filtered = handle_outliers_zscore(X_test_filtered)
X_future_filtered = handle_outliers_zscore(X_future_filtered)

print("✅ 异常值已处理（z-score > 3 替换为中位数）。")

#模型开跑
print("\n=== 开始模型训练 ===")
train_start = time.time()

train_data = lgb.Dataset(X_train_filtered, label=y_train)
val_data = lgb.Dataset(X_val_filtered, label=y_val, reference=train_data)

# 模型训练
lgb_model = lgb.train(
    params=lgb_params,
    train_set=train_data,
    valid_sets=[val_data],
    num_boost_round=1000
)

print(f"模型训练完成，耗时: {time.time()-train_start:.2f}秒")

from sklearn.metrics import roc_curve, auc

def plot_roc_curve(y_true, y_score, title, save_path):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.4f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# 模型评估
print("\n=== lgb_3pre ===")
def evaluate_model(model, X, y, dataset_name):
    start = time.time()
    proba = model.predict(X)
    auc = roc_auc_score(y, proba)
    print(f"{dataset_name} AUC: {auc:.4f} | time: {time.time()-start:.2f}秒")
    return proba

from sklearn.metrics import roc_auc_score

lgb_val_proba = evaluate_model(lgb_model, X_val_filtered, y_val, "test")
lgb_val_pred = (lgb_val_proba >= 0.5).astype(int)
print("[Val] Classification Report:")
print(classification_report(y_val, lgb_val_pred, digits=4))
print("[Val] AUC:", roc_auc_score(y_val, lgb_val_proba))

lgb_test_proba = evaluate_model(lgb_model, X_test_filtered, y_test, "test")
print("[TEST] AUC:", roc_auc_score(y_test, lgb_test_proba))
# 验证集 ROC 曲线
plot_roc_curve(y_val, lgb_val_proba,
               title="ROC Curve - Validation Set",
               save_path="lgb_roc_curve_3pre.png")

# Confusion Matrix
cm = confusion_matrix(y_val, lgb_val_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - Validation Set lgb_mice_z")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# PR Curve
precision, recall, _ = precision_recall_curve(y_val, lgb_val_proba)
plt.plot(recall, precision)
plt.title("Precision-Recall Curve - Validation Set lgb_mice_z")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.grid()
plt.show()

# === lgb_3pre ===
# test AUC: 0.5743 | time: 90.90秒
# [Val] Classification Report:
#               precision    recall  f1-score   support
# 
#            0     0.5768    0.6482    0.6104   2856415
#            1     0.5301    0.4549    0.4896   2491785
# 
#     accuracy                         0.5581   5348200
#    macro avg     0.5534    0.5515    0.5500   5348200
# weighted avg     0.5550    0.5581    0.5541   5348200
# 
# [Val] AUC: 0.5743418126637555
# test AUC: 0.5593 | time: 207.60秒
# [TEST] AUC: 0.5592606564786361
"""

"""
# 由于lgb模型可以天然识别缺失值，MICE并非必要，因此这里考虑进一步筛选特征值，直接使用lgb内嵌功能识别缺失值
# 选择思路是选择重要程度高于重要分数中位数的特征
total_start = time.time()

#载入MICE转换后的数据
# X_train_mice = joblib.load("X_train_mice.pkl")
# X_val_mice = joblib.load("X_val_mice.pkl")
# X_test_mice = joblib.load("X_test_mice.pkl")
# X_future_mice = joblib.load("X_future_mice.pkl")

X_train_clean, y_train = joblib.load("Xy_train_clean.pkl")
X_val_clean, y_val = joblib.load("Xy_val_clean.pkl")
X_test_clean, y_test = joblib.load("Xy_test_clean.pkl")
X_future_clean = joblib.load("X_future_clean.pkl")
df_future_eval = joblib.load("df_future_eval_clean.pkl")



# LightGBM 参数设置
lgb_params = {
    'use_missing': True,          # 显式处理缺失值
    'zero_as_missing': False,     # 区分0和真正的缺失
    # 基础参数
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'device': 'gpu',  # 必须启用GPU加速

    # 树结构
    'num_leaves': 511,
    'max_depth': -1,
    'min_data_in_leaf': 500,
    'extra_trees': True,

    # 正则化
    'lambda_l1': 0.05,
    'lambda_l2': 0.05,
    'feature_fraction': 0.7,
    'bagging_fraction': 0.7,

    # 学习策略
    'learning_rate': 0.02,
    'max_bin': 255
}

print("\n=== 开始特征筛选 (RFE) ===")
rfe_start = time.time()

# 创建基础模型
base_model = lgb.LGBMClassifier(**lgb_params)

# 设置RFE参数 - 修改为实际筛选
n_features_to_select = 40  # 目标特征数量调整为40
step = 0.2  # 每次迭代移除20%的特征，加速过程

# 初始化RFE
selector = RFE(
    estimator=base_model,
    n_features_to_select=n_features_to_select,
    step=step,
    verbose=1
)

# 执行RFE
selector = selector.fit(X_train_clean, y_train)

# 获取选择的特征索引
selected_features = selector.support_
selected_indices = np.where(selected_features)[0]

# 筛选特征
X_train_selected = X_train_clean.iloc[:, selected_indices]
X_val_selected = X_val_clean.iloc[:, selected_indices]
X_test_selected = X_test_clean.iloc[:, selected_indices]
X_future_selected = X_future_clean.iloc[:, selected_indices]

# ============ 新增：获取并分析特征重要性 ============
# 训练完整模型获取重要性分数
full_model = lgb.LGBMClassifier(**lgb_params).fit(X_train_clean, y_train)
importances = full_model.feature_importances_

# 创建特征重要性DataFrame
feature_importance = pd.DataFrame({
    'Feature': X_train_clean.columns,
    'Importance': importances,
    'Selected': selected_features,
    'Ranking': selector.ranking_
})

# 计算重要性统计
importance_stats = {
    'Max': np.max(importances),
    'Min': np.min(importances),
    'Mean': np.mean(importances),
    'Median': np.median(importances),
    'Std': np.std(importances)
}

median_importance = np.median(importances)
important_feature_mask = feature_importance['Importance'] >= median_importance
important_features = feature_importance[important_feature_mask]['Feature'].values

# 打印保留的特征信息
print(f"\nBased on median({median_importance:.2f})Number of features retained after filtering: {len(important_features)}")
print("Examples of reserved features:", important_features[:])

# 筛选各个数据集中的特征列
X_train_filtered = X_train_clean[important_features]
X_val_filtered = X_val_clean[important_features]
X_test_filtered = X_test_clean[important_features]
X_future_filtered = X_future_clean[important_features]

# 保存筛选后的数据
joblib.dump((X_train_filtered, y_train), "Xy_train_filtered.pkl")
joblib.dump((X_val_filtered, y_val), "Xy_val_filtered.pkl")
joblib.dump((X_test_filtered, y_test), "Xy_test_filtered.pkl")
joblib.dump(X_future_filtered, "X_future_filtered.pkl")

print("\n已保存基于特征重要性中位数筛选后的数据。")
"""

"""
#使用筛选后的数据集，结果并不比使用MICE后的结果好，可以进行比较，还是用MICE填补缺失值会好一点
X_train_filtered, y_train = joblib.load("Xy_train_filtered.pkl")
X_val_filtered, y_val = joblib.load("Xy_val_filtered.pkl")
X_test_filtered, y_test = joblib.load("Xy_test_filtered.pkl")
X_future_filtered = joblib.load("X_future_filtered.pkl")
df_future_eval = joblib.load("df_future_eval_clean.pkl")


# 创建数据集
print("\n=== 开始模型训练 ===")
train_start = time.time()

lgb_params = {
    'use_missing': True,          # 显式处理缺失值
    'zero_as_missing': False,     # 区分0和真正的缺失
    # 基础参数
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'device': 'gpu',  # 必须启用GPU加速

    # 树结构
    'num_leaves': 511,
    'max_depth': -1,
    'min_data_in_leaf': 500,
    'extra_trees': True,

    # 正则化
    'lambda_l1': 0.05,
    'lambda_l2': 0.05,
    'feature_fraction': 0.7,
    'bagging_fraction': 0.7,

    # 学习策略
    'learning_rate': 0.02,
    'max_bin': 255
}

train_data = lgb.Dataset(X_train_filtered, label=y_train)
val_data = lgb.Dataset(X_val_filtered, label=y_val, reference=train_data)

# 模型训练
lgb_model = lgb.train(
    params=lgb_params,
    train_set=train_data,
    valid_sets=[val_data],
    num_boost_round=1000
)

print(f"模型训练完成，耗时: {time.time()-train_start:.2f}秒")

from sklearn.metrics import roc_curve, auc

def plot_roc_curve(y_true, y_score, title, save_path):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.4f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# 模型评估
print("\n=== 模型评估 ===")
def evaluate_model(model, X, y, dataset_name):
    start = time.time()
    proba = model.predict(X)
    auc = roc_auc_score(y, proba)
    print(f"{dataset_name} AUC: {auc:.4f} | 耗时: {time.time()-start:.2f}秒")
    return proba

from sklearn.metrics import roc_auc_score

lgb_val_proba = evaluate_model(lgb_model, X_val_filtered, y_val, "验证集")
lgb_val_pred = (lgb_val_proba >= 0.5).astype(int)
print("[Val] Classification Report:")
print(classification_report(y_val, lgb_val_pred, digits=4))
print("[Val] AUC:", roc_auc_score(y_val, lgb_val_proba))

lgb_test_proba = evaluate_model(lgb_model, X_test_filtered, y_test, "测试集")
print("[TEST] AUC:", roc_auc_score(y_test, lgb_test_proba))
# 验证集 ROC 曲线
plot_roc_curve(y_val, lgb_val_proba,
               title="ROC Curve - Validation Set",
               save_path="lgb_roc_val_curve.png")

# Confusion Matrix
cm = confusion_matrix(y_val, lgb_val_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - Validation Set")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# PR Curve
precision, recall, _ = precision_recall_curve(y_val, lgb_val_proba)
plt.plot(recall, precision)
plt.title("Precision-Recall Curve - Validation Set")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.grid()
plt.show()

def safe_assign_predictions(df_eval, predictions, feature_index):
 
    # 创建临时Series用于对齐
    pred_series = pd.Series(predictions, index=feature_index, name='pred_proba')

    # 执行外连接对齐
    combined = df_eval.join(pred_series, how='outer')

    # 分析对齐结果
    missing_mask = combined['pred_proba'].isna()
    extra_mask = combined[df_eval.columns].isna().any(axis=1)

    print(f"\n=== 索引对齐报告 ===")
    print(f"原始评估数据数量: {len(df_eval)}")
    print(f"预测结果数量: {len(predictions)}")
    print(f"对齐后总数: {len(combined)}")
    print(f"缺失预测的评估数据数量: {missing_mask.sum()}")
    print(f"多余预测结果数量: {extra_mask.sum()}")

    # 处理缺失预测的情况（保留但标记）
    combined['pred_proba'].fillna(-1, inplace=True)  # 用-1表示缺失预测
    combined['label_bin'] = (combined['responder_6'] > 0).astype(int)

    return combined[~extra_mask]  # 去除多余预测


# 未来预测
print("\n=== 未来数据预测 ===")
future_start = time.time()

lgb_future_proba = lgb_model.predict(X_future_filtered)
print(f"未来数据预测完成，耗时: {time.time() - future_start:.2f}秒")

# 安全赋值
df_plot = safe_assign_predictions(df_future_eval, lgb_future_proba, X_future_filtered.index)


# 生成可视化
print("\n=== 生成可视化 ===")
plot_start = time.time()


def plot_symbol_predictions(df_plot, symbol_id, save=True):
    df_symbol = df_plot[df_plot["symbol_id"] == symbol_id].copy()

    # 二值化标签
    df_symbol["label_bin"] = (df_symbol["responder_6"] > 0).astype(int)

    # 排序 & 去重
    df_symbol = df_symbol.sort_values("time_id")
    df_symbol = df_symbol.drop_duplicates(subset="time_id", keep="first")

    # 设置图像大小
    plt.figure(figsize=(120, 4))

    # 屏蔽 pred_proba < 0.05 的位置，用 NaN 替换，使其不连线
    line_data = df_symbol.copy()
    line_data['plot_proba'] = line_data['pred_proba'].where(line_data['pred_proba'] >= 0.05, np.nan)

    # 绘制主预测折线（已剔除低值）
    plt.plot(line_data["time_id"], line_data["plot_proba"],
             label="Predicted Probability", linewidth=2, color='tab:blue')

    # 标记 responder_6 > 0 的点
    df_pos = df_symbol[df_symbol["label_bin"] == 1]
    plt.scatter(df_pos["time_id"], df_pos["label_bin"],
                color='green', s=10, alpha=0.7, label="responder_6 > 0")

    # 标记 responder_6 ≤ 0 的点
    df_neg = df_symbol[df_symbol["label_bin"] == 0]
    plt.scatter(df_neg["time_id"], df_neg["label_bin"],
                color='red', s=10, alpha=0.5, label="responder_6 ≤ 0")

    # 标记被屏蔽的低置信度区域（x标记，固定y=0.5）
    low_pred = df_symbol[df_symbol["pred_proba"] < 0.05]
    plt.scatter(low_pred["time_id"], [0.5] * len(low_pred),
                color='black', s=15, marker='x', label='MICE fit missing (y=0.5)')

    # 图像设置
    plt.title(f"Prediction vs Actual - Symbol {symbol_id}")
    plt.xlabel("time_id")
    plt.ylabel("Predicted Probability / Actual Label")
    plt.ylim(-0.1, 1.1)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if save:
        plt.savefig(f"symbol_{symbol_id}_plot.png", dpi=200)
        plt.close()
    else:
        plt.show()


symbols_to_plot = df_plot['symbol_id'].unique()[:3]
for symbol_id in symbols_to_plot:
    plot_symbol_predictions(df_plot, symbol_id)



print(f"可视化完成，耗时: {time.time() - plot_start:.2f}秒")
# 保存模型和预测结果
joblib.dump(lgb_model, 'lgbm_mice_model.pkl')
df_plot.to_csv('future_predictions.csv', index=False)

#MICE 结果
# === 模型评估 ===
# 验证集 AUC: 0.5827 | 耗时: 87.41秒
# [Val] Classification Report:
#               precision    recall  f1-score   support
#
#            0     0.5826    0.6452    0.6123   2856415
#            1     0.5362    0.4702    0.5010   2491785
#
#     accuracy                         0.5636   5348200
#    macro avg     0.5594    0.5577    0.5567   5348200
# weighted avg     0.5610    0.5636    0.5605   5348200
#
# [Val] AUC: 0.5826989396728915
# 测试集 AUC: 0.5673 | 耗时: 198.55秒
# [TEST] AUC: 0.5672834103737856

# === 索引对齐报告 ===
# 原始评估数据数量: 6274576
# 预测结果数量: 6140024
# 对齐后总数: 6274576
# 缺失预测的评估数据数量: 134552
# 多余预测结果数量: 0

#筛选特征的结果
=== 模型评估 ===
验证集 AUC: 0.5771 | 耗时: 100.00秒
[Val] Classification Report:
              precision    recall  f1-score   support

           0     0.5786    0.6481    0.6114   2856415
           1     0.5322    0.4588    0.4928   2491785

    accuracy                         0.5599   5348200
   macro avg     0.5554    0.5535    0.5521   5348200
weighted avg     0.5569    0.5599    0.5561   5348200

[Val] AUC: 0.5770575725622308
测试集 AUC: 0.5616 | 耗时: 218.35秒
[TEST] AUC: 0.561602168553753
"""


# 普通的三层mlp代码，使用MICE梳理后的数据，但是效果不好，这个模型是mlp模型的基准线
# 确认设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

import torch
import torch.nn as nn

# 载入数据
X_train_mice = joblib.load("X_train_mice.pkl")
X_val_mice = joblib.load("X_val_mice.pkl")
X_test_mice = joblib.load("X_test_mice.pkl")
X_future_mice = joblib.load("X_future_mice.pkl")
X_train_clean, y_train = joblib.load("Xy_train_clean.pkl")
X_val_clean, y_val = joblib.load("Xy_val_clean.pkl")
X_test_clean, y_test = joblib.load("Xy_test_clean.pkl")
X_future_clean = joblib.load("X_future_clean.pkl")
df_future_eval = joblib.load("df_future_eval_clean.pkl")

# 基础直接使用MICE版本
# 标准化数据
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_mice)
X_val_scaled = scaler.transform(X_val_mice)
X_test_scaled = scaler.transform(X_test_mice)
X_future_scaled = scaler.transform(X_future_mice)


# 设置输入维度
input_size = X_train_scaled.shape[1]

# 定义模型
model = nn.Sequential(
    nn.Linear(input_size, 256),   # Layer 1
    nn.ReLU(),
    nn.Linear(256, 128),          # Layer 2
    nn.ReLU(),
    nn.Linear(128, 64),           # Layer 3
    nn.ReLU(),
    nn.Linear(64, 1)              # Final output layer (logit)
).to(device)

# 损失函数 & 优化器
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 准备训练数据
X_tensor = torch.FloatTensor(X_train_scaled)
y_tensor = torch.FloatTensor(y_train.values if hasattr(y_train, 'values') else y_train)

train_dataset = TensorDataset(X_tensor, y_tensor)
train_loader = DataLoader(train_dataset, batch_size=4096, shuffle=True)

# 训练模型
EPOCHS = 20  # 你可以根据内存/GPU 情况加大
start_time = time.time()
print("=== 开始训练 MLP 模型 ===")

model.train()
for epoch in range(EPOCHS):
    epoch_start = time.time()  # ⏱️ 记录每个 epoch 的开始时间
    epoch_loss = 0

    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(x_batch).squeeze()
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * x_batch.size(0)

    avg_loss = epoch_loss / len(train_loader.dataset)

    model.eval()
    with torch.no_grad():
        val_tensor = torch.FloatTensor(X_val_scaled).to(device)
        val_outputs = model(val_tensor).squeeze()
        val_probs = torch.sigmoid(val_outputs).cpu().numpy()
        val_auc = roc_auc_score(y_val, val_probs)

    epoch_time = time.time() - epoch_start
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}, Val AUC: {val_auc:.4f}, Time: {epoch_time:.2f}s")

import gc
torch.cuda.empty_cache()
gc.collect()

# 模型评估
print("\n=== 模型评估 ===")
def evaluate_model(model, X_scaled, y_true, dataset_name, batch_size=4096):
    model.eval()
    y_pred_all = []

    X_tensor = torch.FloatTensor(X_scaled)
    data_loader = DataLoader(X_tensor, batch_size=batch_size)

    with torch.no_grad():
        for x_batch in data_loader:
            x_batch = x_batch.to(device)
            logits = model(x_batch).squeeze()
            probs = torch.sigmoid(logits).cpu().numpy()
            y_pred_all.extend(probs)

    y_pred_all = np.array(y_pred_all)
    auc = roc_auc_score(y_true, y_pred_all)
    print(f"{dataset_name} AUC: {auc:.4f}")
    return y_pred_all

from sklearn.metrics import roc_auc_score

mlp_val_proba = evaluate_model(model, X_val_scaled, y_val, "test")
mlp_val_pred = (mlp_val_proba >= 0.5).astype(int)
print("[Val] Classification Report:")
print(classification_report(y_val, mlp_val_pred, digits=4))
print("[Val] AUC:", roc_auc_score(y_val, mlp_val_proba))

mlp_test_proba = evaluate_model(model, X_test_scaled, y_test, "test")
print("[TEST] AUC:", roc_auc_score(y_test, mlp_test_proba))

def plot_roc_curve(y_true, y_score, title, save_path):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.4f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
# 验证集 ROC 曲线
plot_roc_curve(y_val, mlp_val_proba,
               title="ROC Curve - Validation Set",
               save_path="mlp_roc_curve_mice.png")

# Confusion Matrix
cm = confusion_matrix(y_val, mlp_val_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - Validation Set mlp_mice")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# PR Curve
precision, recall, _ = precision_recall_curve(y_val, mlp_val_proba)
plt.plot(recall, precision)
plt.title("Precision-Recall Curve - Validation Set mlp_mice")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.grid()
plt.show()

# ---------- FUTURE 可视化 (使用 MLP 模型预测概率) ----------
print("\n=== 可视化 Future 集预测结果 ===")

model.eval()
with torch.no_grad():
    X_future_tensor = torch.FloatTensor(X_future_scaled).to(device)
    future_logits = model(X_future_tensor).squeeze()
    y_future_pred_proba = torch.sigmoid(future_logits).cpu().numpy()

df_plot = df_future_eval.iloc[:len(y_future_pred_proba)].copy()
df_plot["pred_proba"] = y_future_pred_proba
df_plot["label_bin"] = (df_plot["responder_6"] > 0).astype(int)

symbol_id_to_plot = 0
df_symbol = df_plot[df_plot["symbol_id"] == symbol_id_to_plot].sort_values("time_id")
df_symbol = df_symbol.drop_duplicates(subset="time_id", keep="first")  # 避免重复 time_id

plt.figure(figsize=(150, 5))

plt.plot(
    df_symbol["time_id"],
    df_symbol["pred_proba"],
    label="Predicted Prob (MLP)",
    linewidth=2,
    color='tab:blue'
)

df_up = df_symbol[df_symbol["label_bin"] == 1]
plt.scatter(df_up["time_id"], df_up["label_bin"], color='green', s=15, alpha=0.6, label="responder_6 > 0")

df_down = df_symbol[df_symbol["label_bin"] == 0]
plt.scatter(df_down["time_id"], df_down["label_bin"], color='red', s=15, alpha=0.4, label="responder_6 ≤ 0")

plt.title(f"MLP Prediction vs Actual (symbol_id={symbol_id_to_plot})")
plt.xlabel("time_id")
plt.ylabel("Probability / Direction")
plt.ylim(-0.1, 1.1)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
# === 模型评估 ===
# test AUC: 0.5544
# [Val] Classification Report:
#               precision    recall  f1-score   support
# 
#            0     0.5664    0.6276    0.5955   2856415
#            1     0.5128    0.4493    0.4789   2491785
# 
#     accuracy                         0.5445   5348200
#    macro avg     0.5396    0.5384    0.5372   5348200
# weighted avg     0.5414    0.5445    0.5412   5348200
# 
# [Val] AUC: 0.5543638987687336
# test AUC: 0.5376
# [TEST] AUC: 0.5376174109499735


"""
# mlp模型第二版，用筛选的特征
# + Dropout，提升表达能力同时防止过拟合
# 加入 pos_weight 强调正样本
# 更好的随机种子设定，提高可复现性
# 结果上正样本确实recall提升了，但是0也下降了
# 看训练过程nn.ReLU()明显是死了，并未成功传导，所以下一版本要设置
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import joblib, time, gc, random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve
)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader

# 设置种子
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
seed_everything()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 数据加载
X_train_mice = joblib.load("X_train_mice.pkl")
X_val_mice = joblib.load("X_val_mice.pkl")
X_test_mice = joblib.load("X_test_mice.pkl")
X_future_mice = joblib.load("X_future_mice.pkl")
X_train_clean, y_train = joblib.load("Xy_train_clean.pkl")
X_val_clean, y_val = joblib.load("Xy_val_clean.pkl")
X_test_clean, y_test = joblib.load("Xy_test_clean.pkl")
X_future_clean = joblib.load("X_future_clean.pkl")
df_future_eval = joblib.load("df_future_eval_clean.pkl")

#筛选特征版本
selected_features = [
  'feature_00', 'feature_01', 'feature_02', 'feature_03', 'feature_04',
  'feature_05', 'feature_07', 'feature_08', 'feature_14', 'feature_15',
  'feature_17', 'feature_19', 'feature_20', 'feature_22', 'feature_23',
  'feature_24', 'feature_25', 'feature_28', 'feature_29', 'feature_30',
  'feature_36', 'feature_39', 'feature_47', 'feature_49', 'feature_50',
  'feature_51', 'feature_52', 'feature_53', 'feature_54', 'feature_55',
  'feature_58', 'feature_59', 'feature_60', 'feature_66', 'feature_68',
  'feature_69', 'feature_71', 'feature_72'
]

X_train_selected = X_train_mice[selected_features]
X_val_selected = X_val_mice[selected_features]
X_test_selected = X_test_mice[selected_features]
X_future_selected = X_future_mice[selected_features]

# 标准化数据
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_selected)
X_val_scaled = scaler.transform(X_val_selected)
X_test_scaled = scaler.transform(X_test_selected)
X_future_scaled = scaler.transform(X_future_selected)

input_size = X_train_scaled.shape[1]

# 定义改进后的模型
class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

model = MLP(input_size).to(device)

# 计算正负样本比例
pos_weight_val = (len(y_train) - y_train.sum()) / y_train.sum()
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_val]).to(device))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 数据加载器
X_tensor = torch.FloatTensor(X_train_scaled)
y_tensor = torch.FloatTensor(y_train.values if hasattr(y_train, 'values') else y_train)
train_dataset = TensorDataset(X_tensor, y_tensor)
train_loader = DataLoader(train_dataset, batch_size=4096, shuffle=True)

# 训练
EPOCHS = 20
start_time = time.time()
print("=== 开始训练 MLP 模型 ===")

for epoch in range(EPOCHS):
    epoch_start = time.time()
    model.train()
    epoch_loss = 0
    y_train_probs = []

    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(x_batch).squeeze()
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * x_batch.size(0)
        y_train_probs.extend(torch.sigmoid(outputs).detach().cpu().numpy())

    avg_loss = epoch_loss / len(train_loader.dataset)
    train_auc = roc_auc_score(y_train, y_train_probs)

    # 验证集 AUC
    model.eval()
    with torch.no_grad():
        val_tensor = torch.FloatTensor(X_val_scaled).to(device)
        val_outputs = model(val_tensor).squeeze()
        val_probs = torch.sigmoid(val_outputs).cpu().numpy()
        val_auc = roc_auc_score(y_val, val_probs)

    epoch_time = time.time() - epoch_start
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}, Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f}, Time: {epoch_time:.2f}s")

# 清理内存
torch.cuda.empty_cache()
gc.collect()

# ========== 模型评估 ========== #
print("\n=== 模型评估 ===")
def evaluate_model(model, X_scaled, y_true, dataset_name):
    model.eval()
    y_pred_all = []

    X_tensor = torch.FloatTensor(X_scaled)
    loader = DataLoader(X_tensor, batch_size=4096)

    with torch.no_grad():
        for x_batch in loader:
            x_batch = x_batch.to(device)
            logits = model(x_batch).squeeze()
            probs = torch.sigmoid(logits).cpu().numpy()
            y_pred_all.extend(probs)

    y_pred_all = np.array(y_pred_all)
    auc_score = roc_auc_score(y_true, y_pred_all)
    print(f"{dataset_name} AUC: {auc_score:.4f}")
    return y_pred_all

# 验证集评估
val_proba = evaluate_model(model, X_val_scaled, y_val, "Validation")
val_pred = (val_proba >= 0.5).astype(int)
print("[Val] Classification Report:\n", classification_report(y_val, val_pred, digits=4))
print("[Val] AUC:", roc_auc_score(y_val, val_proba))

# 测试集评估
test_proba = evaluate_model(model, X_test_scaled, y_test, "Test")
print("[TEST] AUC:", roc_auc_score(y_test, test_proba))

# ROC Curve
def plot_roc_curve(y_true, y_score, title, save_path):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc_val = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {auc_val:.4f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(title)
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

plot_roc_curve(y_val, val_proba, "ROC Curve - Validation", "mlp_roc_val.png")

# 混淆矩阵
cm = confusion_matrix(y_val, val_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - Validation Set")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# PR Curve
precision, recall, _ = precision_recall_curve(y_val, val_proba)
plt.plot(recall, precision)
plt.title("Precision-Recall Curve - Validation Set")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.grid(True)
plt.show()

# ---------- FUTURE 可视化 (使用 MLP 模型预测概率) ----------
print("\n=== 可视化 Future 集预测结果 ===")

model.eval()
with torch.no_grad():
    X_future_tensor = torch.FloatTensor(X_future_scaled).to(device)
    future_logits = model(X_future_tensor).squeeze()
    y_future_pred_proba = torch.sigmoid(future_logits).cpu().numpy()

df_plot = df_future_eval.iloc[:len(y_future_pred_proba)].copy()
df_plot["pred_proba"] = y_future_pred_proba
df_plot["label_bin"] = (df_plot["responder_6"] > 0).astype(int)

df_symbol = df_plot[df_plot["symbol_id"] == symbol_id_to_plot].sort_values("time_id")
df_symbol = df_symbol.drop_duplicates(subset="time_id", keep="first")  # 避免重复 time_id

plt.figure(figsize=(150, 5))

plt.plot(
    df_symbol["time_id"],
    df_symbol["pred_proba"],
    label="Predicted Prob (MLP)",
    linewidth=2,
    color='tab:blue'
)

df_up = df_symbol[df_symbol["label_bin"] == 1]
plt.scatter(df_up["time_id"], df_up["label_bin"], color='green', s=15, alpha=0.6, label="responder_6 > 0")

df_down = df_symbol[df_symbol["label_bin"] == 0]
plt.scatter(df_down["time_id"], df_down["label_bin"], color='red', s=15, alpha=0.4, label="responder_6 ≤ 0")

plt.title(f"MLP Prediction vs Actual (symbol_id={symbol_id_to_plot})")
plt.xlabel("time_id")
plt.ylabel("Probability / Direction")
plt.ylim(-0.1, 1.1)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === 模型评估 ===
# Validation AUC: 0.5732
# [Val] Classification Report:
#                precision    recall  f1-score   support
#
#            0     0.5828    0.5798    0.5813   2856415
#            1     0.5211    0.5242    0.5226   2491785
#
#     accuracy                         0.5539   5348200
#    macro avg     0.5520    0.5520    0.5520   5348200
# weighted avg     0.5541    0.5539    0.5540   5348200
#
# [Val] AUC: 0.5732464366179105
# Test AUC: 0.5562
# [TEST] AUC: 0.5562462054863556
"""


"""
# mlp 第三版
# 从 38 个特征中选出 top10 特征用于构造交叉项
# RobustMLP 模型（改进版两层结构 + 残差）
# CyclicLR 学习率调度
# 考虑内存反正三层传导的神经也是死的，改成两层并强制传输残差，并加入重要程度最高的特征交叉项

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 数据加载
X_train_mice = joblib.load("X_train_mice.pkl")
X_val_mice = joblib.load("X_val_mice.pkl")
X_test_mice = joblib.load("X_test_mice.pkl")
X_future_mice = joblib.load("X_future_mice.pkl")

X_train_clean, y_train = joblib.load("Xy_train_clean.pkl")
X_val_clean, y_val = joblib.load("Xy_val_clean.pkl")
X_test_clean, y_test = joblib.load("Xy_test_clean.pkl")
X_future_clean = joblib.load("X_future_clean.pkl")

df_future_eval = joblib.load("df_future_eval_clean.pkl")

selected_features_38 = [
    'feature_00', 'feature_01', 'feature_02', 'feature_03', 'feature_04',
    'feature_05', 'feature_07', 'feature_08', 'feature_14', 'feature_15',
    'feature_17', 'feature_19', 'feature_20', 'feature_22', 'feature_23',
    'feature_24', 'feature_25', 'feature_28', 'feature_29', 'feature_30',
    'feature_36', 'feature_39', 'feature_47', 'feature_49', 'feature_50',
    'feature_51', 'feature_52', 'feature_53', 'feature_54', 'feature_55',
    'feature_58', 'feature_59', 'feature_60', 'feature_66', 'feature_68',
    'feature_69', 'feature_71', 'feature_72'
]

X_train = X_train_mice[selected_features_38].copy()
X_val = X_val_mice[selected_features_38].copy()
X_test = X_test_mice[selected_features_38].copy()
X_future = X_future_mice[selected_features_38].copy()

top10 = ['feature_04', 'feature_01', 'feature_15', 'feature_59', 'feature_05',
         'feature_52', 'feature_55', 'feature_58', 'feature_54', 'feature_53']

cross_features = []
for i in range(len(top10)):
    for j in range(i+1, len(top10)):
        f1, f2 = top10[i], top10[j]
        fname = f"{f1}_x_{f2}"
        for df in [X_train, X_val, X_test, X_future]:
            df[fname] = df[f1] * df[f2]
        cross_features.append(fname)

final_features = selected_features_38 + cross_features

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train[final_features])
X_val_scaled = scaler.transform(X_val[final_features])
X_test_scaled = scaler.transform(X_test[final_features])
X_future_scaled = scaler.transform(X_future[final_features])

class RobustMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.act1 = nn.LeakyReLU(0.01)

        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.act2 = nn.LeakyReLU(0.01)

        self.shortcut = nn.Linear(128, 64)
        self.output = nn.Linear(64, 1)
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='leaky_relu')
        nn.init.xavier_normal_(self.output.weight)

    def forward(self, x):
        x1 = self.act1(self.bn1(self.fc1(x)))
        x2 = self.act2(self.bn2(self.fc2(x1)))
        return self.output(x2 + self.shortcut(x1))

# 模型初始化
input_size = X_train_scaled.shape[1]
model = RobustMLP(input_size).to(device)

# Loss & Optimizer
pos_weight_val = (len(y_train) - y_train.sum()) / y_train.sum()
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_val]).to(device))
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-5, max_lr=1e-3,
                                              step_size_up=200, mode='triangular2', cycle_momentum=False)

# 数据加载器
train_loader = DataLoader(
    TensorDataset(torch.FloatTensor(X_train_scaled), torch.FloatTensor(y_train.values)),
    batch_size=2048, shuffle=True
)

# 训练模型
for epoch in range(35):
    model.train()
    epoch_loss = 0
    y_train_pred = []

    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb).squeeze()
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        scheduler.step()
        epoch_loss += loss.item() * xb.size(0)
        y_train_pred.extend(torch.sigmoid(logits).detach().cpu().numpy())

    train_auc = roc_auc_score(y_train, y_train_pred)

    model.eval()
    with torch.no_grad():
        val_tensor = torch.FloatTensor(X_val_scaled).to(device)
        val_logits = model(val_tensor).squeeze()
        val_probs = torch.sigmoid(val_logits).cpu().numpy()
        val_auc = roc_auc_score(y_val, val_probs)

    print(f"Epoch {epoch+1} | Loss: {epoch_loss/len(train_loader.dataset):.4f} | "
          f"Val AUC: {val_auc:.4f}")

# ========== 模型评估 ========== #
print("\n=== 模型评估 ===")
def evaluate_model(model, X_scaled, y_true, dataset_name):
    model.eval()
    y_pred_all = []

    X_tensor = torch.FloatTensor(X_scaled)
    loader = DataLoader(X_tensor, batch_size=4096)

    with torch.no_grad():
        for x_batch in loader:
            x_batch = x_batch.to(device)
            logits = model(x_batch).squeeze()
            probs = torch.sigmoid(logits).cpu().numpy()
            y_pred_all.extend(probs)

    y_pred_all = np.array(y_pred_all)
    auc_score = roc_auc_score(y_true, y_pred_all)
    print(f"{dataset_name} AUC: {auc_score:.4f}")
    return y_pred_all

# 验证集
val_proba = evaluate_model(model, X_val_scaled, y_val, "Validation")
val_pred = (val_proba >= 0.5).astype(int)
print("[Val] Classification Report:\n", classification_report(y_val, val_pred, digits=4))

# 测试集
test_proba = evaluate_model(model, X_test_scaled, y_test, "Test")

# ========== ROC Curve ========== #
def plot_roc_curve(y_true, y_score, title, save_path):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc_val = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {auc_val:.4f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(title)
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

plot_roc_curve(y_val, val_proba, "ROC Curve - Validation mlp_cross", "mlp_roc_val_cross.png")

cm = confusion_matrix(y_val, val_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - Validation Set mlp_cross")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

precision, recall, _ = precision_recall_curve(y_val, val_proba)
plt.plot(recall, precision)
plt.title("Precision-Recall Curve - Validation Set mlp_cross")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.grid(True)
plt.show()

# ========== Future 可视化预测 ========== #
print("\n=== 可视化 Future 集预测结果 ===")
model.eval()
with torch.no_grad():
    X_future_tensor = torch.FloatTensor(X_future_scaled).to(device)
    future_logits = model(X_future_tensor).squeeze()
    y_future_pred_proba = torch.sigmoid(future_logits).cpu().numpy()

df_plot = df_future_eval.iloc[:len(y_future_pred_proba)].copy()
df_plot["pred_proba"] = y_future_pred_proba
df_plot["label_bin"] = (df_plot["responder_6"] > 0).astype(int)

symbol_id_to_plot = 0
df_symbol = df_plot[df_plot["symbol_id"] == symbol_id_to_plot].sort_values("time_id")
df_symbol = df_symbol.drop_duplicates(subset="time_id")

plt.figure(figsize=(150, 5))
plt.plot(df_symbol["time_id"], df_symbol["pred_proba"],
         label="Predicted Prob (MLP)", linewidth=2, color='tab:blue')

df_up = df_symbol[df_symbol["label_bin"] == 1]
plt.scatter(df_up["time_id"], df_up["label_bin"], color='green', s=15, alpha=0.6, label="responder_6 > 0")
df_down = df_symbol[df_symbol["label_bin"] == 0]
plt.scatter(df_down["time_id"], df_down["label_bin"], color='red', s=15, alpha=0.4, label="responder_6 ≤ 0")

plt.title(f"MLP Prediction vs Actual (symbol_id={symbol_id_to_plot})")
plt.xlabel("time_id")
plt.ylabel("Probability / Direction")
plt.ylim(-0.1, 1.1)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

"""















































