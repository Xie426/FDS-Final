import os
import pandas as pd
import numpy as np
import time
import pyarrow.parquet as pq
import lightgbm as lgb
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
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
# 使用MICE方法处理缺失值，但是跑得太慢了所以先取样1%跑一遍流程，再用晚上睡觉跑10%之类的更多的量试试
# 记录总开始时间
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
    'max_iter': 5,            # 减少迭代次数
    'tol': 1e-2,              # 放宽收敛阈值
    'n_nearest_features': 20, # 限制相关特征数
    'initial_strategy': 'mean',
    'random_state': 42,
    # 移除了n_jobs参数
}


# 修改后的TrackedMICE类
class TrackedMICE(IterativeImputer):
    def _setup_pbar(self, n_features):
        self._pbar = tqdm(total=self.max_iter * n_features, desc="MICE迭代进度")

    def _print_verbose_msg(self, iteration, n_imputed, n_missing):
        # tqdm 更新进度
        if hasattr(self, "_pbar") and self._pbar:
            self._pbar.update(n_imputed)

    def fit_transform(self, X, y=None):
        self._setup_pbar(X.shape[1])  # 添加这个以初始化进度条
        with joblib.parallel_backend('threading', n_jobs=-1):
            result = super().fit_transform(X, y)
        self._pbar.close()
        return result


# 分阶段数据处理
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


# 修改后的主处理流程
def run_mice_pipeline(sample_ratio=0.2):
    # 加载原始数据（保留原始索引）
    X_train_clean, y_train = joblib.load("Xy_train_clean.pkl")
    X_val_clean = joblib.load("Xy_val_clean.pkl")[0]
    X_test_clean = joblib.load("Xy_test_clean.pkl")[0]
    X_future_clean = joblib.load("X_future_clean.pkl")

    # 记录原始索引
    original_indices = {
        'train': X_train_clean.index.copy(),
        'val': X_val_clean.index.copy(),
        'test': X_test_clean.index.copy(),
        'future': X_future_clean.index.copy()
    }

    # 智能采样（保留索引）
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
run_mice_pipeline(sample_ratio=0.1)
"""

"""
# 进行lgb模型建模，由于MICE算法原因，future数据集中存在一些行缺失值过多，这种数据MICE会拟合失败并删去，占比在百分之2%多
# 在可视化阶段对于这种被删掉的点进行了标注，虽然有这个问题但还是想尝试不同的缺失值填充手段
total_start = time.time()

# 载入MICE转换后的数据
X_train_mice = joblib.load("X_train_mice.pkl")
X_val_mice = joblib.load("X_val_mice.pkl")
X_test_mice = joblib.load("X_test_mice.pkl")
X_future_mice = joblib.load("X_future_mice.pkl")

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
# 创建数据集
print("\n=== 开始模型训练 ===")
train_start = time.time()

train_data = lgb.Dataset(X_train_mice, label=y_train)
val_data = lgb.Dataset(X_val_mice, label=y_val, reference=train_data)

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

lgb_val_proba = evaluate_model(lgb_model, X_val_mice, y_val, "验证集")
lgb_val_pred = (lgb_val_proba >= 0.5).astype(int)
print("[Val] Classification Report:")
print(classification_report(y_val, lgb_val_pred, digits=4))
print("[Val] AUC:", roc_auc_score(y_val, lgb_val_proba))

lgb_test_proba = evaluate_model(lgb_model, X_test_mice, y_test, "测试集")
print("[TEST] AUC:", roc_auc_score(y_test, lgb_test_proba))
# 验证集 ROC 曲线
plot_roc_curve(y_val, lgb_val_proba,
               title="ROC Curve - Validation Set",
               save_path="lgb_roc_val_curve.png")

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

lgb_future_proba = lgb_model.predict(X_future_mice)
print(f"未来数据预测完成，耗时: {time.time() - future_start:.2f}秒")

# 安全赋值
df_plot = safe_assign_predictions(df_future_eval, lgb_future_proba, X_future_mice.index)



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
"""



# 记录总开始时间
total_start = time.time()

# 检查GPU可用性
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

load_start = time.time()
# 数据标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_mice)
X_val_scaled = scaler.transform(X_val_mice)
X_test_scaled = scaler.transform(X_test_mice)
X_future_scaled = scaler.transform(X_future_mice)

print(f"数据加载和标准化完成，耗时: {time.time() - load_start:.2f}秒")

# 转换为PyTorch张量
train_dataset = TensorDataset(
    torch.FloatTensor(X_train_scaled),
    torch.FloatTensor(y_train.values)
)
val_dataset = TensorDataset(
    torch.FloatTensor(X_val_scaled),
    torch.FloatTensor(y_val.values)
)

# 创建数据加载器
batch_size = 4096
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)


# 定义MLP模型
class FinancialMLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


# 初始化模型
model = FinancialMLP(X_train_scaled.shape[1]).to(device)
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
criterion = nn.BCELoss()

# 训练参数
num_epochs = 50
best_auc = 0
train_losses = []
val_aucs = []

print("\n=== 开始模型训练 ===")
train_start = time.time()

for epoch in range(num_epochs):
    epoch_start = time.time()
    model.train()
    running_loss = 0.0

    # 训练阶段
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    # 验证阶段
    model.eval()
    val_preds = []
    val_labels = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs).squeeze().cpu().numpy()
            val_preds.extend(outputs)
            val_labels.extend(labels.numpy())

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_auc = roc_auc_score(val_labels, val_preds)
    train_losses.append(epoch_loss)
    val_aucs.append(epoch_auc)

    # 保存最佳模型
    if epoch_auc > best_auc:
        best_auc = epoch_auc
        torch.save(model.state_dict(), "best_mlp_model.pth")

    print(f"Epoch {epoch + 1}/{num_epochs} | "
          f"Loss: {epoch_loss:.4f} | Val AUC: {epoch_auc:.4f} | "
          f"Time: {time.time() - epoch_start:.2f}s")

print(f"训练完成，总耗时: {time.time() - train_start:.2f}秒")

# 加载最佳模型
model.load_state_dict(torch.load("best_mlp_model.pth"))


def evaluate_mlp(X, y):
    model.eval()
    X_tensor = torch.FloatTensor(scaler.transform(X)).to(device)
    with torch.no_grad():
        outputs = model(X_tensor).squeeze().cpu().numpy()

    # 计算预测类别（阈值0.5）
    y_pred = np.where(outputs >= 0.5, 1, 0)

    # 返回AUC和分类报告
    return (
        roc_auc_score(y, outputs),
        classification_report(y, y_pred, digits=4)
    )


# 修改后的评估调用代码
print("\n=== 模型评估 ===")
test_auc, test_report = evaluate_mlp(X_test_mice, y_test)
print(f"测试集 AUC: {test_auc:.4f}")
print("\n分类报告：")
print(test_report)

# ========== 1. 安全索引对齐函数 ==========
def safe_assign_predictions(df_eval, predictions, feature_index):
    pred_series = pd.Series(predictions, index=feature_index, name='pred_proba')
    combined = df_eval.join(pred_series, how='outer')

    # 标记缺失情况
    missing_mask = combined['pred_proba'].isna()
    extra_mask = combined[df_eval.columns].isna().any(axis=1)

    print(f"\n=== 索引对齐报告 ===")
    print(f"原始评估数据数量: {len(df_eval)}")
    print(f"预测结果数量: {len(predictions)}")
    print(f"对齐后总数: {len(combined)}")
    print(f"缺失预测的评估数据数量: {missing_mask.sum()}")
    print(f"多余预测结果数量: {extra_mask.sum()}")

    # 用 -1 填充缺失预测
    combined['pred_proba'].fillna(-1, inplace=True)
    combined['label_bin'] = (combined['responder_6'] > 0).astype(int)

    return combined[~extra_mask]  # 移除多余预测


# ========== 2. MLP 模型未来预测 ==========
print("\n=== 未来数据预测（MLP） ===")
future_start = time.time()

# 如果你使用的是 PyTorch 模型
X_future_tensor = torch.FloatTensor(X_future_scaled).to(device)
with torch.no_grad():
    mlp_future_proba = model(X_future_tensor).squeeze().cpu().numpy()

# 如果你使用的是 sklearn MLP 模型（取消上面注释，启用下面一行）
# mlp_future_proba = model.predict_proba(X_future_mice)[:, 1]

print(f"未来数据预测完成，耗时: {time.time() - future_start:.2f}秒")


# ========== 3. 安全对齐与保存 ==========
df_plot = safe_assign_predictions(df_future_eval, mlp_future_proba, X_future_mice.index)

df_plot['pred_label'] = (df_plot['pred_proba'] >= 0.5).astype(int)

# 保存结果
np.save('mlp_future_predictions.npy', mlp_future_proba)
df_plot.to_csv('mlp_future_predictions.csv', index=False)
joblib.dump(df_plot, 'mlp_future_predictions.pkl')


# ========== 4. 可视化函数（风格统一 + 标注补全） ==========
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


# ========== 5. 执行可视化 ==========
print("\n=== 生成可视化 ===")
plot_start = time.time()

symbols_to_plot = df_plot['symbol_id'].unique()[:3]
for symbol_id in symbols_to_plot:
    plot_symbol_predictions(df_plot, symbol_id)

print(f"可视化完成，耗时: {time.time() - plot_start:.2f}秒")







































