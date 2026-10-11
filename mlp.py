"""用多层感知机（MLPClassifier）完成 10 类文本分类。

对照 svm.py 的结构，本脚本输出：
  1) mlp_tuning_heatmap.png —— 超参数（hidden_layer_sizes × alpha）在验证集上的表现
  2) mlp_loss_curve.png     —— 训练集 + 验证集两条损失曲线（一张图）
  3) predictions_mlp.csv    —— 测试集预测结果

调参维度参照 TUNING.md 中对 MLPClassifier 的建议：
  - hidden_layer_sizes: 单层 (50,)/(100,) 起步，再试更宽 (200,) 或更深 (100, 50)
  - alpha: L2 正则，区间 [0.0001, 0.001, 0.01, 0.1]
  - activation: 保持默认 'relu'
  - max_iter: 默认可能不收敛，这里放大到 300

早停（Early Stopping）：本项目**不使用** sklearn 内置的早停（它把监控指标硬编码为
accuracy，切分方式也不便调整），而是**自己实现**——用 partial_fit 逐 epoch 训练，
每轮在「内部验证集」上计算监控指标，连续 n_iter_no_change 轮没有实质改善
（提升 < tol）就停止，并回退到监控指标最优那一轮的权重。
监控指标由 MONITOR_METRIC 指定（默认 f1_macro），因此「调参选参 / 早停 / 报告」
三处的口径完全一致。
"""

from sklearn.neural_network import MLPClassifier
import util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, log_loss
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # 无图形界面环境下直接存图，避免 plt.show() 阻塞
import matplotlib.pyplot as plt
import seaborn as sns

# 让图中的中文能正常显示（Windows 常用中文字体）
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 固定随机性，保证结果可复现（MLP 的权重初始化 + Adam 的随机性都靠它）
np.random.seed(42)

# ── 自己实现的早停（Early Stopping）设置 ───────────────────
MONITOR_METRIC = "f1_macro"  # 早停的监控指标，可选 "f1_macro" / "accuracy"
N_ITER_NO_CHANGE = 10        # 监控指标连续这么多轮没有改善，就停止训练
TOL = 1e-4                   # 提升幅度小于该值不算「改善」
VALIDATION_FRACTION = 0.1    # 从训练数据中切出的「内部验证集」比例（供早停监控）
MAX_EPOCHS = 200             # 单个模型的最大训练轮数（通常会被早停提前结束）


def monitor_score(y_true, y_pred, metric=MONITOR_METRIC):
    """早停用的监控指标。想换成别的指标，改 MONITOR_METRIC 和这里即可。"""
    if metric == "f1_macro":
        return f1_score(y_true, y_pred, average="macro")
    if metric == "accuracy":
        return accuracy_score(y_true, y_pred)
    raise ValueError(f"不支持的监控指标: {metric}")


def mlp_loss(model, X, y):
    """复刻 sklearn MLPClassifier 内部记录的损失，便于与官方口径对齐：

        loss = 交叉熵(log loss) + 0.5 * alpha * ‖W‖² / n_samples

    对应 sklearn 源码 `_compute_loss_grad` 里的
    `loss += (0.5 * self.alpha) * values / n_samples`。
    """
    ce = log_loss(y, model.predict_proba(X), labels=model.classes_)
    l2 = sum(float(np.dot(w.ravel(), w.ravel())) for w in model.coefs_)
    return float(ce + 0.5 * model.alpha * l2 / X.shape[0])


def new_mlp(**kwargs):
    """构造一个「关掉 sklearn 内部早停、逐轮由我们控制」的 MLP。

    - max_iter=1          : 每次 partial_fit 只走 1 个 epoch；
    - early_stopping=False: partial_fit 与 sklearn 内置早停互斥，必须关掉；
    - random_state=RandomState(42): 传「实例」而不是整数 42。因为 partial_fit 每次
      都会用 random_state 重建随机数生成器，传整数会导致每轮 shuffle 出的排列相同；
      传实例则会被复用并向前推进，使每轮打乱顺序不同（更贴近一次 fit() 的训练过程）。
    """
    return MLPClassifier(
        activation="relu", solver="adam",
        max_iter=1,
        early_stopping=False,
        random_state=np.random.RandomState(42),
        **kwargs,
    )


def fit_with_early_stopping(model, X_tr, y_tr, X_val, y_val,
                            monitor=MONITOR_METRIC,
                            n_iter_no_change=N_ITER_NO_CHANGE, tol=TOL,
                            max_epochs=MAX_EPOCHS, record_loss=False):
    """自己实现的早停：逐 epoch 训练，按 monitor 指标挑最优轮次并回退权重。

    - model      : 由 new_mlp() 构造（early_stopping=False、max_iter=1）；
    - X_val/y_val: 用于监控的「内部验证集」，不参与训练；
    - record_loss: 是否同时记录每轮的 train / val 损失（用于画曲线）；
    - 返回 history 字典：每轮监控指标、可选的两条损失曲线、最优轮次与总轮数。
    """
    hist = {"score": [], "train_loss": [], "val_loss": [],
            "best_epoch": 0, "best_score": -np.inf, "n_epochs": 0}
    best_weights, no_improve = None, 0

    for epoch in range(1, max_epochs + 1):
        if epoch == 1:
            model.partial_fit(X_tr, y_tr, classes=np.unique(y_tr))
        else:
            model.partial_fit(X_tr, y_tr)

        # 当前轮次的监控指标（在内部验证集上计算）
        score = monitor_score(y_val, model.predict(X_val), monitor)
        hist["score"].append(score)

        if record_loss:
            hist["train_loss"].append(mlp_loss(model, X_tr, y_tr))
            hist["val_loss"].append(mlp_loss(model, X_val, y_val))

        if score > hist["best_score"] + tol:      # 有实质改善 → 记录最优权重
            hist["best_score"], hist["best_epoch"], no_improve = score, epoch, 0
            best_weights = ([c.copy() for c in model.coefs_],
                            [b.copy() for b in model.intercepts_])
        else:                                     # 没改善 → 计数，够数就停
            no_improve += 1
            if no_improve >= n_iter_no_change:
                break

    if best_weights is not None:                  # 回退到监控指标最优那一轮的权重
        model.coefs_, model.intercepts_ = best_weights
    hist["n_epochs"] = len(hist["score"])
    return hist


# 1. 加载数据
X_train, y_train, X_test_unlabeled = util.load_data()

# 2. 把训练集划分为「训练子集」和「验证集」（分层抽样，验证集占 20%）
#    验证集只用来挑超参数，不参与向量化器的拟合，避免信息泄漏。
X_tr, X_val, y_tr, y_val = util.train_valid_split(
    X_train, y_train, test_size=0.2, random_state=42
)
print(f"训练子集: {len(X_tr)} 条, 验证集: {len(X_val)} 条, 测试集: {len(X_test_unlabeled)} 条")

# 3. 只在训练子集上拟合 TF-IDF 向量化器，再转换验证集
vectorizer = TfidfVectorizer(max_features=5000)
X_tr_tfidf = vectorizer.fit_transform(X_tr)
X_val_tfidf = vectorizer.transform(X_val)

# 4. 在验证集上比较不同超参数组合，确定最佳参数
# hidden_options = [(50,), (100,), (200,), (100, 50)]  # 由小到大：单层 → 更宽 → 更深
hidden_options = [(50,), (100,), (50,25)]
alpha_options = [0.001, 0.01, 0.1]           # L2 正则强度

# 早停需要监控数据，但验证集要留作「模型选择」，所以这里从训练子集里再切出
# VALIDATION_FRACTION 作为「内部验证集」，专门用于早停监控——这与 sklearn 内置早停
# 的做法完全一致，区别只是监控指标换成了 MONITOR_METRIC，且由我们自己控制。
X_it, X_iv, y_it, y_iv = train_test_split(
    X_tr, y_tr, test_size=VALIDATION_FRACTION, random_state=42,
    shuffle=True, stratify=y_tr,
)
# 复用第 3 节的向量化器（已在训练子集上 fit），保证三个集合的特征空间一致
X_it_tfidf = vectorizer.transform(X_it)
X_iv_tfidf = vectorizer.transform(X_iv)

records = []
best_score, best_params = -1.0, None
for hidden in hidden_options:
    for alpha in alpha_options:
        model = new_mlp(hidden_layer_sizes=hidden, alpha=alpha)
        hist = fit_with_early_stopping(model, X_it_tfidf, y_it, X_iv_tfidf, y_iv)

        y_val_pred = model.predict(X_val_tfidf)   # 在「验证集」上评估，用于选参
        acc = accuracy_score(y_val, y_val_pred)
        f1 = f1_score(y_val, y_val_pred, average="macro")
        records.append({
            "hidden_layer_sizes": str(hidden),
            "alpha": alpha,
            "acc": acc,
            "f1_macro": f1,
            "n_iter": hist["n_epochs"],
            "best_epoch": hist["best_epoch"],
        })
        print(f"hidden={str(hidden):<10} alpha={alpha:<7} "
              f"val acc={acc:.4f}  f1_macro={f1:.4f}  "
              f"(早停 {hist['n_epochs']} 轮, 最优 epoch={hist['best_epoch']}, "
              f"内部验证集 {MONITOR_METRIC}={hist['best_score']:.4f})")
        if f1 > best_score:
            best_score = f1
            best_params = {"hidden_layer_sizes": hidden, "alpha": alpha}

results = pd.DataFrame(records)
print(f"\n最佳超参数: {best_params}  (验证集 f1_macro={best_score:.4f})\n")

# 4.1 把验证集结果绘制成热力图：行 = hidden_layer_sizes，列 = alpha，数值 = f1_macro
pivot = results.pivot(index="hidden_layer_sizes", columns="alpha", values="f1_macro")
pivot = pivot.reindex([str(h) for h in hidden_options])   # 保持由小到大的顺序
pivot = pivot[alpha_options]
pivot.columns = [f"{a:g}" for a in pivot.columns]         # alpha 显示成 0.0001 / 0.001 ...

fig, ax = plt.subplots(figsize=(6.5, 4.5))
sns.heatmap(
    pivot,
    annot=True, fmt=".4f", cmap="YlGnBu",
    vmin=pivot.values.min(), vmax=pivot.values.max(),
    linewidths=0.5, linecolor="white",
    cbar_kws={"label": "验证集 F1 (macro)"},
    ax=ax,
)
ax.set_xlabel("alpha (L2 正则)")
ax.set_ylabel("hidden_layer_sizes")
ax.set_title("MLP 超参数在验证集上的表现")
plt.tight_layout()
plt.savefig("mlp_tuning_heatmap.png", dpi=150)
plt.close(fig)
print("热力图已保存: mlp_tuning_heatmap.png\n")

# 5. 绘制「训练 + 验证」损失曲线（含自己实现的早停）
#    MLPClassifier 只有 loss_curve_（训练损失），没有现成的验证损失曲线，
#    所以用 partial_fit 逐 epoch 训练，每轮记录两边的损失。
#    损失按 sklearn 的口径计算：交叉熵 + 0.5*alpha*‖W‖²/n（见 mlp_loss），
#    因此与官方的 loss_curve_ 可比。
#    早停监控的是 MONITOR_METRIC（默认 f1_macro），与调参、报告口径一致；
#    曲线仍画损失，便于直观观察是否过拟合。
mlp_probe = new_mlp(hidden_layer_sizes=best_params["hidden_layer_sizes"],
                    alpha=best_params["alpha"])
hist = fit_with_early_stopping(mlp_probe, X_tr_tfidf, y_tr, X_val_tfidf, y_val,
                               record_loss=True)
train_loss, val_loss = hist["train_loss"], hist["val_loss"]
best_epoch = hist["best_epoch"]

print(f"共训练 {hist['n_epochs']} 个 epoch（最优 epoch = {best_epoch}, "
      f"{MONITOR_METRIC}={hist['best_score']:.4f}）")
print("训练损失:", [round(v, 4) for v in train_loss])
print("验证损失:", [round(v, 4) for v in val_loss])
print("每轮监控指标:", [round(v, 4) for v in hist["score"]])
util.graph(
    train_loss,
    path="mlp_loss_curve.png",
    title="最佳超参数下MLP 训练 / 验证损失",
    xlabel="epoch",
    ylabel="log loss (交叉熵 + L2)",
    val_loss=val_loss,
    train_label="训练集 (训练子集)",
    val_label="验证集",
    mark_epoch=best_epoch,   # 竖线标出早停回退点（按 f1_macro 选出的最优轮次）
)

# 6. 用最佳超参数在「完整训练集」上重新训练
#    早停需要监控数据，而完整训练集不能再借用验证集（会信息泄漏），
#    所以这里从完整训练集里切出 VALIDATION_FRACTION 作为内部验证集，
#    其余用于训练——sklearn 的内置早停当初也是这么做的。
X_ft, X_fv, y_ft, y_fv = train_test_split(
    X_train, y_train, test_size=VALIDATION_FRACTION, random_state=42,
    shuffle=True, stratify=y_train,
)
final_vectorizer = TfidfVectorizer(max_features=5000)
final_vectorizer.fit(X_train)                    # 词表仍用全量训练集（与原脚本一致）
X_ft_tfidf = final_vectorizer.transform(X_ft)
X_fv_tfidf = final_vectorizer.transform(X_fv)
X_test_tfidf = final_vectorizer.transform(X_test_unlabeled)

mlp_model = new_mlp(hidden_layer_sizes=best_params["hidden_layer_sizes"],
                    alpha=best_params["alpha"])
final_hist = fit_with_early_stopping(mlp_model, X_ft_tfidf, y_ft, X_fv_tfidf, y_fv)
print(f"\n[重训] 训练 {len(X_ft)} 条 / 内部验证 {len(X_fv)} 条；"
      f"早停于 {final_hist['n_epochs']} 轮"
      f"（最优 epoch={final_hist['best_epoch']}, "
      f"{MONITOR_METRIC}={final_hist['best_score']:.4f}）")

# 7. 对测试集进行预测并保存结果（不覆盖 svm.py 的 predictions.csv）
predictions = mlp_model.predict(X_test_tfidf)
pd.DataFrame(predictions).to_csv("predictions_mlp.csv", index=False, header=False)
print(f"\n预测结果已保存: predictions_mlp.csv ({len(predictions)} 条)")

# 7.1 输出「验证集」上的指标（最终版本重训阶段）
#     说明：`mlp_model` 是在「完整训练集」上重训的，而完整训练集包含验证集，
#     直接用它评估验证集会信息泄漏。所以这里用只见过训练子集、且权重已回退到
#     最优轮次的 `mlp_probe`（第 5 节）在验证集上评估。
#     注意：`mlp_probe` 的最优轮次是在验证集上选出来的，故这两个数略偏乐观。
val_pred = mlp_probe.predict(X_val_tfidf)
print(f"\n[最终模型] 验证集 accuracy = {accuracy_score(y_val, val_pred):.4f}   "
      f"(验证集 f1_macro = {f1_score(y_val, val_pred, average='macro'):.4f})")
