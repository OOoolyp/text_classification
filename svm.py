from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
import util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, hinge_loss
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # 无图形界面环境下直接存图，避免 plt.show() 阻塞
import matplotlib.pyplot as plt
import seaborn as sns

# 让图中的中文能正常显示（Windows 常用中文字体）
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ── 关于结果可复现性 ─────────────────────────────────────────────
# 1) train_test_split(random_state=42)  → 固定「训练子集 / 验证集」的划分；
# 2) TfidfVectorizer 本身没有随机成分   → 同样的文本 + max_features，词表和矩阵完全一致；
# 3) SVC(random_state=42)              → 固定 libsvm 内部的随机数（多分类/概率估计等会用到）；
# 4) SGDClassifier(random_state=42)    → 固定 SGD 的权重初始化与样本遍历顺序；
# 5) 全程不依赖 PYTHONHASHSEED 之类的环境变量，换台机器结果也一致。
# （若想让 numpy 层的随机也彻底固定，可额外调用 np.random.seed(42)）
np.random.seed(42)

# 1. 加载数据
X_train, y_train, X_test_unlabeled = util.load_data()

# 2. 把原始训练集划分为「训练子集」和「验证集」（分层抽样，验证集占 20%）
#    验证集只用来挑选超参数，绝不能参与向量化器的拟合，否则会造成信息泄漏。
X_tr, X_val, y_tr, y_val = util.train_valid_split(
    X_train, y_train, test_size=0.2, random_state=42
)
print(f"训练子集: {len(X_tr)} 条, 验证集: {len(X_val)} 条, 测试集: {len(X_test_unlabeled)} 条")

# 3. 只在训练子集上拟合 TF-IDF 向量化器，再转换验证集
vectorizer = TfidfVectorizer(max_features=5000)
X_tr_tfidf = vectorizer.fit_transform(X_tr)
X_val_tfidf = vectorizer.transform(X_val)

# 4. 在验证集上比较不同超参数组合，确定最佳参数
#    已去掉 gamma：rbf 直接使用 sklearn 默认的 gamma='scale'
C_values = [0.1, 1, 10, 100]
param_grid = (
    [{"kernel": "linear", "C": c} for c in C_values]
    + [{"kernel": "rbf", "C": c} for c in C_values]
)

records = []
best_score, best_params, best_model = -1.0, None, None
for params in param_grid:
    model = SVC(random_state=42, **params)  # random_state 保证每轮训练结果可复现
    model.fit(X_tr_tfidf, y_tr)
    y_val_pred = model.predict(X_val_tfidf)
    acc = accuracy_score(y_val, y_val_pred)
    f1 = f1_score(y_val, y_val_pred, average="macro")
    records.append({**params, "acc": acc, "f1_macro": f1})
    print(f"{str(params):<35} val acc={acc:.4f}  f1_macro={f1:.4f}")
    if f1 > best_score:
        # 记住最佳模型：它只在「训练子集」上训练过，可用来无泄漏地评估验证集
        best_score, best_params, best_model = f1, params, model

results = pd.DataFrame(records)
print(f"\n最佳超参数: {best_params}  (验证集 f1_macro={best_score:.4f})\n")

# 4.1 把验证集结果绘制成热力图：行 = C，列 = kernel，数值 = 验证集 f1_macro
pivot = results.pivot(index="C", columns="kernel", values="f1_macro")

fig, ax = plt.subplots(figsize=(5.5, 5))
sns.heatmap(
    pivot,
    annot=True, fmt=".4f", cmap="YlGnBu",
    vmin=pivot.values.min(), vmax=pivot.values.max(),
    linewidths=0.5, linecolor="white",
    cbar_kws={"label": "验证集 F1 (macro)"},
    ax=ax,
)
ax.set_xlabel("kernel")
ax.set_ylabel("C")
ax.set_title("SVM 超参数在验证集上的表现")
ax.set_yticklabels([f"{v:g}" for v in pivot.index], rotation=0)
plt.tight_layout()
plt.savefig("svm_tuning_heatmap.png", dpi=150)
plt.close(fig)
print("热力图已保存: svm_tuning_heatmap.png\n")

# 5. 用最佳超参数在「完整训练集」上重新训练
#    向量化器也重新在全量训练集上拟合，以利用全部数据（含之前留出的验证集）
final_vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = final_vectorizer.fit_transform(X_train)
X_test_tfidf = final_vectorizer.transform(X_test_unlabeled)

svm_model = SVC(random_state=42, **best_params)
svm_model.fit(X_train_tfidf, y_train)

# 6. 对测试集进行预测并保存结果
predictions = svm_model.predict(X_test_tfidf)
pd.DataFrame(predictions).to_csv('predictions_svm.csv', index=False, header=False)

# 6.1 输出「验证集」上的 accuracy（最终版本重训阶段）
#     说明：`svm_model` 是在「完整训练集」上重训的，而完整训练集包含验证集，
#     直接拿它评估验证集会信息泄漏。所以这里用只见过训练子集的最佳模型
#     `best_model` 在验证集上评估，得到的才是可信的泛化估计。
val_acc = accuracy_score(y_val, best_model.predict(X_val_tfidf))
print(f"\n[最终模型] 验证集 accuracy = {val_acc:.4f}   "
      f"(验证集 f1_macro = {best_score:.4f})")

# 7. 绘制 Loss 曲线
#    为什么不能直接画 SVC 的 loss？
#    SVC 走的是 libsvm 的二次规划求解器，一次性解出最优解，
#    没有「逐轮迭代的损失」可以记录（sklearn 只为 MLPClassifier 提供 loss_curve_）。
#    所以这里用 SGD 优化的线性 SVM（loss='hinge' 即合页损失）来展示损失下降过程：
#    它优化的是与 SVM 完全相同的目标（合页损失 + L2 正则），只是用梯度下降求解。
sgd = SGDClassifier(loss="hinge", random_state=42)
train_loss, val_loss = [], []
n_epochs = 30
for epoch in range(n_epochs):
    if epoch == 0:
        sgd.partial_fit(X_tr_tfidf, y_tr, classes=np.unique(y_tr))
    else:
        sgd.partial_fit(X_tr_tfidf, y_tr)
    # 每个 epoch 结束后，分别在「训练子集」和「验证集」上计算平均多分类合页损失。
    # 注意验证集从未参与拟合，所以它的损失能真实反映泛化能力：
    # 若训练损失继续下降、验证损失却开始回升，就是过拟合的信号。
    train_loss.append(
        float(hinge_loss(y_tr, sgd.decision_function(X_tr_tfidf), labels=sgd.classes_))
    )
    val_loss.append(
        float(hinge_loss(y_val, sgd.decision_function(X_val_tfidf), labels=sgd.classes_))
    )

best_epoch = int(np.argmin(val_loss)) + 1  # 验证损失最小的轮次（模拟早停点）

print("训练集损失:", [round(v, 4) for v in train_loss])
print("验证集损失:", [round(v, 4) for v in val_loss])
print(f"验证损失最低的 epoch: {best_epoch} (val loss={min(val_loss):.4f})")

util.graph(
    train_loss,
    val_loss=val_loss,
    path="svm_loss_curve.png",
    title="最佳超参下SVM等价目标的 SGD 版训练 / 验证损失曲线",
    xlabel="epoch",
    ylabel="hinge loss",
    train_label="训练集 (训练子集)",
    val_label="验证集",
    mark_epoch=best_epoch,
)