# 当代人工智能实验报告：文本分类

## 一、实验目的

使用 TF-IDF 特征结合两种经典分类器（**SVM** 与 **MLP 多层感知机**）完成 10 类文本分类任务，比较二者性能，并掌握超参数调节与防止信息泄漏的基本方法。

## 二、数据与评价指标

- 数据：`train_data.csv`（7368 条，列 `text`、`target`，10 类类别均衡）、`test_data_unlabeled.csv`（2457 条，仅 `text`）。
- 划分：将训练集按 **8:2 分层抽样** 划为「训练子集 / 验证集」（`stratify=y`, `random_state=42`），验证集只用于挑选超参数。
- 评价指标：以 **`f1_macro`** 为主指标，同时参考 `accuracy`（类别均衡时两者接近）。

## 三、实验方法

1. **特征提取**：`TfidfVectorizer(max_features=5000)`。向量化器**只在训练子集上 `fit`**，验证集/测试集仅用 `transform`，避免信息泄漏。
2. **SVM**：`SVC`，多分类默认采用 one-vs-one；其目标为**合页损失 + L2 正则**，

   $$
   \min_{w,b}\ \tfrac{1}{2}\lVert w\rVert^2 + C\sum_i \max\bigl(0,\,1-y_i f(x_i)\bigr),
   $$

   $$
   a_V
   $$

   `C` 越小越容忍错分（防过拟合），越大越追求全部分对（易过拟合）。
3. **MLP**：`MLPClassifier`，`activation='relu'`、`solver='adam'`、`max_iter=300`，以交叉熵（log loss）为优化目标。
4. **损失曲线**：`SVC` 由二次规划一次性求解，没有逐轮损失可画，因此另用等价目标的 `SGDClassifier(loss='hinge')` 逐 epoch 记录合页损失；MLP 用 `partial_fit` 逐 epoch 记录训练/验证 log loss。

## 四、调参（网格搜索）

- **总体流程**：划分验证集 → 训练子集上拟合 TF-IDF → 遍历超参数网格 → 按验证集 `f1_macro` 选优 → **用最优参数在完整训练集上重训并重新拟合向量化器**。
- **SVM 网格**：`kernel ∈ {linear, rbf}` × `C ∈ {0.1, 1, 10, 100}`；`rbf` 使用默认 `gamma='scale'`（实测 `gamma='auto'` 在本任务完全失效，准确率仅约 0.10，已弃用）。
- **MLP 网格**：`hidden_layer_sizes ∈ {(50,), (100,), (50,25)}` × `alpha ∈ {0.001, 0.01, 0.1}`（`alpha` 为 L2 正则强度）。
- **早停**：MLP 采用 `early_stopping=True`（`n_iter_no_change=10`, `tol=1e-4`），验证损失连续多轮不改善即停止并回退到最优权重，防止过拟合。
- **可复现性**：统一固定 `random_state=42` 与 `np.random.seed(42)`。

## 五、实验结果

**超参数搜索结果（验证集）**

| 模型        | 最佳超参数                     | val f1_macro     |
| ----------- | ------------------------------ | ---------------- |
| SVM         | `kernel=rbf, C=100`          | **0.9154** |
| SVM（次优） | `kernel=linear, C=100`       | 0.9112           |
| MLP         | `hidden=(100,), alpha=0.001` | **0.9177** |

- **SVM**：`linear` 与 `rbf` 两个核均随 `C` 增大而提升；`rbf` 在极小 `C=0.1` 时严重欠拟合（f1≈0.70）。
- **MLP**：单隐藏层 `(100,)` 且弱正则（`alpha=0.001`）表现最佳，略优于 SVM。
- **损失曲线**：SGD 合页损失随 epoch 平稳下降并收敛（0.0854 → 0.0251）；MLP 训练/验证 log loss 曲线结合早停，回退到验证损失最优的 epoch。
- 生成图：`svm_tuning_heatmap.png`、`svm_loss_curve.png`、`mlp_tuning_heatmap.png`、`mlp_loss_curve.png`。

## 六、结论

TF-IDF + SVM 即可达到约 0.91 的 macro-F1，MLP 略优；**向量化器只在训练数据上拟合**、**用独立验证集选参**是获得稳定、可复现结果的关键。
