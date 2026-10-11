# my_experiment.py

# 1. 从我们提供的帮助脚本中导入加载函数
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

def load_data():
    train_df = pd.read_csv('train_data.csv')
    test_df = pd.read_csv('test_data_unlabeled.csv')
    X_train = train_df['text'].astype(str).tolist()
    y_train = train_df['target'].values
    X_test_unlabeled = test_df['text'].astype(str).tolist()
    return X_train, y_train, X_test_unlabeled

def train_valid_split(X, y, test_size=0.2, random_state=42):
    """把训练集按分层抽样的方式划分为「训练子集」和「验证集」。

    - X: 文本列表（list[str]）
    - y: 标签数组
    - test_size: 验证集所占比例，默认 0.2
    - stratify=y: 保证各折中类别比例与原始数据一致（本任务类别均衡，很重要）

    返回: X_train, X_valid, y_train, y_valid
    """
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
        stratify=y,
    )

def prepare():
    # 2. 调用函数来获取数据
    #    这个函数会自动读取 .csv 文件并返回你需要的所有内容
    X_train, y_train, X_test_unlabeled = load_data()

    # 3. (验证步骤) 检查一下数据是否加载成功
    print("--- 数据加载成功 ---")
    print(f"训练集样本数量: {len(X_train)}")
    print(f"训练集标签数量: {len(y_train)}")
    print(f"无标签测试集样本数量: {len(X_test_unlabeled)}")
    print("-" * 20)

    # 打印第一个训练样本和它的标签，感受一下数据
    print("第一个训练样本内容:")
    print(X_train[0])
    print(f"\n第一个训练样本的标签: {y_train[0]}")
    print("-" * 20)

    # 打印第一个需要你预测的测试样本
    print("第一个无标签测试样本内容:")
    print(X_test_unlabeled[0])
    print("\n" + "="*50)

# 用于绘制loss函数
def graph(loss, path='loss_curve.png', title='训练损失曲线',
          xlabel='epoch', ylabel='loss', val_loss=None,
          train_label='train loss', val_label='val loss', mark_epoch=None):
    """绘制损失随训练轮次变化的曲线。

    - loss: 每个 epoch 的训练损失值（列表或数组），长度即训练轮数
    - val_loss: 可选的验证集损失值（列表或数组，与 loss 等长）。
                给定时会在同一张图上再画一条曲线，便于观察过拟合。
    - path: 图片保存路径
    - title / xlabel / ylabel: 标题与坐标轴标签
    - train_label / val_label: 两条曲线的图例名称
    - mark_epoch: 可选的轮次，给定时画一条竖直虚线标出该点
                  （常用于标注早停时验证损失最优的 epoch）

    返回保存的图片路径。
    """
    import matplotlib
    matplotlib.use('Agg')  # 只保存文件，不弹出窗口
    import matplotlib.pyplot as plt

    # 让图中的中文能正常显示（否则会显示成方框）
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    loss = list(loss)
    epochs = list(range(1, len(loss) + 1))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(epochs, loss, marker='o', markersize=3, linewidth=1.5,
            color='#2b6cb0', label=train_label)
    if val_loss is not None:
        if len(val_loss) != len(loss):
            raise ValueError('val_loss 与 loss 长度必须一致')
        ax.plot(epochs, list(val_loss), marker='s', markersize=3, linewidth=1.5,
                color='#e53e3e', label=val_label)
    if mark_epoch is not None:
        # 标出早停时验证损失最优（即回退点）所在的轮次
        ax.axvline(mark_epoch, color='#718096', linestyle='--', linewidth=1.2,
                   label=f'best epoch = {mark_epoch}')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

    print(f'Loss 曲线已保存: {path}')
    return path
# --- 在这里开始你的实验！ ---
# 现在，你可以使用 X_train, y_train, 和 X_test_unlabeled 这三个变量
# 来进行后续的特征提取、模型训练和预测了。

# 举例：
# 1. 创建TF-IDF向量化器

# X_train, y_train, X_test_unlabeled = load_data()


# # 2. 训练一个模型...
# from sklearn.svm import SVC
# svm_model = SVC()
# svm_model.fit(X_train_tfidf, y_train)

# # 3. 对测试集进行预测...
# X_test_tfidf = vectorizer.transform(X_test_unlabeled)
# predictions = svm_model.predict(X_test_tfidf)

# # 4. 保存你的预测结果...
# import pandas as pd
# pd.DataFrame(predictions).to_csv('predictions.csv', index=False, header=False)