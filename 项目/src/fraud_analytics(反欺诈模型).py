# 该数据集来自kaggle官网Credit Card Fraud Detection URL:https://www.kaggle.com/datasets/kaushalnandania/credit-card-fraud-detection
# 一。数据加载与检查
# 导入必需库
from glob import escape
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import warnings

print("Congratulation!Everything is fine!")  #环境配置完成
warnings.filterwarnings("ignore")
sns.set_theme(style="darkgrid", font="Arial", palette="colorblind")
plt.rcParams["figure.figsize"] = (12, 8)  #配置图表尺寸
plt.rcParams["figure.dpi"] = 100  #设置图表分辨率
plt.rcParams["font.sans-serif"] = ["SimHei"]  #设置中文字体为黑体
plt.rcParams["axes.unicode_minus"] = False  #正常显示负号
train_df = pd.read_csv(r"C:\Users\范彬\OneDrive\桌面\train.csv")#个人储存在桌面,路径可个人进行修改
test_df = pd.read_csv(r"C:\Users\范彬\OneDrive\桌面\test.csv")
print(f"{train_df.shape[0]:,} rows * {train_df.shape[1]} columns")
print(f"{test_df.shape[0]:,} rows * {test_df.shape[1]} columns")

# 数据大体发布和统计汇总
train_df.head()#数据前5行
train_df.info()#数据类型相关
train_df.describe()#相关统计汇总信息
#欺诈数据发布及统计
fraud_counts=train_df["is_fraud"].value_counts()
print(f"该样本欺诈数据总数是{fraud_counts}")
fraud_rate=train_df["is_fraud"].mean()
print(f"该样本欺诈率是{fraud_rate:.2%}")

#由于kaggle来源的数据集已经去敏及数据预处理，下面清洗过程仅供参考(个人见解)
# 二.数据清洗(预处理)
# 主要分为数据清洗或数据预处理(分别结合CDA2级教材4.3数据清洗和6.3数据预处理基础)
from glob import escape#导入必备的库
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import os
from sklearn.preprocessing import MinMaxScaler

#数据读取和统计值汇总
train_df=pd.read_csv(r"C:\Users\范彬\OneDrive\桌面\train.csv")#引用地址为个人习惯地址，可自行更改
train_df.head()
train_df.info()


# 重复值处理(此处基于CDA二级教程4.3.1重复值处理)
var = train_df[train_df.duplicated()]#查询重复值
if  var is None:
    escape()
else:
    train_df.drop_duplicates()#如果有重复值，去重
    print("The duplicate rows were dropped.")
# 缺失值处理(此基于CDA二级教程4.3.2缺失值处理)
esp=train_df.apply(lambda x:sum(x.isnull())/x.size)#查询缺失值
if esp is None:
    escape()
else:
    train_df.fillna(np.nan,inplace=True)
    print("The escaping contents were dropped.")

#离群值识别并处理
#这里结合CDA2级教程6.3.3中利用自定义函数使用盖帽法实现连续变量离群值识别和处理
# 类似方法如:四分位数法及分箱法，WoE法
def blk(floor,root):
    def f(x):
        if x < floor:
            x=floor
        elif x > root:
            x=root
        return x
    return f
q1=train_df["amt"].quantile(0.01)#计算百分位数
q99=train_df["amt"].quantile(0.99)#设置上下阀值
blk_tok=blk(q1,q99)
train_df["amt"]=train_df["amt"].map(blk_tok)#map用于series中对每个元素执行百分比操作
train_df["amt"].describe()

# 连续变量中心标准化或归一化(结合CDA2级教程6.3.7连续变量中心标准化，此处使用sklearn库实现极差标准化)
num_cols=["amt","city_pop"]# 选择特征列
scaler=MinMaxScaler()#初始归一化工具
train_df[num_cols]=scaler.fit_transform(train_df[num_cols])#拟合并转换
print(train_df[num_cols].head())
# 三.探索性数据分析
labels = ['Legitimate', 'Fraud']
colors = ['#2ecc71', '#e74c3c']
fig,axes = plt.subplots(1,2,figsize=(16,6))
bars=axes[0].bar(labels,fraud_counts,color=colors,edge_color="black")




