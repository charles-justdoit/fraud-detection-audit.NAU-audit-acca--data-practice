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

# 三.数据可视化
# 该notebook主要分为数据可视化(分别结合CDA2级教材第5章数据可视化，主要分为matplotlib库和seaborn库)
#导入必备的库
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

train_df=pd.read_csv(r"C:\Users\范彬\OneDrive\桌面\train.csv")  #同上文引用train_dict变量
# 结合CDA2级教程5.2描述性统计分析思维进行数据探索
# 这里思路借鉴了AUro15kaggle上公开笔记本中的探索分析并加以修改,这里进行风险评估模型相关的风险系数计算
# 一.对主属性(is_fraud)进行单独分析
# 1. 准备数据
labels = ['amt', 'fraud'] # List format for plotting
colors = ['#2ecc71', '#e74c3c']
fraud_counts=train_df["is_fraud"].value_counts()#设置风险统计数
fraud_rate=train_df["is_fraud"].mean()#设置风险评估系数
# 2. 设置分窗图布局
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 左边图表:条形图
bars = axes[0].bar(labels, fraud_counts, color=colors, edgecolor='red')
axes[0].set_title('Transaction Distribution', fontweight='bold', fontsize=12)
axes[0].bar_label(bars, padding=3, fontweight='bold')
axes[0].set_ylabel('Count')

# 右边图表:饼图
axes[1].pie(fraud_counts, labels=labels, autopct='%1.3f%%', colors=colors,
            startangle=70, explode=(0, 0.2), shadow=True)
axes[1].set_title('Fraud Percentage', fontweight='bold')

plt.tight_layout()
plt.show()

print(f"风险评估系数为 {fraud_rate:.2%}")

#接下来根据amt和is_fraud变量结合CDA2级第5章2个数据库(matplotlib库及seaborn库进行双变量关系分析)
# 二.对可能影响主属性的相关变量进行相关性分析,分为amt(金额)和gender(性别)对is_fraud影响和amt与gender相关性
#首先使用5.1.1matplotlib库中散点图进行相关性数据可视化
#相关图表属性配置
plt.rcParams['font.size'] = 12#调节字体大小
plt.rcParams["font.sans-serif"] = ['SimHei']#显示中文
plt.rcParams['axes.unicode_minus'] = False#避免符号显示干扰
plt.figure(figsize=(12, 8))#设置图表布局
#变量相关性数据可视化分析,下面使用matplotlib中散点图,seaborn库中的联合图，带核密度的联合图进行单独分析

#这里使用matplotlib库中的散点图
#先探索amt(金额)和is_fraud(欺诈值)关系_
plt.scatter(train_df["is_fraud"],train_df["amt"])
plt.show()
#接下来探索gender(性别)和is_fraud(欺诈值)关系_
plt.scatter(train_df["is_fraud"],train_df["gender"])
plt.show()
#这时候发现两者关系不够明显，优化环境同时分析gender和amt关系,因为amt和is_fraud明显呈现反比例关系
plt.scatter(train_df["amt"],train_df["gender"])
plt.show()


#结果很明显,没有明显的数据可视化效果,于是转向seaborn库中的联合图
sns.jointplot(x="is_fraud",y="amt",data=train_df)
plt.show()
sns.jointplot(x="is_fraud",y="gender",data=train_df)
plt.show()
sns.jointplot(x="amt",y="gender",data=train_df)
plt.show()
#可视化效果得到有效提高，但明显amt和gender有更好的可视化效果

#接下来针对amt和gender进行添加带核密度估计的联合分布图
sns.displot(data=train_df,x="gender",y="amt")
plt.show()

#接下来观察变量分布情况
#密度曲线图图再次查看amt变量的分布(结合CDA5.1。2seaborn库内容)
#鉴于amt分布较为不均,因此使用取对数法进行优化
# train_df["log_amt"]=np.log(train_df["amt"])
# plt.figure(figsize=(12, 8))
# sns.displot(data=train_df,x="is_fraud",y="log_amt",color="#2c3e50",kind="kde")
# plt.title("交易金额对数密度曲线分布图")
# plt.show()
#由于取对法和密度曲线图使用算力和时间较长,此处以注释仅提供思路和学生个人想法,个人本地环境无法生成






