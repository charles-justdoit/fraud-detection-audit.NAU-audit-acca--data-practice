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

# 四。逻辑回归分析(探索性分析)
# 该部分主要分为数据清洗或数据预处理(分别结合CDA2级教材第8章8.5逻辑回归内容)
# 导入必备库(结合CDA2级教程相关框架及思路)
from sklearn.preprocessing import StandardScaler #此处新加为解决对本数据集中连续变量进行方差膨胀因子的计算报错问题,查询资料并结合CDA2级6.3.7连续变量中心标准化思路使用特征标准化
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
#读取数据集,测试集和训练集导入(由于kaggle数据集已经处理并分类,此处导入即可)
train=pd.read_csv(r"C:\Users\范彬\OneDrive\桌面\train.csv")
test=pd.read_csv(r"C:\Users\范彬\OneDrive\桌面\test.csv")#此处为个人数据集存放地址
print('训练集样本量 : %i \n 测试集样本量 ： %i'  %(len(train),len(test)))#个人进行数据集(训练集,测试集)体量统计
#首先,结合CDA2级教材思路,个人先对amt(金额)变量进行一元论逻辑回归模型建模(使用训练集)
lg=smf.logit("is_fraud ~ amt",train).fit()
lg.summary()
#结合CDA2级教程思路及一元逻辑回归模型结果,做出如下分析结论:
# 该模型生成信息分为二个部分,第一部分为模型的基本信息,第二部分是模型的参数估计与检验。
# 1.amt(金额)的系数为0.0028,而且p值显示系数不显著,但由于amt(金额)波动幅度相对较大,不能说明amt(金额)对欺诈影响不明显,仅显示影响幅度一般
# 2.结合CDA2级教程思路,计算OR值得e**0.0028=1.0028,即说明发生比的比值大于1，amt(金额每参加1个单位后欺诈率的可能性是原来的1.0028倍)
#结合CDA2级教程思路,多元逻辑回归中变量筛选阶段方法分为:向前回归法,向后回归法以及逐步回归法,此处借鉴书上使用的AIC准则进行的向前回归法变量筛选
# 自定义函数(复现CDA二级第八章8.5逻辑回归P219页)如下:
def forward_select(data,response):
    remaining=set(data.columns)
    remaining.remove(response)
    selected=[]
    current_score,beat_new_score=float("inf"),float("inf")
    while remaining:
        aic_with_candidates=[]
        for candidate in remaining:
            formula="{}~{}".format(response,"+".join(selected+[candidate]))
            aic=smf.logit(formula=formula,data=data).fit().aic
            aic_with_candidates.append((aic,candidate))
        aic_with_candidates.sort(reverse=True)
        best_new_score,best_candidate=aic_with_candidates.pop()
        if current_score>beat_new_score:
            remaining.remove(best_candidate)
            selected.append(best_candidate)
            current_score=best_new_score
            print("arc is {},continuing!".format(current_score))
        else:
            print("forward selection over!")
            break
    formula="{}~{}".format(response,"+".join(selected))
    print("final formula is {}".format(formula))
    model=smf.logit(formula=formula,data=data).fit()
    return model
# 以上为对CDA2级教程自定义函数的复现,但并不适用于本大体量数据集(1000000行+),以下仅供学生思路分享,大一学生学习能力有限,可能包含差错。
#接下来应用CDA2级书上使用的自定义函数即向前回归法进行对连续变量的筛选
# train_sample=train.sample(n=1000,random_state=42)#个人学生在第一次使用原数据集协方差矩阵计算,计算量过大,尝试使用取样,但出现取样is_fraud变量全部为空值，导致报错现象,故标注释化体现学生个人在学习使用CDA2级时第8章8.5.2逻辑回归模型及实现时复现线性回归法出现报错,主要原因为使用数据集体量过大及该回归模型为协方差矩阵，计算量极大,而且使用抽样法时response变量is_fraud为二分类变量,随机取样容易取空导致报错，在日后优化及版本更新过程中将使用sklearn等优化方法
#只有连续变量可以通过逐步回归进行筛选
# candidates=["amt","lat","long","merch_lat","merch_long","city_pop","trans_num","unix_time"]#沿用该数据集中的连续变量
# data_for_select=train_sample[candidates]
# lg_m1=forward_select(data=data_for_select,response="is_fraud")
# lg_m1.summary()
# print(f"原来有{len(candidates)-1}个变量")
# print(f"筛选剩下{len(lg_m1.index.tolist())}个(包含intercept 截距项)。")
# 由于数据集体量过大,使用抽样方法会导致抽取样本is_fraud类全部为0,导致直接报错

#接下来结合CDA2级教程思路,对分类变量进行显著性分析
# 补充CDA2级相关知识点，针对分类变量3种主要解决方案为:
# 1.逐一进行变量的显著性测试。
#2.使用sklearn中的决策树等方法筛选变量
# 3.使用WoE转换,通常需要进行恰当的归并或分箱。

# 接下来通过显著性测试逐一判断分类变量的显著性
class_col=["merchant","category","first","last","gender","street","city","state","zip","job"]
for i in class_col:
    tab=pd.crosstab(train[i],train["is_fraud"])
    print(i,""" p-value = %6.4f"""  %stats.chi2_contingency(tab)[1])
#得到分类变量p值均为0,属于显著，由于样本体量过大(1000000+行),本学生尝试使用抽样法进行尝试
train_sample=train.sample(n=1000,random_state=42)
for i in class_col:
    tab1=pd.crosstab(train_sample[i],train_sample["is_fraud"])
    print(i,""" p-value = %6.4f"""  %stats.chi2_contingency(tab)[1])
# 得到相同结果,在日后优化及版本更新过程中结合ai算法推荐及资料查询将使用IV值或者分箱法等优化方法

#此处第二次迭代提交时,amt(金额)变量进行一元论逻辑回归模型建模(使用训练集)被折叠,无法显示,如果查看,请查询github仓库第2次迭代记录
formula="""is_fraud~amt+C(gender)"""#此处同样进行了高基数变量的减少,由于本地环境报错及关于线性
# lg_m=smf.logit(formula=formula,data=train_sample).fit(method="bfgs",maxiter=100)#此处因报错查询使用拟牛顿法进行优化,但结果出现大量NAN值,经查询,属于准完全分离,故查询资料优化,尝试进行合并低频变量category
# cat_counts=train_sample["category"].value_counts()
# small_cats=cat_counts[cat_counts<50].index
# train_sample["category"]=train_sample["category"].replace(small_cats,"other")
# 由于处理无效,故删除大部分变量(包括category),仅保留amt和gender因变量
lg_m=smf.logit(formula=formula,data=train_sample).fit(method="bfgs",maxiter=300)
lg_m.summary()
# 此次模型拟合成功,但模型解释力不足,Pseudo R-squ=0.116,由于gernder变量的p值为0.528，大于0.5,故接下来移除gender变量并逐步增加变量
# 删除gender变量
formula1="""is_fraud~amt"""
lg_m1=smf.logit(formula=formula1,data=train_sample).fit(method="bfgs",maxiter=300)
lg_m1.summary()
# 得到模型单amt变量p值变大,为0.147，Pseudo R-squ=0.1116
# 接下来逐步增加变量
formula1="""is_fraud~amt+C(state)"""
lg_m=smf.logit(formula=formula1,data=train_sample).fit(method="bfgs",maxiter=300)
lg_m.summary()
# 出现大量NAN值,Pseudo R-squ=0.7826，说明state变量是显著性变量
# 此处沿用上面方法,单独使用category变量
formula2="""is_fraud~amt+C(category)"""
lg_m=smf.logit(formula=formula2,data=train_sample).fit(method="bfgs",maxiter=300)
lg_m.summary()
# 出现大量NAN值,Pseudo R-squ=0.7445，说明category变量是显著性变量
# 接下来尝试将state和category变量结合进行拟合
formula3="""is_fraud~amt+C(state)+C(category)"""
lg_m=smf.logit(formula=formula3,data=train_sample).fit(method="bfgs",maxiter=300)
lg_m.summary()
# 出现大量NAN值,Pseudo R-squ=0.9123，说明category和state变量对is_fraud变量是显著性影响变量,但会发现shopping_net,shopping_pops的coef值极高,属于极端值,经查询资料,本次出现了准分离问题，日后需要进行相关极端值处理。(本部分结合CDA2级教程p221页,但由于出现了大量NAN值,经查询,属于完全准分离暂时忽略,优先考虑Pseudo R-squ)

# 结合CDA二级教程思路及框架(具体为8.5.2逻辑回归模型最后部分中),接下来针对自变量的多重共线性使用方差膨胀因子进行分析
# 方差膨胀因子检验(自定义函数借鉴引用CDA2级教程P222,个人进行相关注释及原理补充)
def vif(df,col_i):
    from statsmodels.formula.api import ols
    cols=list(df.columns)#取出所有类名
    cols.remove(col_i)#移除要计算VIF的变量
    cols_noti=cols#剩余类作为自变量
    formul=col_i+"~"+"+".join(cols_noti)#构建回归公式 当前类~其他列
    r2=ols(formul,df).fit().rsquared##拟合线性回归并获取R**2
    return 1/(1-r2)#代入公式VIF=1/(1-R**2),计算VIF
# 经查询资料,补充方差膨胀因子(VIF)压原理:
# 将每个自变量依次作为因变量,对其他自变量作线性回归,得到R**2,再用公式VIF=1/(1-R**2)进行计算
# VIF阀值参考:
# VIF<5 :共线性极弱,可忽略
# 5<=VIF<=10:存在一定共线性,需要关注
# VIF》10:存在严重共线性,必须处理(如删除变量,PCA降维)

# 继续根据CDA2级教程思路运用自定义函数,对本数据集中连续变量进行方差膨胀因子的计算
# 定义连续变量列表
candidates=["amt","lat","long","merch_long","merch_lat","city_pop"]
# 提取列表目标变量,删除目标变量is_fraud
# exog=train[candidates].drop(["is-fraud"],axis=1)
# 第一次运行上行,由于is_fraud列本身不在category里,故出现报错,经查询资料后进行相关修改
exog=train[candidates]
for i in exog.columns:
    print(i,"/t",vif(df=exog,col_i=i))
# 运行后出现了大量报错提醒,查询资料并结合CDA2级6.3.7连续变量中心标准化思路使用特征标准化
scaler=StandardScaler()
exog_scaled=pd.DataFrame(scaler.fit_transform(exog.values),columns=exog.columns)
# 再次计算VIF(方差膨胀因子)
print(i,"/t",vif(df=exog,col_i=i))
# 根据生成的分析结果及查询资料得:
# lat，long,merch_long,merch_lat值过大,说明具有严重的多重共线性问题
# amt,city_pop的VIF值小于5,共线性极小,可忽略

#根据CDA2级教材思路(此处复现CDA2级教程框架p223),接下来使用predict函数进行预测
train["proba"]=lg_m.predict(exog_scaled)
test["proba"]=lg_m.predict(exog_scaled)
test[["is_fraud","proba"]].head()
# 此处报错,与原formula4中state变量冲突,此处暂时搁置,后来版本进行优化更新

#%%

""" 该notebook主要分为用户分群(分别结合CDA2级教材第9章9.4.3层次聚类应用及9.5聚类算法相关内容)
导入必备库(结合CDA2级教程相关框架及思路)"""
import pandas as pd
from sklearn import preprocessing
from sklearn.decomposition import PCA

model_data=pd.read_csv(r"C:\Users\范彬\OneDrive\桌面\train.csv",encoding='gbk')
model_data.head()

""" 直接运行出现数据类型不匹配问题报错,故经查询资料进行更改"""
print(model_data.dtypes)
model_data2=model_data.select_dtypes(include=["number"])#只选择数值类
"""结合CDA2级9.4.3节P256思路,在主成分分析前使用sklearn库的进行变量中心标准化"""
model_data=preprocessing.scale(model_data2)
"""结合CDA2级7.3.4节P153~P157思路,进行主成分分析"""
pca=PCA(n_components=5)
newData=pca.fit(model_data)
print(pca.explained_variance_)#显示主成分方差
print(pca.explained_variance_ratio_)#显示主成分方差比
""" 由生成结果显示:前3个主成分方差占比大,占比近63%，第一个主成分占比最大,近2.9，前5个主成分总共解释了近83.3%的信息，但系统提醒有较大的极大值影响相关结果."""

""" 结合CDA2级9.4.3节P256~P257思路及框架,用fa_kit框架进行主成分分析和因子分析
# 导入库"""
from fa_kit import FactorAnalysis
from  fa_kit import plotting as fa_plotting
fa=FactorAnalysis.load_data_samples(
    model_data,preproc_demean=True,preproc_scale=True
)
fa.extract_components()
"""设定提取主成分的方式,这里使用CDA2级教程推荐方法"top_n"法"""
fa.find_comps_to_retain(method="top_n",num_keep=2)
""" 补充CDA2级教程相关知识点:因子分析法的优势是通过旋转和拉伸让系数极端化,便于理解各变量重要性
# 通过最大方差法进行因子旋转"""
fa.rotate_components(method="varimax")
""" 查看各因子在在各变量下的权重"""
print(pd.DataFrame(fa.comps["rot"]))
"""结合CDA2级教程9.4.3层次聚类应用示例思路及框架,可得出相关结论如下:
# 根据前面的主成分分析结果,设置了2个因子,在进行最大方差的因子旋转后称为因子0和因子1
# 显然,原始变量3，4，5，8，9重要性大,具体可分为2种情况:1.因子0权重大，因子1权重小 2.因子1权重大，因子0权重小 具体如下:
# 1.因子0权重大，因子1权重小类型:原始变量3,5,9
# 2.因子1权重大，因子0权重小类型:原始变量4,8"""


"""根据CDA2级教程P257指出并加以借鉴修改,对于因子0,主要反映除第二个变量以外的其他变量的信息,所以可以将因子0称为欺诈总量因子,同理根据因子系数的解释,将其命名为Fraud_Avg
# 结合CDA2级教程9.4.3层次聚类应用示例思路及框架,接下来通过这两个因子来展示原始数据的维度,并获得因子得分
# 输出因子得分,为方便拼接,转化 成数据框"""
fa_scores=fa.get_component_scores(model_data)
fa_scores=pd.DataFrame(fa_scores,columns=["Fraud_Gross","Fraud_Avg"])
print(fa_scores)

""" 结合CDA2级教程9.4.3层次聚类应用示例思路及框架P259，继续使用教程方法中scipy库进行层次聚类"""
import scipy.cluster.hierarchy as sch
import numpy as np

"""生成点与点的距离矩阵,这里使用欧式距离
disMat=sch.distance.pdist(sampled_df[["Fraud_Gross","Fraud_Avg"]],"euclidean")#教程上使用的citi10_fa,这里直接使用fa_scores
# # 进行层次聚类
# F=sch.linkage(disMat,method="ward")
# # 将层次聚类结果用树状图展示出来"""


"""由于样本集体量较大(超过1000000行),故经过查询资料使用sample抽样法(与02_eda可视化操作相同)
# sampled_df=model_data.sample(n=595,random_state=42)#此处已报错,出现AttributeError,model_data属于numpy数据类型,sample方法仅适用pandas数据类型,查询资料进行优化
# sampled_df=np.random.choice(model_data,size=595,replace=False)#此处再次报错,出现a must be 1-dimensional报错,查询资料进行优化
#再次出现报错,fa_score使用计算体量过大,占用6.12TB,故接下来经过查询资料,先提取出索引,再重新进行抽样
# 生成model_data的索引"""
indices=np.arange(len(model_data))
"""对索引进行随机抽样"""
sampled_indices=np.random.choice(indices, size=595, replace=False)
"""根据抽样索引提取"""
sampled_df=model_data[sampled_indices]

fa_scores2=fa.get_component_scores(sampled_df)
fa_scores2=pd.DataFrame(fa_scores2,columns=["Fraud_Gross","Fraud_Avg"])
print(fa_scores2.head())
""" 再次进行尝试,并将fa_score优化成抽样后的fa_score2"""
disMat=sch.distance.pdist(fa_scores2[["Fraud_Gross","Fraud_Avg"]],"euclidean")
"""进行层次聚类"""
F=sch.linkage(disMat,method="ward")
"""将层次聚类结果用树状图表示出来"""
P=sch.dendrogram(F,labels=sampled_indices)


#%%
"""结合CDA2级教程9.5聚类算法思路及框架,先补充相关知识点如下:
# K_maens聚类算法主要分为四个步骤:
# (1)设定k值,确定聚类数,软件随机分配聚类中心所需的种子
# (2)计算每个记录到类中心的距离
# (3)把k值中心作为新的中心,重新计算距离
# (4)迭代至达到标准为止
# K_means算法主要用于用户细分,用户分群部分"""

""" 根据CDA2级教程P260~P269,K-mean聚类算法实操步骤分为项目目的，数据解读，数据预处理，建立聚类模型
# 首先进行数据解读,但由于数据体量极大(TB级),且同为聚类方法,故沿用sample_df作为抽样处理后的数据
# 一.项目目的:在企业内部及相关流程现实场景检索并稽查出欺诈人群及舞弊行为
# 二.数据解读:
直接引用sample_df"""
# indices=["is_fraud","amt","lat","long","city_pop","unix_time","zip"]
# data=sampled_df[indices]
# data
# 报错,出现了indexerror,先查询sample_df 的列名
# print(sampled_indices.columns)
# 报错,发现sampled_index为numpy数据,无columns属性
# print(sampled_indices)
# 检查原先代码,发现model_data已被标准化,找出先前报错原因
model_dataF=pd.read_csv(r"C:\Users\范彬\OneDrive\桌面\train.csv",encoding='gbk')
sampled_dataF=model_dataF.sample(n=595,random_state=42)
# print(sampled_dataF.head())#此处为了简洁,故不进行可视化显示
# 用inidices进行筛选并使用规范dataframe格式
indices=["amt","lat","long","city_pop","unix_time","zip"]
sampled_dataF=pd.DataFrame(sampled_dataF,columns=indices)
sampled_dataF.head()

""" 三.数据预处理
# 结合CDA2级教程思路及框架,补充P262相关知识点:
# 商业聚类算法对变量的基本要求是尽可能反映不同方面的信息,使用聚类算法通常要求样本内无缺失值
# 由于本项目数据集为kaggle冷门公开去敏数据集，查询地址URl:https://www.kaggle.com/datasets/kaushalnandania/credit-card-fraud-detection
# 其本身已进行过数据清洗和数据预处理,故本样本(数据集)并无缺失值的情况
# 结合CDA2级9.5.2的教程思路及框架,接下来查看变量的相关系数矩阵,以判断做变量降维的必要性(非必须)"""
corr_matrix=sampled_dataF.corr(method="pearson")#使用皮尔逊系数 即corr_matrix=corr_matrix.abs()
print(corr_matrix)

"""结合相关资料及皮尔逊系数相关系数矩阵的查询,可以得出以下结论:
long(经度)与zip(邮政编码)的相关系数为-0.903880,说明两者高度线性负相关
lat(纬度)与city_pop(城市人口)的相关系数为-0.159577,lat与zip为-0.1269393,说明维度与城市人口,邮政编码呈现中等程度负相关
多数变量间的相关系数绝对值较小(如amt与其他变量的多在0.03以内),说明这些变量的线性相关性较弱
结合教程结论,相关系数最高的约为0.9,处于必须要做的范围,并且为了解释聚类方便,选择降维处理"""



"""接下来根据CDA2级中的思路及框架,使用sklearn框架做主成分分析,与先前的标准化操作相同,本次对抽样后样本sampled_dataF进行操作"""
"""from sklearn import preprocessing#先前已经引入相关库,此处注释化，为了显示区别,使用head方法分别显示中心标准化前后变化"""
print(sampled_dataF.head())#原数据集前5行
processed_data=preprocessing.scale(sampled_dataF)
"""print(processed_data.head())#中心标准化的数据集前5行"""
print(processed_data)#中心标准化的数据集前5行
"""proceesed_data属于numpy格式，出现引用的小报错,由于样本体量过大,系统就显示头尾的3行"""

# 接下来根据CDA2级中的思路及框架,使用方差比来计算并评估并保留多少个主成分合适
from sklearn.decomposition import PCA#引入必备的库
"""此处引用CDA2级教程第9章9.5.2 K-means聚类算法应用:用户分群P263相关注释:
说明1:第一次的n_components 参数应该大一点
说明2:观察explained_variance_ratio_累计大于0.85,explained_variance_需要保留的最后一个主成分大于0.8
"""
pca=PCA(n_components=3)
data=pca.fit(processed_data)
print(pca.explained_variance_)
print(pca.explained_variance_ratio_)
# 结果显示,三个主成分之和大约为0.698,大约能解释近70%的数据,但由于小于80%,故将主成分调整至4
print(f"三个主成分之和为{sum(pca.explained_variance_ratio_)}，大约为70%,小于CDA2级教程要求的80%,故将n_components调整至4")

pca1=PCA(n_components=4)
data1=pca1.fit(processed_data)
print(pca1.explained_variance_)
print(pca1.explained_variance_ratio_)
print(f"三个主成分之和为{sum(pca1.explained_variance_ratio_)}，大约为85%,满足CDA2级教程要求的80%,故n_components保留4最合适")
""" 结果显示,四个主成分之和大约为0.854,大约能解释近85%的数据,根据CDA2级9.5.2的思路及框架,得出如下结论:
# 四个主成分能够解释抽样后数据(processed_data)80%以上的变异,所以选用4个主成分适合.接下来查看各主成分在原始变量的权重.如系数极端而且不易于业务解释,需进行因子分析
# 此处数据预处理和主成分,因子分析与先前的聚类方法(fa-kit方法)属于2种,此处使用k-means聚类算法,两种方法流程上有重复处，但均严格按照CDA2级教程思路及框架"""



#%%
"""接下来根据CDA2级教程9.5.2聚类算法应用相关内容，使用k-means算法进行第二次聚类尝试中,
在计算出方差比并确定合适的主成分(n_components)后进行因子分析"""
print(pd.DataFrame(pca1.components_).T)