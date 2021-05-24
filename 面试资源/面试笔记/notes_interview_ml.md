@[toc]
# lr/Logistic Regression
## 为什么lr估计参数的时候要用对数似然函数？
参数估计的目的在于，让模型贴合实际数据，也即让预测的所有概率最大。这里的所有概率，其实就是所有样本概率的乘积，也即需要最大化$P = \prod_{i=1}^np(x_i)$

对二分类的lr来说，y有2种取值，0或者1。 用p表示y=1的概率，则1-p就是y=0的概率，这样为了直接写统一，可以用$p^y*(1-p)^{1-y}$来表示当前y的概率（y=1时概率为p，y=0时概率为1-p），再对每个概率连乘就有：
$$
P = \prod_{i=1}^n{p_i^{y_i}*(1-p_i)^{1-y_i}} = L(y) = L(\theta x)
$$
所以最大化的函数和似然函数时基本相同的

可以使用其他的损失函数，比如MSE等，但是此时函数不是凸函数，优化很困难

## lr的损失函数可以使用MSE吗？
可以使用，但是有很多问题。

1. 使用MSE时，会出现梯度消失的问题。当使用MSE作为损失函数时，梯度变为：
$$\begin{aligned}
\frac{\partial{L}}{\partial{w}} &= [y - \sigma(wx)] \sigma'(wx) w \\
&= [y - \sigma(wx)] \sigma(wx)[1 - \sigma(wx)] w
\end{aligned}$$
根据$\sigma(x)$函数的特性（可见[这里](https://blog.csdn.net/sinat_41679123/article/details/107144355)），可知只要$wx \notin [-3, 3]$，则$\sigma'(wx)$就会接近0，这样会导致梯度特别小（梯度消失）从而很难优化。
2. 使用MSE的话，此时的损失函数为：$L = \frac{1}{2}(y - \sigma(wx))^2$，是一个非凸函数，很容易求出局部最优解，而无法求出全局最优解。

# svm
## svm对缺失的数据敏感吗？为什么？
敏感。

理由：
1. 根据svm的公式，svm实际上考察的是点到分隔平面的距离。距离计算包括了平方，所以对缺失值很敏感。
2. svm内部机制中没有对缺失值的处理


## svm不同核函数有什么区别？
高斯核函数通常用于非线性分布的数据，线性核用于线性分布的数据
## svm为什么要把原始问题转换为对偶问题？
1. 对偶问题求解更加高效
	* 把问题转换为求解$\alpha$
		* 只有支持向量对应的$\alpha$才非0
		* $\alpha$只有N个（N是样本点的个数）
2. 更方便引入核函数

## SVM是用的是哪个库？Sklearn/libsvm中的SVM都有什么参数可以调节？
主要用的sklearn.svm.svc，这个库也是基于libsvm实现的。主要调节的参数有下面几个：

parameter | default | 效果 | 备注
---|---|---|---|
C| 1.0 |松弛变量前面那个C。C大，更加惩罚松弛变量，松弛变量倾向变小，分得更准，泛化能力更差|正则化系数，平方L2正则化
kernel| rbf | |核函数
degree| 3 | | 当kernel选择poly时有用，是多项式核函数$K(x,z) = (x \cdot z + 1)^p$的那个p
gamma | scale | | 核函数的参数，默认是scale，即1 / n_feature * X.var()，其中$X.var()$是x的方差
coef0|0.0||核函数的常数项，只当kernel = 'poly' or 'sigmoid'时有用

## svm如何选择核函数？
根据特征维数和样本数决定。

* 特征维数d > 样本数m时（文本分类中常见）：线性核
* 特征维数d < 样本数m，
	* 样本数m不大：RBF核
	* 样本数m很大：建议直接使用深度模型

## svm能用MSE作为损失函数吗？
如果svm用作分类问题，可见[为什么分类问题不能用MSE作为损失函数？](https://blog.csdn.net/sinat_41679123/article/details/107120983)
如果svm用作回归问题，在线性可分时是可以的，在其他情况下则不行。在引入松弛变量的情况下，可以认为是存在异常点的线性可分。MSE有距离的计算，对异常点很敏感。（==在引入核的情况下？？==）

# 树类
## gbdt常调的参数有哪些？/gbdt怎样调参？
<a id = 'gbdt_tune'></a>
sklearn库下的参数
parameter|含义|备注|default
---|:---|:---|---|
n_estimators|迭代次数|过大会过拟合，过小会欠拟合。通常不会欠拟合，可以调高一些。实际调参的时候和`learning_rate`联调|100
learning_rate|学习率/步长，$y_m = y_{m-1} + \epsilon T_t(x)$里的$\epsilon$|可以先确定`learning_rate`然后用cv寻找最优`n_estimators`|0.1
subsample|样本采样率|每轮迭代时使用的样本采样率|1（使用全部样本）
max_features|特征选择时考虑的最大特征数|相当于在节点分裂时对特征采样，不考虑全部特征|None
max_depth|树的最大深度|如果过拟合，可以减小树的深度|3
min_samples_split|节点分裂最少需要的样本数|如果某个特征下的样本过少，就不再分裂了|2
min_samples_leaf|叶子节点最少的样本数|如果叶子结点上样本过少，会和兄弟节点一起被剪枝|1
min_weight_fraction_leaf|叶子节点上所有样本的权重和最小的比例|例如`min_weight_fraction_leaf = 0.3`，则如果叶子节点上所有样本的权重和之和小于0.3 * 全部样本的权重和之和，那么该节点会被剪枝。<br>这个可以用来处理正负样本比例不平衡的数据集（给少数样本增大权重即可）<br>如果在`fit`时没有给`sample_weight`参数，那么默认所有样本`weight`相同|0.0

一般gbdt调参的步骤是：
1. 总体调：确定learning_rate，找最佳的n_estimators
2. 对树调：确定max_depth，min_samples_leaf/min_weight_fraction_leaf（有权重的样本采用后者）
3. 树内调：确定min_samples_split，这里通常需要和min_samples_leaf联调
4. 针对欠/过拟合调：这一步之后就差不多了，剩下根据是否过拟合/欠拟合，再调调subsample/max_features
5. 确定最终模型：确定完上面几个参数之后，就可以减小步长，增大n_estimators（增加泛化能力+保证拟合能力）来确定最终模型了


## gbdt防止过拟合的方法？
[普适方法](#ml_fix_overfit)+树的方法 + 特定方法

**树解决过拟合方法**（仿照RF）：<a id = 'tree_fix_overfit'></a>
1. 对样本进行采样（类似bagging）。就是建树的时候，不是把所有的样本都作为输入，而是选择一个子集
2. 对特征进行采样。类似样本采样一样, 每次建树的时候，只对部分的特征进行切分。

特定方法：每轮新模型有个学习率，即$y_m = y_{m-1} + \epsilon T_t(x)$

## gbdt在训练和预测的时候都用到了步长，这两个步长一样么？
==步长和梯度下降里面的学习率一样的，在sklearn里甚至名字就叫做learning_rate==
一样的，步长就是$y_m = y_{m-1} + \epsilon T_t(x)$里面的$\epsilon$

## gbdt的步长（和梯度下降的学习率几乎完全一样）有什么用？
步长的作用是使得每次更新模型的时候，使得loss能够平稳地沿着负梯度的方向下降，不至于发生震荡。

## 怎么设置gbdt步长的大小？
两种方法，一种是按照规则策略，另外一种是当成一个参数学习

1. 规则策略
	* 固定，如$\alpha = 1e-3$
	* 初始化后，随迭代次数减小 
2. 当成需要学习的参数

## gbdt的步长（梯度下降的学习率同）太大/太小有什么影响？
**过大**：在训练的时候容易发生震荡，从而导致模型无法收敛
**过小**：收敛速度过慢

##  xgb和gbdt的区别？
区别 | xgb | gbdt
---|---|---|
基分类器|CART树，线性分类器（此时XGBoost相当于带L1和L2正则化项的Logistic回归（分类问题）或者线性回归（回归问题））| 回归树
导数信息|二阶泰勒展开，支持自定义损失函数，只要损失函数一阶、二阶可导 | 负梯度
正则项|目标函数加了正则项， 相当于预剪枝，使得学习出来的模型更加不容易过拟合|没有（起码sklearn中没有）
列抽样|XGBoost支持列采样，与随机森林类似，用于防止过拟合|没有
缺失值处理|如果某个样本该特征值缺失，会将其划入右分支|需要在数据预处理部分处理
并行化（仅特征纬度）|预先将每个特征按特征值排好序，存储为块结构，分裂结点时可以采用多线程并行查找每个特征的最佳分割点，极大提升训练速度|没有


## xgb常调的参数有哪些？
基本和[gbdt调参](#gbdt_tune)的思路相同，只不过有些特有的参数，下面是xgb的原生库
parameter|含义|备注|对应的gbdt参数
:---|:---|:---|---|
eta|学习率/步长，$y_m = y_{m-1} + \epsilon T_t(x)$里的$\epsilon$|可以先确定`learning_rate`然后用cv寻找最优`n_estimators`|learning_rate
gamma|节点分裂最小的`loss_reduction`|如果没有达到gamma，那么节点就不会分裂。过大欠拟合|min_impurity_decrease
max_depth|树的最大深度|如果过拟合，可以减小树的深度|max_depth
min_child_weight|叶子节点上所有样本的权重和的最小和|如果小于该参数，则叶子节点会被剪枝|min_weight_fraction_leaf（gbdt是ratio，xgb是具体数值）
subsample|样本采样率|每轮迭代都会sample|subsample
colsample_bytree|生成树的时候特征采样率|采样后的特征，是一棵树用到的总的特征|xgb独有
colsample_bylevel|树每一层用到的特征采样率|采样后的特征，是这棵树该层用到的总的特征|xgb独有
colsample_bynode|节点分裂时的特征采样率|采样后的特征，是这个节点分裂时采用的特征|max_features
lambda|正则化项里面的$\lambda$|过大，欠拟合|xgb独有
scale_pos_weight|控制正负样本权重的平衡|建议值为：$\frac{n\_positive}{n\_negative}$|xgb独有

注意：
`colsample_by*`这3个参数是有层级关系的。比如设置`colsample_bytree = 0.5, colsample_bylevel = 0.5, colsample_bynode = 0.5`，则若共有64个feature，单棵树只用`32`个feature，这棵树每一层只用`16`个feature，这层中每个节点分裂时，只考虑其中`8`个feature

调参步骤：
1. 总体调：确定`eta`后调`n_estimators`（==我居然在原生库里没找着这个参数，不能啊==）
2. 对树调：`max_depth, min_child_weight`
3. 树内调：`gamma`
4. 针对欠/过拟合调：`subsample, colsample_bytree, colsample_bylevel, colsample_bynode`
5. 最终模型：降低`eta`，增大`n_estimators`得到最终模型

## XGBoost为什么使用泰勒二阶展开
* 精准性：相对于GBDT的一阶泰勒展开，XGBoost采用二阶泰勒展开，可以更为精准的逼近真实的损失函数
* 可扩展性：损失函数支持自定义，只需要新的损失函数二阶可导
* 训练快速：二阶近似可以更加快速地优化

## xgb怎样处理类别不平衡数据？
* 用AUC做评估时：设置scale_pos_weight来平衡正样本和负样本的权重（增加了少数样本的权重）（eg：当正负样本比例为1:10时，scale_pos_weight可以取10）
* 用pr做评估时：设置max_delta_step为一个有限数字来帮助收敛（基模型为LR时有效）

## XGBoost为什么快？
* 二阶泰勒展开：下降快
* 分块并行：训练前每个特征按特征值进行排序并存储为Block结构，后面查找特征分割点时重复使用，并且查找分割点时，多线程并行查找每个特征的分割点
* 候选分位点：提出一个加权的分位点算法，每个特征采用常数个分位点作为候选分割点
* 对稀疏数据进行了处理：只遍历有值的样本，用来给missing value的样本确定一个默认划分方向
* CPU cache 命中优化： 使用缓存预取的方法，对每个线程分配一个连续的buffer，读取每个block中样本的梯度信息并存入连续的Buffer中。
* Block 处理优化：Block预先放入内存；Block按列进行解压缩；将Block划分到不同硬盘来提高吞吐

## XGBoost防止过拟合的方法
[普适方法](#ml_fix_overfit)+[树方法](#tree_fix_overfit)+特定方法。
树方法在这里的展现是：

* 列抽样：训练的时候只用一部分特征（不考虑剩余的block块即可）（不仅能够防止过拟合，而且还能加速训练（加速了特征选择那一步））
* 子采样：每轮计算可以不使用全部样本，使算法更加保守

特定方法：

* 目标函数添加正则项：叶子节点个数+叶子节点权重的L2正则化
* shrinkage: 可以叫学习率或步长，为了给后面的训练留出更多的学习空间。即给每次新生成的树增加系数，原先是$y_m = y_{m-1} + T_t(x)$，现在是$y_m = y_{m-1} + \epsilon T_t(x)$


## XGBoost为什么可以并行训练？
只是特征级别的并行。因为训练前，对每个特征按特征值进行排序并存储为Block结构（1个特征1个block），后面查找特征分割点时可以重复使用，并且因为每个特征都存在了不同的block中，可以用多线程并行查找每个特征的分割点。

## XGBoost如何处理缺失值？/xgb如何处理稀疏特征？
在特征k上寻找最佳 split point 时，只对该特征上有值的数据遍历，对missing的样本，分别计算将样本分配到左分支和右分支的`loss_reduction`，然后取大的那个作为节点的default方向

## XGBoost中的一棵树的停止生长条件？
* 当新引入的一次分裂所带来的增益Gain<0时，放弃当前的分裂。这是训练损失和模型结构复杂度的博弈过程
* 当树达到最大深度时，停止建树，因为树的深度太深容易出现过拟合，这里需要设置一个超参数max_depth
* 当引入一次分裂后，重新计算新生成的左、右两个叶子结点的样本权重和。如果任一个叶子结点的样本权重低于某一个阈值，也会放弃此次分裂。这涉及到一个超参数：最小样本权重和，是指如果一个叶子节点包含的样本数量太少也会放弃分裂，防止树分的太细。

## XGB如何评价特征重要性？
可以调用`get_score`查看特征重要性，这个方法有个叫`importance_type`的参数，其对应的值和意义如下：
importance_type取值|意义
---|---|
weight|the number of times a feature is used to split the data across all trees.
gain|the average gain across all splits the feature is used in.
cover|the average coverage across all splits the feature is used in.
total_gain|the total gain across all splits the feature is used in.
total_cover| the total coverage across all splits the feature is used in.

## xgb和lgb的区别？
可见[LightGBM: A Highly Efficient Gradient Boosting Decision Tree](https://blog.csdn.net/sinat_41679123/article/details/107144314)
## 为什么用xgboost/gbdt在在调参的时候把树的最大深度调成6就有很高的精度了，但是用DecisionTree/RandomForest的时候需要把树的深度调到15或更高？
xgb/gbdt是boosting的方法，方法本身能够降低偏差（因为模型拟合的是上一轮的残差），所以模型训练的时候需要降低方差防止过拟合，因此模型比较简单（很浅就可以）

dt/rf是bagging的方法，bagging的方法本身能降低方差（用不同的基训练器投票得出最终结果），所以模型训练的时候要降低偏差防止欠拟合，因此模型都比较复杂，深度很深，甚至可能树压根就不剪枝。


## rf和gbdt的区别？
**相同点**
都是由多棵树组成，最终结果依赖全部树的结果
**不同点**
树的组成：gbdt是回归树，rf可以是回归树也可以是分类树。gbdt拟合的是残差，如果用分类树的话，A类-B类没什么意义
样本：gbdt每次使用全部样本，rf每次使用随机抽样的样本
并行性：gbdt只能串行（每次拟合上一次的结果），rf可以并行生成树
泛化能力：gbdt容易过拟合，rf不容易过拟合
## ID3, C4.5, CART建树时如何处理缺失值？
**在有缺失的数据上如何做特征选择？**
在信息增益前面增加系数，表示不缺失的样本占比，即$gain(D,A) = \rho *gain(\hat{D}, A)$，其中$\hat{D}$是有该特征的数据
**选择了某个特征，如果某样本在该特征对应的数值缺失，如何划分样本？**
分到所有节点上，但是增加权重系数，权重是该特征下每个取值的样本数占总体样本数的比例

## C4.5决策树算法如何处理连续数值型属性？
根据特征值，求样本数据的切分点。本质上仍然是把连续属性变为了离散属性。

## 为什么CART不用信息增益/信息增益比作为分类问题的损失函数？
因为信息增益/信息增益比都要计算熵，熵要计算对数，计算比较慢

## 为什么bagging能保证集成后的效果不错？
1. 从泛化角度来看，集成学习可以保证融合后模型的泛化性能比较好
2. 从算法收敛角度来看，算法收敛时容易陷入局部最小值，多个分类器可以一定程度上缓解这个问题
3. 某些学习的真实假设可能不在当前学习算法所考虑的假设空间中，所以通过融合不同学习的算法，更有可能找到最优解

## 从数学角度看，为什么bagging能降低方差？
背景知识：
对2个随机变量X和Y，有：
方差：$D(X + Y) = D(X) + D(Y) + 2cov(X, Y)$
相关系数：$\rho(X, Y) = \frac{cov(X, Y)}{\sqrt{D(X)} \sqrt{D(Y)}}$


对于bagging模型，假设有n个基分类器，每个基分类器的方差为$\sigma^2$，通过投票（这里取投票方式为求所有基分类器的平均值）得到最终结果，则最终结果的方差为$D(\frac{\sum_{i = 1}^nX_i}{n}) = \rho \sigma^2 + \frac{1 - \rho}{n} \sigma^2$

因为基分类器的方差都相同，所以整理一下相关系数的公式，可以有：$cov(X_i, X_j) = \rho(X_i, X_j)\sqrt{D(X_i)} \sqrt{D(X_j)} = \rho \sigma^2$

对bagging方差的公式推导如下：
$$\begin{aligned}
D(\frac{\sum_{i = 1}^nX_i}{n}) &= \frac{1}{n^2} D(\sum_{i = 1}^nX_i) \\
&= \frac{1}{n^2}[\sum_{i = 1}^nD(X_i) + \sum_{i = 1}^n\sum_{j = i + 1}^n cov(X_i, X_j)] \\
&= \frac{1}{n^2}(n \sigma^2 + \sum_{i = 1}^n\sum_{j = i + 1}^n \rho \sigma^2) \\
&= \frac{1}{n^2}(n \sigma^2 + \frac{n(n - 1)}{2} \rho \sigma^2) \\
&= \rho \sigma^2 + \frac{1 - \rho}{n} \sigma^2 \\
\end{aligned}$$

因为相关系数$\rho$的取值范围是[-1, 0]或[0, 1]，所以$D(\frac{\sum_{i = 1}^nX_i}{n}) = \rho \sigma^2 + \frac{1 - \rho}{n} \sigma^2 < \sigma^2$，即bagging后的模型，降低了方差
# 无监督学习
## kmeans和其他聚类算法有啥优缺点？
优点：参数少（只需要调1个K）
缺点：对初始值设置敏感、对离群值/噪声敏感、只适用于数值型数据，不适合categorical类数据、无法解决非凸的数据聚类问题

# 深度学习类
## 为什么神经网络权重不能初始化为0？但是lr可以？
浅层：如果参数初始化为一样的常数，那么每个神经元的输入都是一样的。比如对于这样的网络：

![在这里插入图片描述](https://img-blog.csdnimg.cn/2020070520194569.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3NpbmF0XzQxNjc5MTIz,size_16,color_FFFFFF,t_70)
因为$w_1 = w_2 = w_3 = w_4 = w_5 = w_6$，故$h_1 = h_2$，对最终输出的$o_1, o_2$的贡献也相同，因此计算梯度时，梯度也相同。这样就导致每层虽然有很多神经元，但是实际上所有神经元的作用完全相同，就和只有1个神经元的效果一样了。

对于LR而言，因为本来就只有1层，所以挺合适的。

---

下面的内容主要参考[https://zhuanlan.zhihu.com/p/32063750](https://zhuanlan.zhihu.com/p/32063750)

对于lr而言，如果使用MSE作为损失函数，设函数方程为：$\hat{y} = \sigma(w_1x_1 + w_2x_2 + b)$，则：
$$\begin{aligned}
\frac{\partial{L}}{\partial{w_1}} &= \frac{\partial{L}}{\partial{\hat{y}}} * \frac{\partial{\hat{y}}}{\partial{\sigma}} * \frac{\partial{\sigma}}{\partial{w_1}} \\
&= (\hat{y} - y) * [\hat{y} * (1 - \hat{y})] * x_1
\end{aligned}$$
就算$w_1$初始化为0，经过1次反向传播，由上面公式可知最后和$x_1$有关了，所以$w_1$的梯度下降会改变$w_1$


如果是神经网络，反向传播可见[这里](https://blog.csdn.net/sinat_41679123/article/details/107304366)，其中MLP中计算BP的部分，假设所有$w$都初始化为0，对于：
$$\begin{aligned}
\frac{\partial{L}}{\partial{w_1}} &= \frac{\partial{L}}{\partial{o_1}} * \frac{\partial{o_1}}{\partial{z_{o_1}}} * \frac{\partial{z_{o_1}}}{\partial{h_1}} * \frac{\partial{h_1}}{\partial{z_{h_1}}} * \frac{\partial{z_{h_1}}}{\partial{w_1}} + \frac{\partial{L}}{\partial{o_2}} * \frac{\partial{o_2}}{\partial{z_{o_2}}} * \frac{\partial{z_{o_2}}}{\partial{h_2}} * \frac{\partial{h_2}}{\partial{z_{h_2}}} * \frac{\partial{z_{h_2}}}{\partial{w_2}}
\end{aligned}$$

因为
$$
\frac{\partial{h_1}}{\partial{z_{h_1}}} = h_1 * (1 - h_1)
$$
所以当$w_1, b_1$初始化为0时，$h_1 = 0$，因此损失函数对$w_1$的梯度是0。

同理损失函数对其他的参数梯度也都是0，这样就导致反向传播计算梯度下降优化时，压根不会改变参数



## 常见的优化方法有哪些？
梯度下降类的：梯度下降法、SGD（mini-batch sg）、momentum（加入动量的梯度下降）
自适应梯度下降类：adagrad、Adam
牛顿法、拟牛顿法

（很少用到的随机搜索算法）粒子群优化算法、蚁群算法、遗传算法、模拟退火

## 怎么判断函数是凸函数还是非凸函数？
凸函数的定义是：对函数中的任意两点$x_1 < x_2$，总有$\frac{f(x_1) + f(x_2)}{2} > f(\frac{x_1 + x_2}{2})$
判断条件可以是：
1. 函数存在二阶导并且为正
2. (?)多元函数的Hessian矩阵半正定则均为凸函数

## CNN中的1*1卷积核有什么用？
1. 降维/升维：当1x1卷积核的个数小于输入depth时，可以降维。比如输入是6x6x32，经过16个1x1的卷积核后，得到的是6x6x16的特征图。同样的道理，如果卷积核个数大于输入的depth，就升维了
2. 增加非线性特征：用1x1的卷积核，若不考虑channel层面，在feature层面是没有任何变化的，此时后面再接个ReLU，相当于给原图增加了非线性特征
3. 可以让网络自己去决定使用什么depth

## CNN中，如何增加输出单元的感受野？
1. 增加卷积核大小
2. 增加层数
3. 在卷积之前进行pooling
4. 空洞卷积，给卷积核中插入空洞

## RNN为什么有梯度爆炸/梯度消失问题？
深层的神经网络都有这个问题，在RNN中尤其明显。
（在前馈神经网络的反向传播中，对某个参数求导也会涉及到激活函数的导数。）

因为对于RNN来说，如果计算$w_h$的导数，因为权重共享，需要对所有的$h_t, y_t$计算

## 怎么解决梯度爆炸/梯度消失？
**解决梯度爆炸**
梯度截断：当梯度大于阈值$gate$，就$g = \frac{gate}{||g||} * g$

**解决梯度消失**
新模型：LSTM、GRU
增加残差：残差网络
BN：对每层进行归一化之后，可以让数据分布在激活函数的非饱和区域，这样激活函数的导数能稍微大一些

## 为什么Adam优化算法中，需要做偏差修正？
adam优化算法的步骤如下：
1. 初始化参数：衰减系数$\rho_1 = 0.9, \rho_2 = 0.999$，一阶变量$s_0 = 0$，二阶变量
2. 计算损失函数梯度：$g_t = \nabla_{\Theta}{J_t(\Theta)}$
3. 更新有偏一阶矩估计（一阶动量）：$s_t = \rho_1 * s_{t-1} + (1 - \rho_1) * g_t$
5. 更新有偏二阶矩估计（二阶动量）：$\gamma_t = \rho_2 * \gamma_{t - 1} + (1 - \rho_2) g_t \odot g_t$
6. 修正一阶矩估计偏差：$\hat{s_t} = \frac{s_t}{1 - \rho_1^t}$
7. 修正二阶矩估计偏差：$\hat{\gamma_t} = \frac{\gamma_t}{1 - \rho_2^t}$
8. 参数更新：$\Theta_t = \Theta_{t - 1} -  \frac{\alpha}{\sqrt{\hat{\gamma_t}} + \delta} * \hat{s_t}$

偏差修正的步骤存在的原因是，有偏的矩估计，实际上是$g_t$的加权和，而不是加权平均。把$s_t$拆开写：
$$\begin{aligned}
s_t &= \rho_1 * s_{t-1} + (1 - \rho_1) * g_t \\
&= \rho_1(\rho_1 * s_{t-2} + (1 - \rho_1) * g_{t - 1}) + (1 - \rho_1)g_t \\
&= \rho_1^2s_{t - 2} + \rho_1(1 - \rho_1)g_{t - 1} + (1 - \rho_1)g_t \\
&= \rho_1^2(\rho_1s_{t - 3} + (1 - \rho_1) g_{t - 2}) + \rho_1(1 - \rho_1)g_{t - 1} + (1 - \rho_1)g_t \\
&= \rho_1^3s_{t - 3} + \rho_1^2(1 - \rho_1)g_{t - 2} + \rho_1(1 - \rho_1)g_{t - 1} + (1 - \rho_1)g_t \\
&= \dots \\
&= \rho_1^ts_0 + (1 - \rho_1)\sum_{i = 0}^t {\rho_1^i g_{t - i}} \\
&= (1 - \rho_1)\sum_{i = 0}^t {\rho_1^i g_{t - i}} \\
&= (1 - \rho_1)(g_t + \rho_1g_{t - 1} + \rho_1^2g_{t - 2} + \dots + \rho_1^tg_0)
\end{aligned}$$

所以没做修正前，$s_t$是$g_t$的一个加权和，想要求加权平均，应该有：
$$\begin{aligned}
weighted\_avg &= \frac{g_t + \rho_1g_{t - 1} + \rho_1^2g_{t - 2} + \dots + \rho_1^tg_0}{1 + \rho_1 + \rho_1^2 + \dots + \rho_1^t} \\
&= \frac{g_t + \rho_1g_{t - 1} + \rho_1^2g_{t - 2} + \dots + \rho_1^tg_0}{\frac{1 - \rho_1^t}{1 - \rho_1}} \\
&= \frac{(1 - \rho_1)(g_t + \rho_1g_{t - 1} + \rho_1^2g_{t - 2} + \dots + \rho_1^tg_0)}{1 - \rho_1^t} \\
\end{aligned}$$

因为分子和$s_t$是一样的，所以修正时，只需要除以$1 - \rho_1^t$即可，也即：
$$
\hat{s}_t = \frac{s_t}{1 - \rho_1^t}
$$
如果不做修正，因为$\rho_1$比$\rho_2$小，所以$s_1$比$\gamma_1$大，所以$\frac{\alpha}{\sqrt{\hat{\gamma_1}} + \delta} * \hat{s_1}$会比较大，也即最开始更新梯度的时候步子会迈得很大

## dropout训练和测试有什么区别吗？
在训练时，通常开启`dropout`，在预测时关掉`dropout`，并且给参数矩阵乘个系数`1 - drop_prob`

详细原因可见[dropout](https://blog.csdn.net/sinat_41679123/article/details/107144355)

大概原因：训练时，由于会丢弃神经元，所以参数会比较大。在预测时，为了平衡这部分偏大的参数，需要乘该神经元被保留下来的概率。

## BN对每一层都加还是只对一层加？【未回答】

## cnn和全连接网络的区别？
cnn的特点：权重共享（减少了参数）、局部连接、时间/空间上的次采样


## 深度模型有什么训练技巧？【未回答】

## 对比不同的激活函数
可见[激活函数对比](https://blog.csdn.net/sinat_41679123/article/details/107144355)

# 推荐系统类
## fm模型的隐藏层参数怎么设置？【未回答】
## fm模型的特征是几维的？
原论文中的图是这样的：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200731224947821.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3NpbmF0XzQxNjc5MTIz,size_16,color_FFFFFF,t_70)
实际上这张图中的$x^{(1)}$是样本，也就是我常用的$x_i$，在公式推导中我用的$x_1$代表第1个特征，实际上是这里的蓝色方框/橘色方框的内容。所以fm中的特征维度，如果是离散特征的话，那么特征维度和离散特征的取值有关。假设离散特征$x_1$的取值有10个，那么做了one-hot编码后，$x_1$的维度就是10维

## FM固然可以用作为打分模型，但它可以用来做matching吗，如果可以，如何做？【未回答】
## item2Vec模型在业界是如何缓解冷启动的问题的？【未回答】
## 双塔模型优势在哪？【未回答】

## 深度模型到底是如何做matching的，是离线计算好结果还是实时的对网络进行前向计算？【未回答】

## DeepFM具体实现时，wide端和deep端的优化方式是一样的吗？【未回答】

## 基于Graph的推荐方法在业界的应用目前是怎样的？【未回答】

## user-based CF和item-based CF比较？
假设有`n`个用户，`m`个物品，则最坏情况下：
user-based CF计算复杂度为$o(n^2m)$
item-based CF计算复杂度为$o(m^2n)$

因为user-based CF需要计算两两user之间的相似度，所以复杂度是$o(n^2)$的，又需要对每个user，对所有物品计算，所以是$o(n^2m)$的

通常情况下，item-item相似度表可以离线计算，而user-user相似表需要在线计算。这是因为user可能会对新的物品打分，导致user-item改变，而item更新的频率低很多

同时item-based CF比user-based CF更适合解决冷启问题

# 图相关
## node2vec中，不同的p, q参数会对生成的vector有什么影响？
BFS的游走方式，训练出来的embedding，会更看重节点的结构信息，也即具有相似结构的节点embedding更相似。
DFS的游走方式，更能体现节点的“宏观”邻居，然而有过拟合的风险。
## node2vec中的负样本是怎么选择的？【未回答】

# nlp相关
## 为什么w2v向量在语义空间内有很好的数学性质，比如相加减？【未回答】

## negative sampling介绍一下
参考文献：[word2vec Explained: Deriving Mikolov et al.’s Negative-Sampling Word-Embedding Method](https://arxiv.org/pdf/1402.3722.pdf)

对`(w,c)`的词pair，用$p(D = 1 | w, c; \theta)$表示该pair出现在corpus中，$p(D = 1 | w, c; \theta)$可以用sigmoid写成：
$$p(D = 1 | w, c; \theta) = \frac{1}{1 + e^{-v_c v_w}}$$
用Negative sampling，目标函数推导为：
$$\begin{aligned}
& \argmax\limits_{\theta} \prod_{(w,c) \in D}{p(D = 1 | w, c; \theta)}  \prod_{(w,c) \in D'} p(D = 0 | w, c; \theta)  \\
=& \argmax\limits_{\theta} \prod_{(w,c) \in D} p(D = 1 | w, c; \theta) \prod_{(w,c) \in D'} [1 - p(D = 1 | w, c; \theta)] \\
=&  \argmax\limits_{\theta} \sum_{(w,c) \in D} log \;p(D = 1 | w, c; \theta)  \sum_{(w,c) \in D'} log[1 - p(D = 1 | w, c; \theta)] \\
=&  \argmax\limits_{\theta} \sum_{(w,c) \in D} log \frac{1}{1 + e^{-v_w v_c}}  \sum_{(w,c) \in D'} log[1 - \frac{1}{1 + e^{-v_w v_c}}] \\
=& \argmax\limits_{\theta} \sum_{(w,c) \in D} log \frac{1}{1 + e^{-v_w v_c}} + \sum_{(w,c) \in D'} log\frac{1}{1 + e^{v_w v_c}} \\
&= \argmax\limits_{\theta} \sum_{(w,c) \in D} log \; \sigma(v_w v_c) + \sum_{(w,c) \in D'} log \; \sigma(-v_w v_c)
\end{aligned}$$

**抽负样本的方法**
计算所有词的`noise distribution`，计算方法是：
$$
P(w) = (\frac{U(w)}{Z})^\alpha
$$
其中：
$U(w)$是`w`这个词的`uni-gram`次数，也即出现次数
$Z$是归一化因子，保证$\sum_{w \in W}\frac{U(w)}{Z} = 1$
$\alpha$是一个超参数，在w2v文章里面作者取了$\alpha = \frac{3}{4}$

计算出每个词的`noise distribution`后，再按照`noise prob`进行抽样即可。

例如：
```python3
# noise distribution
noise_dist_normalized = {
	'apple': 0.044813853132981724, 
	'bee': 0.15470428538870049, 
	'desk': 0.33785130228003507, 
	'chair': 0.4626305591982827
}
# sample
K = 10	# usually k = 4~10
np.random.choice(list(noise_dist_normalized.keys()), size = K, p = list(noise_dist_normalized.values()))
>>> array(['apple', 'chair', 'bee', 'desk', 'chair', 'bee', 'bee', 'chair',
       'desk', 'chair'], dtype='')
```
## word2vec基本原理【未回答】

## skip-gram和cbow的区别？
skip-gram是利用中心词去预测背景词
cbow是利用背景词==的平均向量==来预测中心词

skip-gram来说，
**优点**：每次都计算中心词和背景词的pair来调整向量。假设中心词是C，对于这样的文本`A,B,C,D,E`，每次会计算`(a,c), (b,c), (c,d), (c,e)`这样4个pair来调整中心词`c`的embedding，因此训练数据较少、存在较多生僻词，也可以训练得很好。
**缺点**：假设背景窗口大小为`k`，文章中有`v`个词，因为要对每个词计算pair，所以整体的计算复杂度是`kv`

对cbow来说，
**优点**：通过计算背景词，来预测中心词，以预测结果调整背景词的词向量。同样假设中心词是`c`，对于`a,b,c,d,e`这样的文本，先计算窗口内所有背景词的平均值作为背景词向量，然后调整时，因为求的是平均值，所以对窗口内所有词的调整幅度是一样的，因此对生僻词的支持不如skip-gram。
**缺点**：假设共有`v`个词，因为只需要对每个词进行预测，所以整体的计算复杂度是`v`

具体的过程见[word2vec](https://blog.csdn.net/sinat_41679123/article/details/107144314)

## 层次softmax原理【未回答】

## Transformer基本原理【未回答】
http://nlp.seas.harvard.edu/2018/04/03/attention.html）

## 手写attention【未回答】

## 手写LR、KMeans（手写GBDT predict过程【未回答】

## word2vec和node2vec有什么联系？【未回答】

## fasttext中，有了词向量，怎么做分类？
fasttext是对文本利用n-gram提取词，然后将所有词的词向量平均后得到文本向量，最后在文本向量上接一个多分类器，进行分类的。




# 其他
## 多分类模型和二分类模型的优缺点？【未回答】
## 为什么auc适合正负样本比例不均衡的模型评估？【未回答】
## 常见的缺失值处理方式有哪些？
1. 对于缺失值较多的特征，可以考虑直接丢弃，否则会带来较大的noise
2. 用数值填充
	* 0
	* 平均数（有相同label的该特征的平均值）
	* 中位数
3. 用模型学习补充缺失值，这里经常用随机森林等树模型，因为树模型对缺失值不敏感

## 模型正则化有哪些方式？
<a id = 'reg'></a>
**L1正则化**
L1正则化公式为：
$$
L_1(w) = \sum_{i = 1}^n |w_i|
$$
在0点不可导

画一下函数图就能看出来，在$w>0$和$w<0$时，L1都以恒定的速度趋向0，即便是w已经很小时，L1趋向0的速度也不会变化，这样就可能导致大量的w直接变成0，从而“消灭”稀疏的特征。

利用这个特性，可以使用L1正则化选择特征，因为比较稀疏的特征直接会变成0

**L2正则化**
公式为：
$$
L_2(w) = \frac{1}{2}\sum_{i = 1}^n w_i^2
$$
同样画图可以看出来，当w越靠近0时，函数下降的速度就会越慢，这样可以保证很小的w不会轻易变成0，从而避免特征稀疏



## 机器学习怎样解决过拟合问题？
<a id = 'ml_fix_overfit'></a>
1. 增加样本：[SMOTE](#smote)、过采样等
2. 减少特征：选择特征（去掉不重要的特征）、特征降维（PCA）
3. 调整模型：[模型正则化](#reg)、模型简单化、减少树的深度、模型融合（集成学习）

## SMOTE算法是什么？【未回答】
<a id='smote'></a>

## 深度学习怎样解决过拟合问题？
<a id = 'dl_fix_overfit'></a>
在[机器学习解决过拟合](#ml_fix_overfit)的基础上，还有：

1. dropout：每层隐藏层随机丢掉一定比例的隐藏单元，即这些hidden unit的权重不更新。
 ==为什么dropout能解决过拟合？== 
 从集成学习角度：dropout相当于训练了非常多个仅仅有部分隐层单元的神经网络，每个这种半数网络，都能够给出一个分类结果，这些结果有的是正确的，有的是错误的。随着训练的进行，大部分半数网络都能够给出正确的分类结果。那么少数的错误分类结果就不会对终于结果造成大的影响。
 ==RNN不能丢弃$h_t$，只能丢弃$x_t$的部分==
2. early-stop：当某个评估值不再提升时，就退出训练。以accuracy为例，在训练的过程中，记录到目前为止最好的validation accuracy，当连续10次Epoch（或者更多次）没达到最佳accuracy时，则可以认为accuracy不再提高了，此时stop即可。缺点：可能会欠拟合
3. batch-normalization：主要功能是为了防止输入的数据分布在过深的网络中漂移，但是也可以防止过拟合。这是因为神经网络做预测时，除了考虑样本本身的特征，还会考虑同一批次中的其他样本，用批归一化时，选取的batch是随机的，所以可以防止过拟合
4. 模型融合



## 怎样解决欠拟合问题？
1. 特征方面：增加交叉的特征
2. 模型方面：增加模型迭代的轮次、减小正则化项

## 哪些机器学习算法不需要做归一化处理？
概率模型不需要归一化，因为它们不关心变量的值，而是关心变量的分布和变量之间的条件概率。概率模型包括：决策树、RF

其他需要归一化的模型包括：Adaboost、GBDT、XGBoost、SVM、LR、KNN、KMeans之类的最优化问题就需要归一化

## 为什么L1正则先验分布是Laplace分布，L2正则先验分布是Gaussian分布？【未回答】

## 分类问题和回归问题分别使用什么损失函数？为什么？
分类问题一般使用交叉熵损失。
分类问题中的数据是离散的，所以希望能衡量模型predict出的数据分布和真实的数据分布。衡量2个概率分布的方法，通常是KL散度，也就是相对熵。记真实数据的分布为p，模型预测的数据分布为q，则KL散度为：
$$\begin{aligned}
KL(p, q) &= \sum_xp(x)log\frac{p(x)}{q(x)} \\
&= \sum_xp(x)logp(x) - \sum_xp(x)logq(x) \\
&= -H(p) + H(p, q)
\end{aligned}$$
其中$H(p)$是真实数据的概率分布的熵，$H(p, q)$是两个概率分布之间的交叉熵。因为$H(p)$是一个定值，所以`min KL(p, q)`就是最小化交叉熵，所以分类问题使用交叉熵作为损失函数。

回归问题一般使用MSE，凸函数，并且计算梯度也很方便。

## 为什么分类问题不能用MSE作为损失函数？
MSE的物理意义是求a和b之间的欧几里得距离，通常情况下，计算真实值`y`和预测值$\hat{y}$的欧几里得距离，认为数据是在一维空间下的，所以欧几里得距离就直接是两个数的差的平方再开方，即$\sqrt{(y - \hat{y})^2}$，为了计算方便，去掉根号，再加个`1 / 2`，就有MSE了：
$$
loss = \frac{1}{2} (y - \hat{y})^2
$$
而对于分类问题而言，真实值和预测值都是`0, 1`，此时计算欧几里得距离是没有任何意义的。

## 比较MSE和MAE的优缺点
·|MSE|MAE
---|---|---|
·|类似L2正则化|类似L1正则化
鲁棒性|对离群点敏感（有平方计算）|对离群点不敏感
收敛情况|收敛速度更快，并且可导|0点不可导，损失值较小时梯度不变（参考L1正则化为什么能起特征选择作用）

## 为什么要做特征离散化？
1. 做离散化之后可以减少一些异常值的影响。比如将年龄分段后，年龄300岁，就直接归到年龄>80的类别
2. 增加、减少特征更加容易
3. 方便放到模型使用
4. 方便引入非线性
