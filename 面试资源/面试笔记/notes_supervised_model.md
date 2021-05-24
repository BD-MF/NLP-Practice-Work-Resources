@[toc]
模型的本质是找到一种数学函数，能够拟合现有数据的分布。

模型可以大概分为：
1. 线性模型
2. 非线性模型
3. 神经网络模型

# 线性模型
## lr/Logistic Regression
### 思想
以事件发生的几率考察数据分布
### 模型函数推导
LR适用于离散的、高维稀疏的特征，因为带正则化项的线性模型，不容易过拟合
#### 二项分类
记事件发生的概率为p，则：$事件发生的几率=\frac{事件发生的概率}{事件不发生的概率}=\frac{p}{1-p}$
lr认为，事件发生的几率的对数，是x的线性函数，所以有：
$$
ln\frac{p}{1-p} = wx
$$
整理可得模型函数：$p = \frac{1}{1 + e^{-wx}} = \sigma(wx)$
#### 多项分类
由上面说到的模型函数可以类比推出：若Y的取值为1～K，则：
$$\begin{aligned}
P(y = k) &= \frac{e^{w_kx}}{1 + e^{w_1x} + e^{w_2x} + \dots + e^{w_{K-1}x}} =\frac{e^{w_kx}}{1 + \sum_{k=1}^{K-1}w_kx}, k = 1, 2, \dots, K-1 \;(类比二项分类y=1)\\
P(y = K) &= \frac{1}{1 + \sum_{k=1}^{K-1}w_kx} \;(类比二项分类y=0)
\end{aligned}$$

其实上面的表达式就是softmax函数
### 参数估计/损失函数
有了以上的模型后，就需要根据实际的数据分布来估计模型的参数。在lr中参数只有1个，就是w。lr的参数估计采用极大似然估计法，通过对对数似然函数求极值来求w。 （[为什么要用对数似然函数？](https://blog.csdn.net/sinat_41679123/article/details/107120983)）
似然函数$L(\theta) = \prod_{i=1}^n p^{y_i} (1 - p)^{1 - y_i}$，因为有连乘、指数，不方便求极值，所以代替地求对数似然函数的极值。
对数似然函数$L(\theta)$：
$$\begin{aligned}
L(\theta ) &= \sum {y_i \lg {p} +(1-y_i) \lg (1-p)} \\
&= \sum y_i \lg (\frac{p}{1-p}) + \lg (1-p) \\
&= \sum y_i(w x) - \lg ( 1 + e^{w x})
\end{aligned}$$
在lr中一般用梯度下降、拟牛顿法求极值
## 最大熵模型
### 思想
最大熵原理：在满足约束条件的情况下，认为条件熵最大的模型就是最好的模型。（ ==为什么熵最大就最好？== 因为熵是不确定性的度量，在信息论中，熵越大，富含的信息越多）

熵的一些相关定义见[这里](https://blog.csdn.net/sinat_41679123/article/details/107144355)


### 模型函数推导
最大熵模型就是在满足约束条件下找熵最大的模型
**约束条件**：
对于每个特征，模型的期望必须和训练样本的期望相同，记$\tilde{p}(x)$为x出现的概率，$\tilde{p}(x,y)$为(x,y)在样本中出现的概率，则模型应满足：
$$
\sum_{x,y} \tilde{P}(x,y)f(x,y) = \sum_{x,y}\tilde{P}(x)P(y|x)f(x,y)
$$
其中：
1. P(y|x)是需要求解的模型的条件概率
2. f(x,y) 是一个特征函数，其定义为：
$$
    f(x,y)=
    \begin{cases}
      1, &  \; x与y满足某个事实 \\
      0 &  \; 否则 \\
    \end{cases}
$$

**优化目标**
熵最大（条件熵最大），即：
$H(Y|X) = -\sum_{x,y}\tilde{P}(x)P(y|x)logP(y|x)$最大

### 参数估计
求解带约束的最优化问题：
$$\begin{aligned}
max \;\;\; &H(P(Y|X)) = -\sum \tilde{P}(x)P(y|x)logP(y|x) \\
s.t. \;\;\; & \sum_{x,y}\tilde{P}(x)P(y|x)f(x,y) = \sum_{x,y}\tilde{P}(x,y)f(x,y) \\
& \sum_{x,y} P(y|x) = 1
\end{aligned}$$
引入larange算子，转换为无约束最优化的对偶问题：
$$
L(P, \omega ) = -H(P) + \omega_0[1-\sum_{x,y} P(y|x)] + \sum_{i=1}^n \omega_i[\sum_{x,y}\tilde{P}(x)P(y|x)f(x,y) - \sum_{x,y}\tilde{P}(x,y)f(x,y)]
$$
接下来对larange函数L(p,w)求极大极小值即可
求解过程略
解出来有：
$$\begin{aligned}
P_\omega(y|x) &= \frac{1}{Z_\omega} exp \big(\sum_{i=1}^n \omega_if_i(x,y) \big) \\
Z_\omega &= \sum_y exp \big( \sum_{i=1}^n \omega_if_i(x,y)\big)
\end{aligned}$$

注意上面这个形式和二项的lr模型函数非常类似，lr的模型函数是：
$$
p = \frac{e^{\omega x}}{1 + e^{\omega x}}
$$
类似地，求解以上对偶问题的极大化，等价于最大熵模型的极大似然估计（求解对数极大似然函数），证明略。
## lr和最大熵模型对比
相同点：
1. 求解模型时，都可以归为：求解以对数似然函数为目标函数的最优化问题
2. 模型函数相似，都是对数线性模型（y的对数函数是x的线性函数）

## svm
svm本质上是想寻找一个分类器，这个分类器能够把正负两类样本分开，并且分开的间隔最大。按照已有数据的分布情况，svm的推导是层层递进的。

**当数据本身就可以找到最大间隔分类面时**（线性可分、硬间隔最大化）：
由定义来看，svm是想找到一个分类平面，分类平面满足：所有样本点到分类平面的距离最大，也即样本点到分类平面的最小距离也能大于一个常数，所以模型公式为：
$$\begin{aligned}
max \;& \; d_{min} \\
s.t. \; & \; d_x \geq d_{min}
\end{aligned}$$
其中，$d_{min}$是样本点到分离平面的最小距离。点$(x_1,y_1)$到平面wx + b = 0的距离为：$d = \frac {|w * x_1 + b|} {|w|}$
记取得最小距离的x为$x_{min}$，则上述公式可以推导为：
$$\begin{aligned}
max \;& \; \frac{w * x_{min} + b}{|w|} \\
s.t. \; & \; \frac{|wx_i + b|}{|w|} \geq \frac{w * x_{min} + b}{|w|}, \forall x_i \in X
\end{aligned}$$
其中$w * x_{min} + b$是一个常数，直接化简为1，s.t.中可以同时去掉分母，得到：
$$\begin{aligned}
max \;& \; \frac{1}{|w|} \\
s.t. \; & \; |wx_i + b| \geq 1, \forall x_i \in X
\end{aligned}$$
为了后续计算方便，将求最大值改为求最小值，并且：
$$
max \;  \frac{1}{|w|} \Leftrightarrow min \; |w| \Leftrightarrow min \; \frac{1}{2} w^2
$$
为保证$|wx_i + b|$去掉绝对值后正负号不变，若真实值为$y_i$，可以写成$y_i(wx_i + b)$
故最终的求解方程为：
$$\begin{aligned}
min \;& \; \frac{1}{2} w^2 \\
s.t. \; & \; y_i(wx_i + b)  - 1 \geq 0, \forall x_i \in X
\end{aligned}$$
这是一个凸优化问题，为了求解该凸优化问题，应用拉格拉朗日对偶性，通过求解该问题的对偶问题来获得原始问题的最优解。

求解对偶问题步骤：
1. 构建拉格朗日函数
2. 对原始参数求极大值
3. 对拉格朗日函数的参数（如$\alpha$）求极小值

构建拉格朗日函数：
$$
L(w, b, \alpha) = \frac{1}{2} w^2 - \sum_i^N{\alpha_i [y_i(w x_i + b) - 1]}
$$
对w, b求极大值
对$\alpha_i$求极小值

**当数据本身找不到严格的最大间隔分类面时**（线性、软间隔最大化）
数据中有些outlier，去掉这些异常点后仍然是线性可分。所以这里对限制条件做出一些改变，并不要求所有的$d_x \geq d_{min}$，而是增加一个松弛变量，只要能满足$(w * x + b ) \geq 1 - \epsilon$ 即可。同时要保证该松弛变量$\epsilon$ 不能太大，所以在最小化目标里也把它加进去。
具体的推导过程和线性可分几乎完全一样，这里的公式变为：
$$\begin{aligned}
min \;& \; \frac{1}{2} w^2 + C \sum_i^N {\zeta_i} \\
s.t. \; & \; y_i(wx_i + b) \geq 1 - \zeta_i
\end{aligned}$$

---
svm还可看作是损失函数带有hinge loss的最优化问题，此时svm的目的是最小化这样的损失函数：
$$
L(y_i, \hat{y}_i) = max(0, -y_i\hat{y}_i + 1) + \lambda|w|^2
$$
其中第一项是合页损失hingeloss，定义为：$hingeloss(y, \hat{y}) = max(0, -y\hat{y} + 1)$

合页损失hingeloss可以看作是松弛变量$\zeta_i$
从数学方面来讲，因为$\zeta_i \geq 0$，并且$hingeloss \geq 0$，所以可以直接看作是相同的。
从物理意义上来讲，hingeloss是最大化分离间隔的损失函数，和svm的想法相同。

==hingeloss为什么是最大化分离间隔的损失函数？==
分类问题中，最直观的损失函数是0-1损失，进一步有感知损失，即：
$$
L = \begin{cases}
y_i\hat{y}_i, \;\;&若y_i分类错误 \\
0, \;\; &其他
\end{cases}
$$
因为分类正确时，$y$和$\hat{y}$符号相同，所以上面的感知损失还可以进一步简写成：$L = max(0, -y_i\hat{y}_i)$

合页损失函数为了最大化分离间隔，将损失函数写成：$L = max(0, -y_i\hat{y}_i + 1)$，这样造成的效果是，为了让损失为0，$y$和$\hat{y}$符号相同还不够，还需要$y_i\hat{y}_i > 1$，这样就要求不仅分得对，还要分得离超平面足够远。

---

**当数据本身分布为非线性时**（非线性）
上述两种方法的前提是数据分布为线性，如果数据分布本身不是线性，则考虑通过将数据映射到其他特征空间，转换为线性分布后再用svm。将数据映射到线性分布的特征空间，就用到了核函数。
==核函数是低维向量映射到高维空间后的内积，也即映射函数的内积==，即：
记$\phi (x): X \to H$是将输入空间X映射到特征空间H的映射函数，则
$$
K(x, z) = \phi(x) \cdot \phi(z)
$$
常用的核函数包括：
1. 多项式核函数：$K(x,z) = (x \cdot z + 1) ^p$
2. 高斯核函数/RBF核函数：$K(x,z) = e^{- \frac{|x - z|^2}{2\sigma^2}}$


# 非线性模型
## knn/k-nearest neighbor/k近邻
核心思想：对于新的样本，找和它最靠近的k个样本，然后根据这靠近的k个样本来决定这个样本的分类
重点有3个：
1. 怎么判断最靠近？
2. 用几个k？
3. 分类决策规则？
### 距离度量
统称$L_p$距离或者闵可夫斯基距离，以计算向量$x, y$之间的距离为例，假设$x, y$的维度是n，则：
$$
L_p(x, y) = [\sum_{i = 1}^n |x_i - y_i| ^ p]^{\frac{1}{p}}
$$

p = 1时，$L_p(x, y) = \sum_{i = 1}^n |x_i - y_i|$，即曼哈顿距离（二维平面中的直角边之和）
p = 2时，$L_p(x, y) = \sqrt{\sum_{i = 1}^n |x_i - y_i| ^ 2}$，即欧式距离（二维平面中的直线距离）
p = $\infty$时，$L_p(x, y) = \max\limits_{i = 1, \dots, n}|x_i - y_i|$，即切比雪夫距离（所有坐标中的最大值）

==p = $\infty$时闵可夫斯基距离变成切比雪夫距离的推导==
记所有坐标中的最大值为M，则：
$$\begin{aligned}
L_\infty(x, y) &=\lim_{p \to \infty} [\sum_{i = 1}^n |x_i - y_i| ^ p]^{\frac{1}{p}} \\
&= M * \lim_{p \to \infty} [\sum_{i = 1}^n (\frac{|x_i - y_i|}{M}) ^p]^{\frac{1}{p}}
\end{aligned}$$
因为$\frac{|x_i - y_i|}{M} \leq 1$，所以：
$$\lim_{p \to \infty} [\sum_{i = 1}^n (\frac{|x_i - y_i|}{M}) ^p]^{\frac{1}{p}} = 1$$
所以可以推出：
$$\begin{aligned}
L_\infty(x, y) &= M * \lim_{p \to \infty} [\sum_{i = 1}^n (\frac{|x_i - y_i|}{M}) ^p]^{\frac{1}{p}} \\
&=M
\end{aligned}$$
即变成所有坐标中的最大值

### k值选择
* k较小
	* 如果临近点是噪音，则会出错
	* 整体模型复杂、容易过拟合
* k较大
	* 欠拟合

通常k选小一些，用交叉验证调参
### 分类决策规则
多数表决，哪个多就选哪个（01损失）

### 实现：kd树
#### 构造kd树
循环使用每个坐标的维度，每次把当前维度的坐标从小到大排序，然后拿中位数的那个点作为节点，比它小的节点放左边，比它大的节点放右边。然后用下一个坐标维度，迭代直到划分完。

例子：

	数据集为`{(2,3), (5,4), (9,6), (4,7), (8,1), (7,2)}`，构造1棵平衡kd树
解答：
先选x作为划分标准，然后再选y，再选x，再选y，……
第一次x的坐标为：`2,5,9,4,8,7`，排序后变成：`2,4,5,7,8,9`，中位数是7（6也可以）
x>7的分到右子树，x<7的分到左子树。
先看左子树，左子树的节点是：`(2,3), (5,4), (4,7)`，用y作为划分标准，y的坐标是：`3,4,7`，排序后也是`3,4,7`，用4作为划分节点，以此类推……最终产生的二叉树是这样的：

			 (7,2)
			/     \
		(5,4)	  (9,6)
		/  \	   /  
	(2,3) (4,7)  (8,1)

#### 搜索kd树
给定新点N，需要找k近邻，则：
首先先按照构造kd树时选择坐标维度的顺序，找到和N点特征分类完全一样的叶子节点，记为`leaf`，保存好节点N和`leaf`之间的距离，记为d，再从`leaf`向`root`遍历，每经过一个节点时，看一下该节点另外一边的子节点和S之间的距离，记为dx，如果`dx < d`，则在这个另外的子节点中遍历搜索。这样搜索直到`root`后，如果临近的节点还不足`k`个，则把所有的`dx`从小到大排序后再搜
## 决策树/Decision Tree/DT
### 特征选择
**信息增益**
信息增益用熵来表征。熵代表信息的混乱程度，所以信息增益就是，在没有用这个特征A划分数据D时，原先信息的混乱程度，与用这个特征A划分后信息的混乱程度的差别，也即：
$$
gain(D,A) = H(D) - H(D | A)
$$
其中$H$是熵。熵相关的定义[点这里](https://blog.csdn.net/sinat_41679123/article/details/107144355)

选择特征时，选择能使信息增益最大的那个特征。这种划分标准下，会倾向选择==取值较多的特征==，所以引入增益信息比
**信息增益比**
$$
g_R(D,A) = \frac{g(D,A)}{H_A(D)}
$$
其中$H_A(D)$是数据集D关于特征A的熵，和计算数据集D的熵一样，只不过光计算数据集D中有特征A的这部分数据，即：
$$
H_A(D) = -\sum_{i = 1}^{N_A} \frac{D_i}{D}log\frac{D_i}{D}
$$
其中D是数据集的样本总数
### 决策树生成
**ID3算法**
对每个节点，用==信息增益==选择最大的节点作为分类标准
**C4.5算法**
对每个节点，用==信息增益比==作为选择标准
### 决策树剪枝
剪枝主要是依据损失函数进行的。

损失函数为：
$$
C_\alpha(T) = C(T) + \alpha|T|
$$
其中，C(T)是模型预测和样本实际值之间的差，|T|表示模型的复杂度。损失函数同时考虑了预测差和模型复杂度

剪枝的2种方法：

* 给定$\alpha$，然后比较$C_\alpha(T_a)$与$C_\alpha(T_b)$，这两个分别是保留t节点子树的总体树损失函数、减掉t子树的总体树损失函数
* CART剪枝，不提前确定$\alpha$，而是对需要剪枝的每个节点求$g_t = \frac{C(t) - C(T_t)}{(|T| - 1) }$

### CART/classification and regression tree
树类别|特征选择|定义|决策树生成|决策树剪枝
---|---|---|---|---|
分类树|基尼值|$Gini(p) = \sum_{k = 1}^K{p_k ( 1- p_k)}$|选基尼值最小的特征生成树
回归树|平方误差|$\sum_x{(y_i - f(x_i))^2}$|平方误差最小的特征

[为什么cart的分类树不用信息增益/信息增益比作为特征选择标准？](https://blog.csdn.net/sinat_41679123/article/details/107120983)

**CART剪枝**
记保留节点$T_t$子树对应的损失函数为$C_\alpha(T_t) = C(T_t) + \alpha|T|$，删除节点$T_t$子树，仅保留节点$T_t$作为根节点的损失函数为$C_\alpha(t) = C(t) + \alpha$

若保留节点$T_t$的子树，需要损失函数满足：
$$\begin{aligned}
C_\alpha(T_t) \leq C_\alpha(t) & \\
&\Leftrightarrow C(T_t) + \alpha|T| \leq C(t) + \alpha \\
&\Leftrightarrow  \alpha (|T| - 1) \leq C(t) - C(T_t) \\
&\Leftrightarrow  \alpha \leq \frac{C(t) - C(T_t)}{(|T| - 1) } \\
\end{aligned}$$
也即当$\alpha \leq \frac{C(t) - C(T_t)}{(|T| - 1) }$时，不剪枝，实际上当取等号时，为了降低模型复杂度，还是剪枝的。
所以先对每个节点，都计算$g_t = \frac{C(t) - C(T_t)}{(|T| - 1) }$，然后找到最小的$g_t$，剪去最小的$g_t$对应的$\alpha$，以此不断迭代。

**为什么选最小的$g_t$ 剪枝？**
画一张图：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200704094606624.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3NpbmF0XzQxNjc5MTIz,size_16,color_FFFFFF,t_70)
当$\alpha \geq g_t$时，应该剪枝，如果没有选最小的$g_t$，而是选了次小的$g_t$，那么当$\alpha$大于次小的$g_t$时，次小的$g_t$对应需要剪枝，最小的$g_t$对应的节点也需要剪枝，此时就不好选择该剪那个了。


# 集成学习
## bagging
bagging的思想很简单：学习X个基分类器（各自的分类效果都比较好），利用这X个基分类器的结果来决定最终结果。为了效果比较好，这X个基分类器最好相对独立一些（通过利用不同的训练数据来保证），但是不能完全独立，即训练样本不能完全不同，所以最好使用有交叠的抽样样本进行训练。


对分类问题：基分类器结果投票，如果2个label票数相同，可以增加考虑置信度
对回归问题：求平均（简单平均、加权平均（权重是基分类器的权重））

学习基分类器时，样本的选择有区别：每次学习时，从样本中有放回地选择m个样本，即每次选择1个样本，然后样本放回，重复m次。这样采样出的训练集，约有原先样本的63.2%

公式推导：
假设某个样本在1次采样中没有被采到，则概率为$1 - \frac{1}{m}$，重复m次还没被采到，概率为$(1 - \frac{1}{m})^m$，对$(1 - \frac{1}{m})^m$求极限可得：
$$
\lim_{m \to \infty} (1 - \frac{1}{m})^m = \frac{1}{e} \approx 0.368
$$
所以样本不被采样到的概率是36.8%
### RF/随机森林
随机森林采用决策树作为基分类器，在决策树的训练过程中又增加了一重随机过程：特征选择随机。从当前待选择的m个特征中，随机选出来n个特征，在这n个特征上找最优特征生成决策树。如果m = n，则是传统的决策树生成过程，如果n = 1，则是随机选出来1个特征生成决策树。推荐的取值是$n = log_2m$

总结来看，随机森林有2重随机过程：样本选择的随机，决策树生成的随机。
**样本选择的随机**：和bagging的思路一样，随机从全部样本中有放回地取出m个样本
**决策树生成的随机**：每次不从全部的特征中找最优特征，而是从全部特征的子集特征中找最优特征。寻找最优特征的方法，见决策树部分


## boosting
boosting的思路是，学一个差的方法，然后不断提升(boosting)差分类器，直到差分类器分类效果变好
### adaboost
迭代的思想。迭代步骤：

1. 计算误差$e_m$。$e_m$是所有分错的分类器的权重和，即$e_m = \sum_{i = 1}^N{\omega_{mi} I(G_m(x_i) \neq y_i)}$
2. 计算更新率$\alpha_m$。$\alpha_m = \frac{1}{2} * ln \frac{1 - e_m}{e_m}$，$\alpha_m$和$e_m$成反比，所以误差越大，下一次迭代的权重改变就越大
3. 更新权重$\omega_{m+1}$。

$$
\omega_{m+1} = 
\begin{cases} 
\frac{w_{mi}}{Z_m} e^{-\alpha_m} & G_m(x_i) = y_i \\ 
\frac{w_{mi}}{Z_m} e^{\alpha_m} & G_m(x_i) \neq y_i \\ 
\end{cases}
$$
其中$Z_m$是归一化因子，对上面公式的所有分子求和即可

### 普通提升树
提升树的中心思想是每次用树拟合残差
每轮的新分类器实际上是：
$$
f_{m+1}(x) = f_m(x) + T(x, \Theta)
$$
当需要训练树$T(x, \Theta)$时，所采用的误差函数是：$L(y, f_{m+1}(x)) = L(y, \;f_m(x) + T(x, \Theta))$，如果用平方误差损失，则化简为：
$$
L(y, \;f_m(x) + T(x, \Theta)) = \frac{1}{2} (y - f_m(x) - T)^2
$$
其中$y - f_m(x)$是一个常数，就是样本值和上一轮预测值之间的差，起名叫残差r，所以继续化简有：
$$\begin{aligned}
L(y, \;f_m(x) + T(x, \Theta)) &= \frac{1}{2} (y - f_m(x) - T)^2 \\
&= \frac{1}{2} (r - T)^2
\end{aligned}$$
也就是说，每次训练的树，其实只需要拟合残差即可。

拟合残差成立的要求是，损失函数是MSE，对于一般的损失函数不好计算了，所以改进出了GBDT

### 梯度提升树/GBDT/gradient boosting decision tree
在普通提升树中我们说到可以每轮训练树来拟合残差，但是这个只适用于平方误差损失函数。如果是其他的损失函数，应该怎么办呢？
考虑用负梯度来近似残差，也即：
$$
r_{mi} = - \left[ \frac{\partial L(y_i, f(x_i))}{\partial f(x_i)} \right]_{f(x) = f_{m-1}(x)}
$$
这样对于平方误差来说，就是普通的残差，对于其他的损失函数来说，就是近似的残差，用梯度近似可以泛化到更多的损失函数上

**传统实现**
传统实现中，GBDT需要遍历所有待分割的特征，然后在该特征上划分数据，并遍历所有数据计算拟合残差$r_i = - \left[ \frac{\partial L(y_i, f(x_i))}{\partial f_{m - 1}(x_i)} \right]$，再根据损失函数利用残差计算损失（如：损失函数是MSE，则Loss就是所有残差的平方和），最后选择最小的损失作为本次划分的特征。

可以发现GBDT的计算复杂度和特征数量、样本数量是相关的
#### 随机梯度提升树(Stochastic Gradient Boosting Tree, SGBT)
在gbdt的基础上，增加子采样/sub_sample过程，这里的子采样和rf的样本采样相同，每次有放回地使用全部样本的一个子集来训练。这种方法可以解决过拟合的风险。
### xgboost
对于输入的数据x，输出的$\hat y$ 实际上是这颗树中，x落到的叶节点的权重，即$\hat y = T_t(x) = w_j$
xgb传承自gbdt，在损失函数上更进一步，增加了正则化、用二阶泰勒展开来近似损失
损失函数推导（设共有==n==个样本）：
$$\begin{aligned}
Loss(y, \hat y) &= \sum_{i = 1}^n {L[y_i, f_m(x_i)] + \Omega(T_m)} \\
&=  \sum_{i = 1}^n {L[y_i, f_{m-1}(x_i) + T_m(x_i)] + \Omega(T_m)} \\ 
\end{aligned}$$
注意：$T_m$是指第m次迭代时的所有树

为了方便看，仅对其中一个$x_i$推导，对$L[y_i, f_{m-1}(x_i) + T_m(x_i)]$在$f_{m-1}(x_i)$上用二阶泰勒展开，得到：
$$\begin{aligned}
L[y_i, f_{m-1}(x_i) + T_t(x_i)] &= L(y_i, f_{m-1}(x_i)) + \frac{\partial{L(y_i, f_{m-1}(x_i))}}{\partial{f_{m-1}}} * T_m(x_i) + \frac{1}{2} \frac{\partial{L^2(y_i, f_{m-1}(x_i))}}{\partial{f_{m-1}^2(x_i)}} * T_m^2(x_i) \\
&= L(y_i, f_{m-1}(x_i)) + g(f_{m-1}(x_i)) * T_m(x_i) + \frac{1}{2} h(f_{m-1}(x_i)) * T_m^2(x_i) \\
&= L(y_i, f_{m-1}(x_i)) + g_i * T_m(x_i) + \frac{1}{2} h_i * T_m^2(x_i)
\end{aligned}$$
注意：一阶导数g，二阶导数h都是和$f_{m-1}(x_i)$有关，也即和样本有关的

代入原损失函数，对所有的样本，可以推出：
$$\begin{aligned}
Loss(y, \hat y) &= \sum_{i = 1}^n {L[y_i, f_{m-1}(x_i) + T_m(x_i)] + \Omega(T_m)} \\ 
&= \sum_{i = 1}^n { \{ L[y_i, f_{m-1}(x_i)] + g_i * T_m(x_i) + \frac{1}{2} h_i * T_m^2(x_i) \}+  \Omega(T_m)} \\
&= \sum_{i = 1}^n { \{ L[y_i, f_{m-1}(x_i)] + g_i * T_m(x_i) + \frac{1}{2} h_i * T_m^2(x_i) \}+ \gamma |T| + \frac{1}{2} \lambda \sum_{j = 1}^T | w_j |^2}
\end{aligned}$$

注意：
1. $T$是第m次迭代时，这颗树的叶子节点个数
2. $w_j$是第j个叶子节点的权重

进一步化简损失函数：

1. $L(y_i, f_{m-1}(x_i))$ 和$\gamma |T|$都是常数，可以直接去掉
2. $T_m(x)$是第m次迭代时x的叶节点权重和，所以：
$$\begin{aligned}
\sum_{i = 1}^n g_i * T_m(x) &= \sum_{j = 1}^T w_j * (\sum_{i \in I_j}g_i)\\
\sum_{i = 1}^n \frac{1}{2} h_i * T_m^2(x) &= \frac{1}{2} \sum_{j = 1}^T w_j^2 * (\sum_{i \in I_j}h_i)
\end{aligned}$$
其中$I_j$表示落在第j个叶子节点上的所有样本

因此损失函数化简为：
$$\begin{aligned}
Loss(y, \hat y) &= \sum_{i = 1}^n { \{ L[y_i, f_{m-1}(x_i)] + g_i * T_m(x_i) + \frac{1}{2} h_i * T_m^2(x_i) \}+ \gamma |T| + \frac{1}{2} \lambda \sum_{j = 1}^T | w_j |^2}  \\
&= \sum_{i = 1}^n { [g_i * T_m(x_i) + \frac{1}{2} h_i * T_m^2(x_i)] + \frac{1}{2} \lambda \sum_{j = 1}^T  {|w_j|^2}} \\
&= \sum_{j = 1}^T [(\sum_{i \in I_j}g_i)w_j + \frac{1}{2}(\sum_{i \in I_j}h_i + \lambda)w_j^2]
\end{aligned}$$

损失函数对$w$求导，得到：
$$\begin{aligned}
\frac{\partial L}{\partial {w}} &= \sum{g} + (\sum{h} + \lambda) w 
\end{aligned}$$
令导数为0，求出取最优值时$w$的值（==这个就是叶子节点的权重==）：
$$\begin{aligned}
w^* = -\frac{\sum{g}}{\sum{h} + \lambda}
\end{aligned}$$
代入可以求得最小的损失函数（顺便再把去掉的常数偷偷补上）：
$$\begin{aligned}
Loss(y, \hat y) &= - \frac{1}{2} \sum_{j = 1}^T {[\frac{(\sum{g})^2}{\sum{h} + \lambda}] + \gamma |T|}
\end{aligned}$$


可以看到，损失函数和树的结构有关（公式中的T），若想找到最小损失函数，朴素的想法是遍历所有损失函数，然后找最小的，这样的话就需要遍历所有结构的树，时间复杂度太高，所以放弃。

考虑一种启发性贪心的方法。先从一个单节点开始，逐步向该节点中增加分支。下一步增加哪个分支呢？依据以该特征分叉后损失函数的减少，所以`loss_reduction`和叶子节点总数无关，此时：
$$Loss\_reduction = L_{不分} - (L_{分割后的左边节点们} + L_{分割后的右边节点们})$$
用上面的损失函数代入，就有：
$$\begin{aligned}
Loss\_reduction &= - \frac{1}{2} [\frac{(\sum {g_不})^2}{\sum {h_不} + \lambda} - \frac{(\sum {g_左})^2}{\sum {h_左} + \lambda} - \frac{(\sum {g_右})^2}{\sum {h_右} + \lambda}] + \gamma (1 - 1 - 1) \\
&= \frac{1}{2} [\frac{(\sum {g_左)^2}}{\sum {h_左} + \lambda} + \frac{(\sum {g_右})^2}{\sum {h_右} + \lambda} - \frac{(\sum {g_不})^2}{\sum {h_不} + \lambda} ] - \gamma
\end{aligned}$$

**基于预排序的生成树方法**
对所有feature的值从小到大排序，然后计算：
$$
Loss\_reduction = \frac{1}{2} [\frac{(\sum {g_左)^2}}{\sum {h_左} + \lambda} + \frac{(\sum {g_右})^2}{\sum {h_右} + \lambda} - \frac{(\sum {g_不})^2}{\sum {h_不} + \lambda} ] - \gamma
$$
如果loss_reduction>0，则说明这个特征应该分叉，反之说明应该剪枝。所以这里找==最大==的loss_reduction进行分叉

**缺点**
1. 每轮迭代时需要使用全部的训练数据，放进内存太占内存，反复读写花费时间
2. 决策树生成时，对特征预排序，需要保存排序后的特征结果，空间占用较多

**优点**
计算速度快，可并行。注意：

1. 特征级别的并行，不是数据级别的并行
2. 数据级别做了优化，可并行的近似直方图算法
	* 默认还是特征预排序的方法
	* 不同的机器先在本地构造直方图，然后进行全局的合并，最后在合并的直方图上面寻找最优分割点

### Light GBM
2个改进点，针对训练样本数量、针对样本特征数量，详见[这里](https://blog.csdn.net/sinat_41679123/article/details/107144314)

