@[toc]
# 熵相关的概念
参考链接：[https://www.cnblogs.com/kyrieng/p/8694705.html](https://www.cnblogs.com/kyrieng/p/8694705.html)
## 熵  
对单个随机离散变量x来说，假设x的概率分布为P，则定义熵：
$$
H(x) = -\sum_x P(x) log P(x)
$$

注意，熵只和变量的概率分布有关，和变量本身的取值大小无关
## 联合熵  
对2个随机离散变量x,y来说，记P(x,y)为x,y的联合概率分布，则定义联合熵H(x,y)为：
$$
H(X,Y) = -\sum_{x,y}P(x,y) log P(x,y)
$$

和交叉熵区别：这是1个概率分布对2个变量

## 条件熵  
定义一个随机变量在给定另一个随机变量下的条件熵（条件分布上关于起条件作用的那个随机变量取平均之后的期望值）
$$\begin{aligned}
H(Y|X) &= \sum_xP(x)H(Y|X = x) \\
&= - \sum_xP(x) \sum_yP(y|x) log P(y|x) \\
&= - \sum_{x,y}P(x,y) log P(y|x)
\end{aligned}$$

**链式法则**
$H(x,y) = H(x) + H(y|x) \;\; \Leftrightarrow \;\; H(y|x) = H(x, y) - H(x)$  
**证明**：

$$\begin{aligned}
H(x,y) &= -\sum_{x,y}P(x,y)logP(x,y) \\
&= -\sum_{x,y}P(x,y)log(P(y|x)P(x)) \\
&= -\sum_{x,y}P(x,y)logP(y|x) - \sum_{x,y}P(x,y)logP(x) \\
&= H(y|x) - \sum_{x} \sum_{y}P(x, y)logP(x) \\
&= H(y|x) - \sum_{x} logP(x)\sum_{y}P(x, y) \\
&= H(y|x) - \sum_{x} logP(x)P(x) \\
&= H(y|x) + H(x)
\end{aligned}$$

## 交叉熵
对于分布为p(x)的随机变量，熵`H(p)`表示其最优编码长度。
交叉熵$H(p,q)$的含义是：按照概率分布q的最优编码，对真实分布p的信息进行编码的长度。

$$H(p, q) = -\sum_x p(x) logq(x)$$

和联合熵区别：这是2个概率分布对1个变量

交叉熵广泛用于逻辑回归的Sigmoid和Softmax函数中，作为损失函数使用，原因就是下面的相对熵/KL散度

## 相对熵/KL散度
相对熵也叫做KL散度，用来衡量2个概率分布之间的距离。$KL(p, q)$表示用概率分布q去近似概率分布p时的信息损失

$$\begin{aligned}
KL(p, q) &= \sum_x p(x)log\frac{p(x)}{q(x)} \\
&= \sum_xp(x)logp(x) - \sum_xp(x)logq(x) \\
&=H(p,q) - H(p)
\end{aligned}$$

注意：
1. $KL(p, q) \neq KL(q, p)$
2. 散度的意义是两个概率分布的距离，所以取值范围是$[0, +\infty]$，当且仅当2个概率分布完全一样时取0

在机器学习的损失函数计算中，实际上求的是模型学习到的概率分布q，和数据的真实分布p之间的距离最小值，也就是上述的`min KL(p, q)`
因为$KL(p, q)=H(p,q) - H(p)$，并且数据的真是分布p是一个定值，也即$H(p)$是确定的，此时求KL散度的最小值，就是求 ==交叉熵$H(p, q)$== 的最小值，所以损失函数中会使用交叉熵

# 模型评估指标
内容主要参考西瓜书

## 混淆矩阵、准召、F1
**混淆矩阵**
真实\预测|||
---|---|---|
||正|负
正|TP|FN
负|FP|TN

注意这些结果都是对预测值说的，是把预测值和真实值比较后，预测值的结果

**准召F1**
$$\begin{aligned}
P &= \frac{TP}{TP + FP} \\
R &= \frac{TP}{TP + FN} \\
f1 &= \frac{2PR}{P + R}
\end{aligned}$$
准确率：想看看模型预测出来的正样本中，有多少是真正的正样本，所以分母是所有预测出的正样本。
召回率：想看看模型把多少正样本拉回来了，所以分母是所有真实值的正样本。
F1就是取他俩的平均。分子的乘法可以避免出现一大一小导致结果很大的情况。

## ROC/AUC
先介绍真假正例率。记x是假正例率，y是真正例率，则：
$$\begin{aligned}
x &= \frac{FP}{FP + TN} \\
y &= \frac{TP}{TP + FN}
\end{aligned}$$
假正例率，就是想看看预测出来的错误的正例，在真正的负例中占多少
真正例率（和recall一样），想看看预测出来的正确的正例，在真正的正例中占多少（同recall，把多少正确的正例拉回来了）

x轴代表的是高于阈值的负样本，占全部负样本的比例
y轴代表的是高于阈值的正样本，占全部正样本的比例
如果把正样本复制粘贴，重新评估模型，则AUC没有变化。==这就是为什么AUC对正负样本比例不敏感的原因==

roc曲线，就是把预测值按照prob从大到小排序，然后阈值从最高的prob开始，一直下降到最低的prob，计算真、假正例率，以x为横轴，y为纵轴画图

**对特殊点的讨论**
1. 点`(0, 0)`代表把所有样本都划分为负例，此时：
`FP = 0`，因此x轴假正例率为0
`TP = 0`，因此y轴真正例率为0
2. 点`(1, 1)`代表把所有点都划分为正例，此时：
$FP = n_N$，$n_N$表示真实值负例的个数，$TN = 0$，因此假正例率为$x = \frac{FP}{FP + TN} = \frac{n_N}{n_N + 0} = 1$
$TP = n_P$，$n_P$表示真实值中正例的个数，$FN = 0$，因此真正例率为$y = \frac{TP}{TP + FN} = \frac{n_P}{n_P + 0} = 1$

因为取的是有限个样本点，所以曲线是这样的：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200722140748748.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3NpbmF0XzQxNjc5MTIz,size_16,color_FFFFFF,t_70)
AUC就是ROC曲线中阴影部分的面积。

所以计算的时候，可以用连续好几个梯形的面积和来近似，即对上图，假设有N个样本，则：
$$
AUC = \frac{1}{2} \sum_{i = 1}^{N - 1}(y_i + y_{i + 1})(x_{i + 1} - x_i)
$$

AUC实际体现的是模型把正样本排在负样本前面的能力

### gauc/groupAUC
如果模型输入的是多个用户，评价多个用户对某个广告的评分时，应该把每个用户分开算。即考虑每个用户的排序，所以此时AUC变成groupAUC，即：
$$
GAUC = \frac{\sum_{u \in U}w_uAUC_u}{\sum_{u \in U}w_u}
$$

其中，$w_u$代表每个user的权重，一般可以设为每个用户view的次数或click的次数等
# 损失函数
## 分类问题
### 交叉熵损失
损失函数：
$$
L(y, \hat{y}) = -y^T log\hat{y}
$$

可以通过对KL散度的推导而来，详见上面KL散度
### hinge loss


## 回归问题
### 均方误差/MSE
损失函数：
$$
L(y, \hat{y}) = \frac{\sum_{i = 1}^n{\frac{1}{2} (y_i - \hat{y_i} ) ^2}}{n}
$$

### 平均绝对误差/MAE
损失函数：
$$
L(y, \hat{y}) = \frac{\sum_{i = 1}^n{|y_i - \hat{y_i}|}}{n}
$$

### huber损失
损失函数：
$$
L_\delta(y, \hat{y}) = 
\begin{cases}
\frac{1}{2} (y - \hat{y})^2, & |y - \hat{y}| \leq \delta \\
\delta|y - \hat{y}| + \frac{1}{2}\delta^2, & |y - \hat{y}| > \delta
\end{cases}
$$

优点：改进L1/L2损失的缺点，即在远离0点时采用L1损失，在靠近0点后采用L2损失
缺点：需要调整$\delta$参数

## 回归/分类都能用的
### 01损失
损失函数：
$$
    l(y, \hat{y})=
    \begin{cases}
      1, & if \; y = \hat{y} \\
      0 & if \; y \neq \hat{y} \\
    \end{cases}
$$
优点：
能直观评价模型好坏
缺点：
1. 不连续
2. 导数为0，难以优化


# 优化方法
## 梯度下降/gradient descent
步骤：
1. 确定学习率
2. 参数初始化
3. 选择样本计算损失函数梯度
5. 更新参数

注：
梯度下降能保证在凸函数上找到全局最优点，在非凸函数上找到局部最优点

凸函数：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200708164705604.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3NpbmF0XzQxNjc5MTIz,size_16,color_FFFFFF,t_70)

### 批量梯度下降/batch gradient descent
在基础版本上额外的特点：
1. 选择==全部==样本计算损失函数（损失函数是==所有==样本的预测值和真实值相差之和）
2. 能保证在凸函数上找到全局最优，在非凸函数上找到局部最优

### 随机梯度下降/sgd/Stochastic gradient descent
在基础版本上额外的特点：
1. 选择==1个==样本计算损失函数（损失函数是==1个==样本的预测值和真实值相差），只要有1个样本，就更新全部参数
2. 计算速度快，在线学习可用
3. 损失函数可能会频繁抖动
4. 不能保证在凸函数上找到全局最优（可以通过逐渐减小学习率来解决），不能很快收敛

### 小批量梯度下降/Mini-batch gradient descent
综合上面2个的优点，在基础版本上额外的特点：
1. 选择==batch个==样本计算损失函数（损失函数是==batch个==样本的预测值和真实值相差）

通常情况下`batch`在`[50, 256]`之间，因为mini-batch gd非常常用，所以其实很多算法里面的`SGD`其实就是`mini-batch gd`

sgd还可以改进的点：
1. 需要手动设置学习率随梯度下降而改变
2. 无法保证收敛到最优
3. 如果数据很稀疏，可能会希望能多更新几次频率较高的特征权重，少更新很稀疏的特征权重

sgd的调优点：

1. 对训练数据要shuffle
2. 对每个mini-batch做batch-normalization
3. 通常情况下学习率和batch大小呈线性关系，即：当批量大小增加m倍时，学习率也增加m倍（仅batch比较小的时候适用）
### 加入动量的梯度下降/Momentum
为了解决存在多个局部最优解时，梯度下降可能会在这些局部最优解附近震荡的问题。

观察发现震荡时，梯度方向一次左一次右，所以考虑保存上一次的梯度。如果2次梯度方向相同，则迈的步子大一些，如果方向相反，则迈的步子小一些。

步骤变为：
1. 设置超参数：学习率$\alpha$，动量$\gamma$（通常$\gamma = 0.9$）
2. 初始化参数：初始化模型参数$\Theta$，初始化初始速度$v_0$
3. 计算损失函数梯度：$\nabla{J} = \nabla{J(\Theta)}$
4. 更新速度：$v_t = \gamma * v_{t - 1} + \alpha * \nabla{J(\Theta)}$
5. 更新参数：$\Theta_t = \Theta_{t - 1} - v_t$

其实根据具体的公式发现，增加了动量后，如果两次梯度的方向相同，则momentum实际上是用了2次梯度的和来更新参数（方向相同时参数更新更加快速），如果两次梯度方向不同，则是向数值更大的方向走了一步，走的这一步是两次梯度之差（方向不同时参数更新更慢）

momentum更新法可以保证收敛更快，更抗震荡

### Nesterov改进的动量梯度下降
先用速度更新参数得到一个临时量，然后再用这个临时量计算损失函数梯度。（搞不明白这里有啥可改进的。。。）

步骤为：
1. 设置超参数：学习率$\alpha$，动量$\gamma$（通常$\gamma = 0.9$）
2. 初始化参数：初始化模型参数$\Theta$，初始化初始速度$v_0$
3. 计算临时参数：$\hat{\Theta} = \Theta_{t - 1} - \gamma * v_{t - 1}$
4. 计算损失函数梯度：$\nabla{J} = \nabla{J(\hat\Theta)}$
5. 更新速度：$v_t = \gamma * v_{t - 1} + \alpha * \nabla{J(\hat\Theta)}$
6. 更新参数：$\Theta_t = \Theta_{t - 1} - v_t$

---
低维空间中，优化的难点通常在于初始化参数、避免局部最优点。但高维空间中，难点在于避免鞍点


### Adagrad/自适应梯度下降
对每个参数都维护一个学习率，不频繁的特征学习率大（更新快）、频繁出现的特征学习率小（更新慢）

计算步骤：
1. 计算损失函数梯度：$g_t = \nabla_{\Theta}{J(\Theta)}$
2. 计算累计平方梯度：$\gamma_t = \gamma_{t - 1} + g_t \odot g_t$
3. 更新参数：$\Theta_t = \Theta_t - \frac{\alpha}{\epsilon + \sqrt{\gamma_t}} * g_t$

$\epsilon$是为了防止分母为0
$\gamma_{t,i}$比较大时，也就是梯度在这个参数上比较陡峭时，可能预示着发生了震荡，此时通过把$\gamma_t$放在分母上可以保证学习率变小，也即下次更新时变慢。
反之，如果梯度在某个参数上比较平缓，那么分母就小，学习率就变大，下次更新就快一些。

adagrad的优点是：不需要手动调整学习率$\alpha$随着梯度减小而减小
adagrad的缺点是：随着迭代次数增加，$\gamma_t$会变得越来越大（毕竟是累加的），这会导致参数下降变得很缓慢，并且此时很难找到最优点

### RMSprop
为了改进Adagrad中$\gamma_t$会随着时间越来越大的问题，提出RMSprop，计算步骤为：
1. 计算损失函数梯度$g_t = \nabla_{\Theta}{J(\Theta)}$
2. 计算梯度$g_t$平方的指数衰减移动平均：$G_t = \beta G_{t - 1} + (1 - \beta)g_t \odot g_t$
3. 更新参数：$\Theta_t = \Theta_{t - 1} - \frac{\alpha}{\sqrt{G_t} + \epsilon} \odot g_t$

用衰减系数$\beta$来防止$G_t$随时间累积变得过大

### Adadelta
为了改进adagrad的缺点，提出adadelta

计算步骤：
1. 初始化参数：衰减系数$\rho$
2. 选择全部样本
3. 计算损失函数梯度$g_t =  \nabla_{\Theta}{J_t(\Theta)}$
4. 计算累计平方梯度：$RMS[g]_t = \sqrt{E[g^2]_t + \epsilon} = \sqrt{\rho*E[g^2]_t + (1 - \rho )g_t^2 + \epsilon}$
5. 计算累计更新：$RMS[\Delta \Theta]_t = \sqrt{E[\Delta \Theta^2]_t + \epsilon} = \sqrt{\rho * E[\Delta \Theta^2]_{t-1} + (1 - \rho) \Delta \Theta_t^2 + \epsilon}$
6. 计算参数更新：$\Delta \Theta_t = -\frac{RMS[\Delta \Theta]_{t-1}}{RMS[g]_t} * g_t$
7. 更新参数：$\Theta_t = \Theta_{t-1} + \Delta \Theta_t$

### Adam/Adaptive Moment Estimation
计算步骤：
1. 初始化参数：衰减系数$\rho_1 = 0.9, \rho_2 = 0.999$，一阶变量$s_0 = 0$，二阶变量$\gamma_0 = 0$
2. 选择全部样本
3. 计算损失函数梯度：$g_t = \nabla_{\Theta}{J_t(\Theta)}$
4. 更新有偏一阶矩估计（一阶动量）：$s_t = \rho_1 * s_{t-1} + (1 - \rho_1) * g_t$
5. 更新有偏二阶矩估计（二阶动量）：$\gamma_t = \rho_2 * \gamma_{t - 1} + (1 - \rho_2) g_t \odot g_t$
6. 修正一阶矩估计偏差：$\hat{s_t} = \frac{s_t}{1 - \rho_1^t}$
7. 修正二阶矩估计偏差：$\hat{\gamma_t} = \frac{\gamma_t}{1 - \rho_2^t}$
8. 参数更新：$\Theta_t = \Theta_{t - 1} -  \frac{\alpha}{\sqrt{\hat{\gamma_t}} + \delta} * \hat{s_t}$


[adam优化算法为什么要做偏差修正？](https://blog.csdn.net/sinat_41679123/article/details/107120983)

### Adamax
计算步骤：
1. 初始化参数：和adam相同
2. 选择全部样本
3. 计算损失函数梯度：$g_t = \nabla_{\Theta}{J_t(\Theta)}$
4. 更新有偏一阶矩估计：$s_t = \rho_1 * s_{t-1} + (1 - \rho_1) * g_t$
5. 更新有偏二阶矩估计：$\gamma_t = max(\rho_2 * \gamma_{t - 1}, |g_t|)$
6. 修正一阶矩估计偏差：$\hat{s_t} = \frac{s_t}{1 - \rho_1^t}$
7. 参数更新：$\Theta_t = \Theta_{t - 1} -  \frac{\alpha}{\gamma_t} * \hat{s_t}$

### Nadam
1. 初始化参数：衰减系数$\rho_1 = 0.9, \rho_2 = 0.999$，一阶变量$s_0 = 0$，二阶变量$\gamma_0 = 0$
2. 选择全部样本
3. 计算损失函数梯度：$g_t = \nabla_{\Theta}{J_t(\Theta)}$
4. 梯度更新：$\hat{g_t} = \frac{g_t}{1 - \prod_{i = 1}^t{\rho_i}}$
5. 更新有偏一阶矩估计：$s_t = \rho_1 * s_{t-1} + (1 - \rho_1) * g_t$
6. 修正一阶矩估计偏差：$\hat{s_t} = \frac{s_t}{1 - \prod_{i = 1}^{t + 1}\rho_i}$
7. 更新有偏二阶矩估计：$\gamma_t = \rho_2 * \gamma_{t - 1} + (1 - \rho_2) g_t \odot g_t$
8. 修正二阶矩估计偏差：$\hat{\gamma_t} = \frac{\gamma_t}{1 - \rho_2^t}$
9. 再修正一阶矩估计：$\overline{s_t} = (1 - \rho_1^t) * \hat{g_t} + \rho_1^{t + 1} * \hat{s_t}, \rho_1^t = \rho_1 * (1 - 0.5 * 0.96^{\frac{t}{259}})$
10. 参数更新：$\Theta_t = \Theta_{t - 1} -  \frac{\alpha}{\sqrt{\hat{\gamma_t}} + \delta} * \overline{s_t}$

### ANGD
加入了费雪信息矩阵

### 总结
总的来说，对比较稀疏的数据，推荐使用后几种优化器（自适应优化器可以免去手动调整学习率的烦恼），其中：Adam, Nadam基本相同，Adam的效果更好，==通常选择Adam优化器==




## 二阶近似方法
### 牛顿法、拟牛顿法
**牛顿法**
一个不断迭代逼近的方法，作用有：有函数的解、最优化。两个用法的思想完全相同，先以求函数的解为例：

假设需要求解函数`f(x) = 0`，有时候求根公式写不出来，这时候就可以用牛顿法。
在$x_0$上对$f(x)$进行一阶泰勒展开，有：
$$\begin{aligned}
& f(x) = f(x_0) + f'(x_0)(x - x_0) \\
\Leftrightarrow & x = x_0 - \frac{f(x_0)}{f'(x_0)}
\end{aligned}$$

因为只进行了一阶泰勒展开，所以有一定的误差，也就是说上面式子里求出来的$x$不能保证$f(x) = 0$，但是新求出来的$x$一定比原先的$x$更使$f(x)$接近0，所以可以通过不断迭代，最终求出在误差允许范围内的$x$值，也即：
$$\begin{aligned}
x_1 &= x_0 - \frac{f(x_0)}{f'(x_0)} \\
x_2 &= x_1 - \frac{f(x_1)}{f'(x_1)} \\
& \dots \\
x_{n+1} &= x_n - \frac{f(x_n)}{f'(x_n)} \\
\end{aligned}$$

最终可以用$x_{n + 1}$来近似实际真值$x^*$

求最优值的想法和上面完全一样，区别在于，想求$\min\limits_{x \in R}f(x)$，即求$x^*$使得$f'(x^*) = 0$，所以其实就是求导函数的解。

类似地，对于导函数而言，近似的解就是$x_{n + 1} = x_n - \frac{f'(x_n)}{f''(x_n)}$
用$g_n, H_n$分别代表在$x_n$处的一阶导数和二阶导数，则：
$$
x_{n + 1} = x_n - \frac{f'(x_n)}{f''(x_n)} = x_n - g_nH_n^{-1}
$$

以上推导都是对x为单个变量的情况，如果x是向量，则上面的$H_n$称为黑塞矩阵/hessian matrix

对单个变量而言，函数有最小值的条件是：二阶导数为正。对于向量而言，就是hessian matrix为正定矩阵（特征值为正）

在实际应用中，不会直接求hessian matrix的逆，而是转换为求解线性方程组：$H_np_n = -g_n$，从而：$x_{n + 1} = x_n + p_n$

==牛顿法中求解$H_n^{-1}$的计算量很大，所以考虑用矩阵$G_n$来代替，这就是拟牛顿法==

**拟牛顿法**
在牛顿法中求hessian matrix的逆计算量太大了，所以用$G_n$来代替hessian matrix，对于$G_n$，有2个限制条件：
1. $G_n$必须是一个正定矩阵。$G_n$是$H_n^{-1}$，如果$H_n$是正定矩阵，那么$H_n^{-1}$也是正定矩阵，所以$G_n$是正定矩阵。
2. $G_n$必须满足：$G_n(g_{n + 1} - g_n) = x_{n + 1} - x_n$

第2个条件的推导如下：
牛顿法求解导函数的解时，假设迭代到了$x_n$，此时有：
$$
f'(x) = f'(x_n) + f''(x_n)(x - x_n)
$$
令$x = x_{n + 1}$，有：
$$\begin{aligned}
& f'(x_{n + 1}) = f'(x_n) + f''(x_n)(x_{n + 1} - x_n) \\
\Leftrightarrow \;\; &f'(x_{n + 1}) - f'(x_n) = f''(x_n)(x_{n + 1} - x_n) \\
\Leftrightarrow \;\; &g_{n + 1} - g_n = H_n(x_{n + 1} - x_n) \\
\Leftrightarrow \;\; & H_n^{-1}(g_{n + 1} - g_n) = x_{n + 1} - x_n \\
\Leftrightarrow \;\; & G_n(g_{n + 1} - g_n) = x_{n + 1} - x_n \\
\end{aligned}$$

### 共轭梯度
### BFGS
## IIS/改进的迭代尺度法
# 神经网络模型中的激活函数
激活函数需要具有的性质：

性质|原因
---|---|
连续并可导(允许少数点上不可导)的非线性函数|可以直接利用数值优化的方法来学习网络参数|
激活函数及其导函数要尽可能的简单|有利于提高网络计算效率
激活函数的导函数的值域要在一个合适的区间内，不能太大也不能太小|否则会影响训练的效率和稳定性
## sigmoid型
### Logistic 函数
函数：
$$
\sigma(x) = \frac{1}{ 1 + e^{-x}}
$$
性质：
1. $\sigma^{'}(x) = \sigma(x) * (1 - \sigma(x))$
2. $1 - \sigma(x) = \sigma(-x)$

简单画一下sigmoid及其导数的图：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200722130526801.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3NpbmF0XzQxNjc5MTIz,size_16,color_FFFFFF,t_70)
根据图像易知：
对于$\sigma(wx)$函数，当$wx \in [-3, 3]$时，函数处于近似线性域（非饱和域），一旦$wx$超过这个范围，$\sigma(wx)$就要么是0，要么是1了。

同样地，若$wx \leq -3 \; or \; wx \geq 3$，则$\sigma'(wx) \approx 0$，此时梯度几乎消失。

### tanh函数
$$
tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
$$
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200705202510734.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3NpbmF0XzQxNjc5MTIz,size_16,color_FFFFFF,t_70)
导数：
$tanh'(x) = 1 - [tanh(x)]^2$

图像：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200711210917384.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3NpbmF0XzQxNjc5MTIz,size_16,color_FFFFFF,t_70)

上面2个函数的计算开销比较大，可以用分段函数进行近似（0附近线性，-1和1之外恒定）
用<font color = 'red'>泰勒展开</font>来近似：

在0附近对Logistic函数泰勒展开有：
$$\begin{aligned}
g_l(x) &\approx f(0) + f^{'}(0)(x-0) \\
&= 0.5 + 0.25x
\end{aligned}$$
所以logistic的近似函数为：
$$
    hard-logistic(x)=
    \begin{cases}
      1, & g_l(x) >=1 \\
      g_l, & 0 < g_l(x) < 1 \\
      0, & g_l(x) <= 0
    \end{cases}
$$
缩写一下就是$hard-logistic(x) = max( min ( g_l, 1), 0) = max( min ( 0.5 + 0.25x, 1), 0)$

在0附近对tanh函数泰勒展开有：
$$\begin{aligned}
g_t(x) &\approx f(0) + f^{'}(0)(x-0) \\
&= x
\end{aligned}$$
tanh的近似函数为：
$$
    hard-logistic(x)=
    \begin{cases}
      1, & g_t(x) >=1 \\
      g_t, & 0 < g_t(x) < 1 \\
      0, & g_t(x) <= 0
    \end{cases}
$$
缩写为：$hard-tanh(x) = max( min ( x, 1), 0)$

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200705202448123.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3NpbmF0XzQxNjc5MTIz,size_16,color_FFFFFF,t_70)

## Relu类
### 普通的relu(x)
激活函数为：
$$
    relu(x)=
    \begin{cases}
      x, & x \geq 0 \\
      0 & x < 0 \\
    \end{cases}
$$

==dead ReLU problem==
当输入的x<0时，ReLU函数会使得梯度为0。当梯度为0时，反向传播不再更新参数，会导致这个神经元连的所有参数都不再更新

能够引起dead ReLU问题的原因有：
1. 参数初始化有问题
2. 学习率设置的过大，导致在梯度更新时，参数一下子从正的更新成了负的，此时会导致wx < 0

**解决方案**
1. leaky ReLU，在x<0的时候不要通通都变0
2. 采用自动调整学习率的优化算法，避免参数更新过快一下改变了符号

### Leaky ReLU
激活函数为：
$$
    Leaky \; relu(x)=
    \begin{cases}
      x, & x \geq 0 \\
      \lambda x & x < 0 \\
    \end{cases}
$$
### 带参数的relu
激活函数为：
$$
    P  relu_i(x)=
    \begin{cases}
      x, & x \geq 0 \\
      \lambda_i x & x < 0 \\
    \end{cases}
$$
其中$\lambda_i$是可以学习的，可以各个神经元不同，也可以神经元之间共享
### ELU 函数
指数线性单元，激活函数：
$$
ELU(x)=
    \begin{cases}
      x, & x \geq 0 \\
      \lambda (e^x -1) & x < 0 \\
    \end{cases}
$$
### Softplus 函数
激活函数：
$$
Softplus(x)=ln( 1 + e^x)
$$
其导数是logistic函数

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200705202339926.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3NpbmF0XzQxNjc5MTIz,size_16,color_FFFFFF,t_70)
## Swish function (2017)
$$
swish(x) = x\sigma(\beta x)
$$
自门控（Self-Gated）的激活函数，当𝜎(𝛽𝑥) 接近于1 时，门处于“开”状态，激活函数的输出近似于𝑥 本身；当𝜎(𝛽𝑥) 接近于0 时，门的状态为“关”，激活函数的输出近似0

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200705202558443.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3NpbmF0XzQxNjc5MTIz,size_16,color_FFFFFF,t_70)

## GELU 函数
## Maxout 单元

## 激活函数之间的对比
激活函数的发展，大概是`sigmoid -> tanh -> ReLU -> ReLU变种 -> maxout`
激活函数|优点|缺点
---|---|---|
sigmoid|输出在(0,1)之间，输出范围有限|饱和区域梯度几乎为0（梯度消失）<br>输出的均值不是0，导致后面神经元的输入全是正值<br>含有指数计算，速度较慢
tanh|输出在[-1, 1]之间，避免sigmoid输出均值不为0的缺点|饱和区域梯度几乎为0<br>含有指数计算，速度慢
ReLU|在x>0的区域上，解决梯度为0的问题<br>没有指数计算，速度快|dead ReLU problem
ReLU变种|leaky ReLU等，可以解决dead ReLU problem
maxout|

# 归一化/标准化
基本都是指比较常见的特征缩放/feature scaling的方式
## 方法
**最大最小归一化**
$$
x^* = \frac{x - min(x)}{max(x) - min(x)}
$$
经过这种归一化之后，x的取值范围是`[0, 1]`
**标准化**
$$
x^* = \frac{x - \mu}{\sqrt{\sigma^2}}
$$
经过标准化后，x的均值为0，方差为1
**scale to unit length**
$$
x^* = \frac{x}{|x|}
$$
**mean normalization**
$$
x^* = \frac{x - \mu}{max(x) - min(x)}
$$

## 原因
输入的数据，每经过一层hidden layer，数据的分布会有一定的改变，对于较深的网络来说，输入数据的分布会逐渐越来越漂移。

数据的分布保持一致，有2个好处：
1. 可以保证反向传播调整参数时，参数不需要根据分布而改变。网络学习独立同分布的数据时更加有效。
2. 可以调整数据的分布范围，变成1个均值为0，标准差为1的分布。这种分布在`sigmoid`中使用时能更避免数据落在饱和域导致梯度消失的问题

所以需要在训练时对数据进行归一化。

## 批归一化/batch normalization/BN
**总结**
对该层的==所有==神经元进行标准化，即
$$\begin{aligned}
\hat{Z}_l &= \frac{Z_l - \mu}{\sqrt{\sigma^2}} \odot \gamma + \beta \\
O_l &= f(\hat{Z})
\end{aligned}$$
其中，计算$\mu, \sigma^2$时，采用本层中的batch个神经元计算

通过以上方法，达到以下目标：
1. `learning_rate`可以选大一些，训练速度变快了
2. 训练时，数据的分布保持基本不变，训练更加有效（还是训练速度变快了）
3. 一定程度上起到正则化的效果（原文里加了BN后去掉了`dropout`和`L2 norm`

batch normalization是针对batch的，举例来说：
对于神经网络而言，输入的数据形状为：`batch_size x feature_dimension`，batch normalization是对==每个feature==，挑出batch个神经元，进行归一化，并且只对该==feature==进行归一化（所以可以看成是纵向的）

**背景**
图像处理领域中，有一个**输入协方差漂移**的问题，在cv里解决方案是将输入的图像“白化”，也即将输入的数据变成一个标准正态分布（均值为0，标准差为1），BN借鉴这个思路，在深层网络中，对每层的输入都进行简单版本的“白化”，以解决**内部协方差漂移**问题



**细节**
对神经网络中的任意一层的净输入数据，==随机==抽取K个（也就是1个batch）进行操作

记该层的净输入为$Z_l$，输出为$a_l = f(Z_l) = f(W * a_{l - 1} + b)$，通常对净输入$Z_l$进行归一化，方法是标准化：
$$\begin{aligned}
\hat{Z_l} &= \frac{Z_l - E(Z_l)}{\sqrt{var(Z_l) + \epsilon}} \\
&= \frac{Z_l - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}} \\
\end{aligned}$$
其中，$E(Z_l), var(Z_l)$分别代表净输入的期望和方差。

由于一般使用的是mini-batchGD，所以上面式子中的期望和方差不是对于全部样本的，而是对于当前这个mini-batch的，即==随机选取==该层神经元净输入中的==K==个来近似，其中：

$$\begin{aligned}
\mu_B &=  \frac{1}{K} \sum_{i = 1}^K {z_l^i} \\
\sigma_B^2 &=  \frac{1}{K} \sum_{i = 1}^K {[ (z_l^i - \mu_B) \odot (z_l^i - \mu_B)]}
\end{aligned}$$

标准化将净输入的分布改变到了正态分布，这样大多数取值都在0附近，如果使用sigmoid作为激活函数的话，净输入都落在了sigmoid的==线性区间==中，这样会减弱模型的==非线性==能力，所以最好再加额外的**缩放和平移**，最终BN的形式为：
$$\begin{aligned}
\hat{Z_l} &= \frac{Z_l - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}} \odot \gamma + \beta \\
&\overset{\triangle}{=} BN_{\gamma, \beta}(Z_l)
\end{aligned}$$

因为BN中本身就有偏移量，所以用了BN后，网络本身的函数偏移可以去掉了，最终神经元输出变为：
$$
a_l = f(BN_{\gamma, \beta}(Z_l)) = f(BN_{\gamma, \beta}(W * a_{l - 1}))
$$

**适用场景**
mini-batch要比较大
用之前要对训练数据shuffle
## 层归一化/Layer Normalization/LN
和批归一化原理基本相同，但是是对某一层中的==所有==神经元操作，相当于批归一化中K=layer unit number的情况

# dropout
**出现背景**
解决过拟合的一个方式，是模型融合/集成学习，但是对于深层网络而言，训练一个网络的时间就很长，所以预测1个样本时，用多个深度模型集成，时间跟不上，所以考虑在模型的隐藏层中间进行调整。用`dropout` 来模仿集成学习过程。

---
训练模型时，以`drop_prob`的概率随机丢弃一些神经元，以解决过拟合的问题。

通常情况下，对于输入层和输出层的节点，`drop_prob`比较小，介于0和0.5之间。对于中间的隐层，可以使用0.5

**在训练时开`dropout`，在预测时关掉`dropout`**
在训练阶段开启`dropout`，此时由于存在`dropout`，网络的权重会比不开`dropout`时要大。
实际上，预测时也应该开`dropout`，因此这样的网络是一样的，但因为丢弃一部分神经元，所以预测出的`y`不一定是最优解，最好是多次预测`y`，然后求均值作为最终的预测结果。
但是上述的做法太浪费时间了，所以实际使用时，对网络的权重加个权，对权重矩阵直接乘`1 - drop_prob`，然后预测即可

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200811135225107.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3NpbmF0XzQxNjc5MTIz,size_16,color_FFFFFF,t_70)
# 正则化
## L1正则化/Lasso回归
用了L1正则化后，模型的损失函数变为：
$$
Loss = L + \lambda\sum_{i = 1}^n|w_i|
$$
L1正则化除了可以解决过拟合的问题外，还能有一定特征选择的作用，因为使用了L1正则化后，特征会变得非常稀疏，有很多特征的权重会直接变成0，这是因为：

用反向传播迭代求权重的最优解时，需要对$\lambda\sum_{i = 1}^n|w_i|$求导数。画一下$\lambda\sum_{i = 1}^n|w_i|$的函数图，是一个以原点为中心的倒V字，这说明，不管w的大小如何，导数都是不变的，也即w更新的速度保持不变，所以w很容易就变成0

## L2正则化/Ridge回归/岭回归
用了L2正则化后，模型的损失函数变为：
$$
Loss = L + \lambda\sum_{i = 1}^n|w_i|^2
$$
L2正则化==没有==特征选择的作用，这是因为：

用反向传播迭代求权重的最优解时，对$\lambda\sum_{i = 1}^n|w_i|^2$求导数。画一下$\lambda\sum_{i = 1}^n|w_i|^2$的函数图，是一个以原点为顶点的一元二次函数，函数在靠近0点的部分更加平滑。这说明，随着w不断靠近0，导数不断变小，也即w更新的速度随着w靠近0而减小，所以w不容易变成0

### 在0点不可导时的如何求解
次梯度下降，近端梯度下降
# 特征工程
## 特征处理
**长尾类型的特征处理方式**
用统计特征代替，比如求log
## 特征怎么筛选
1. 计算相关性，通常会计算一些相关系数，比如皮尔逊系数和互信息系数
2. 训练能够对特征进行打分的模型，如RF、GBDT、LR等，在sklearn库中可以直接用比如：
`SelectFromModel(LogisticRegression(penalty="l1", C=0.1)).fit_transform(iris.data, iris.target)`，或者`SelectFromModel(GradientBoostingClassifier()).fit_transform(iris.data, iris.target)`

## 特征重要性评估
机器学习模型：在树里有`feature_importance`等方法，通常通过特征分裂的次数、特征分裂的平均gain、特征分裂的coverage等方式评估

深度学习模型：shapley值
## 什么类型的特征，适合什么模型
高维稀疏特征：LR，FM

最好：GBDT -> LR/FM

高维稀疏特征的时候，线性模型会比非线性模型好：带正则化的线性模型比较不容易对稀疏特征过拟合

连续/稠密特征：GBDT

线性模型可以接受高维稀疏特征和普通的稠密特征，对两者没有特殊的偏好。树类模型更适合高维稀疏特征。
