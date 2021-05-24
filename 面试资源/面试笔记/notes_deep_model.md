@[toc]
# 多层感知机MLP/前馈神经网络/DNN
网络中无反馈
![在这里插入图片描述](https://img-blog.csdnimg.cn/2020070520194569.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3NpbmF0XzQxNjc5MTIz,size_16,color_FFFFFF,t_70)
传播函数：
$$\begin{aligned}
Z_l &= W^l X_{l-1} + b^l \\
X_{l-1} &= f_l(Z_{l-1})
\end{aligned}$$

公式中的$Z$表示净输入（还没有经过激活函数），$X$是神经元输出（已经经过激活函数）

其中$f_l(x)$是激活函数， **激活函数需要满足的性质** ：

性质 | 原因
---| ---|
连续、可导(允许少数点上不可导)、非线性|可导保证可以直接利用数值优化的方法来学习网络参数
激活函数及其导函数要尽可能的简单|有利于提高网络计算效率
激活函数的导函数的值域要在一个合适的区间内，不能太大也不能太小|否则会影响训练的效率和稳定性

神经网络的训练过程：
1. 初始化权重向量（图中的w）
2. 前向传播：通过w和b计算出o的大小
3. 确定损失函数
4. 反向传播：计算损失函数对每个参数的导数（见下面的具体例子）
5. 不断迭代，更新参数，直到损失函数的数值小于一个定值：更新即$w_{new} = w_{old} - \frac{\partial{L}}{\partial{w_{old}}}$

假设$x_1 = 1, x_2 = 4, x_3 = 5, y_1 = 0.1, y_2 = 0.05$
$激活函数f_l(x) = \sigma(x) = \frac{1}{1 + e^{-x}}$
$损失函数L(\hat{y}, y) = \frac{1}{2} \sum_i^m{(\hat{y_i} - y_i)^2}$

前向传播：
$$\begin{aligned}
Z_{h1} &= w_1 * x_1 + w_3 * x_2 + w_5 * x_3 + b_1 = 4.3 \\
h_1 &= f_l(Z_{h1}) = 0.9866 \\
Z_{h2} &= w_2 * x_1 + w_4 * x_2 + w_6 * x_3 + b_2 = 5.3 \\
h_2 &= f_l(Z_{h2}) = 0.9950 \\
Z_{o1} &= w_7 * h_1 + w_9 * h_2 + b_2 = 5.3 \\
o_1 &= f_l(Z_{o1}) = 0.8896 \\
Z_{o2} &= w_8 * h_1 + w_10 * h_2 + b_2 = 1.3888 \\
o_2 &= f_l(Z_{o2}) = 0.8004 \\
\end{aligned}$$

$o_1, o_2$就是我们第一次预测出来的$\hat{y_1}, \hat{y_2}$

反向传播过程：
以更新$w_1$为例，计算$\frac{\partial{L}}{\partial{w_1}}$
$$\begin{aligned}
\frac{\partial{L}}{\partial{w_1}} &= \frac{\partial{L}}{\partial{o_1}} * \frac{\partial{o_1}}{\partial{w_1}}+  \frac{\partial{L}}{\partial{o_2}} * \frac{\partial{o_2}}{\partial{w_1}} \\
&= \frac{\partial{L}}{\partial{o_1}} * \frac{\partial{o_1}}{\partial{z_{o_1}}} * \frac{\partial{z_{o_1}}}{\partial{w_1}} + \frac{\partial{L}}{\partial{o_2}} * \frac{\partial{o_2}}{\partial{z_{o_2}}} * \frac{\partial{z_{o_2}}}{\partial{w_1}} \\
&= \frac{\partial{L}}{\partial{o_1}} * \frac{\partial{o_1}}{\partial{z_{o_1}}} * \frac{\partial{z_{o_1}}}{\partial{h_1}} * \frac{\partial{h_1}}{\partial{z_{h_1}}} * \frac{\partial{z_{h_1}}}{\partial{w_1}} + \frac{\partial{L}}{\partial{o_2}} * \frac{\partial{o_2}}{\partial{z_{o_2}}} * \frac{\partial{z_{o_2}}}{\partial{h_2}} * \frac{\partial{h_2}}{\partial{z_{h_2}}} * \frac{\partial{z_{h_2}}}{\partial{w_2}}
\end{aligned}$$
==注意==：
形如$\frac{\partial{o_2}}{\partial{z_{o_2}}}, \frac{\partial{h_1}}{\partial{z_{h_1}}}$的结果都是激活函数的导数，并且：
1. 在计算的时候，如果激活函数是sigmoid函数，则可以：$\sigma'(x) = \sigma(x) * (1 - \sigma(x))$，在上面的式子中，例如：$\frac{\partial{o_1}}{\partial{z_{o_1}}} = o_1 * (1 - o_1)$
2. 所以如果激活函数的导数<1，则会通过连乘导致梯度非常小，逐渐消失，从而导致无法优化

反向传播及图片参考链接：[https://www.anotsorandomwalk.com/backpropagation-example-with-numbers-step-by-step/](https://www.anotsorandomwalk.com/backpropagation-example-with-numbers-step-by-step/)

# CNN
参考链接：[https://nndl.github.io/](https://nndl.github.io/)
## 原理

全连接前馈神经网络缺点：
1. 权重矩阵参数非常多（仅上面简单的示例就有10个）
2. 体现不了一些图像的局部不变性特征（尺度缩放、平移、旋转等操作不影响语义信息）

CNN结构上的特点：
1. 局部连接
2. 权重共享（如何体现见下面）
3. 空间或者时间上的次采样


二维卷积示例：
![在这里插入图片描述](https://img-blog.csdnimg.cn/2020071021284023.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3NpbmF0XzQxNjc5MTIz,size_16,color_FFFFFF,t_70)
卷积和是上图中的中间的红色3*3方格，卷积结果应该是卷积和在原始数据上不断滑动

卷积扩展：增加零填充/padding和滑动步长
若输入了M个神经元，卷积大小为K，步长为S，输入两端各填充P个0，那么输出的神经元个数应该是：
$$
N = \frac{M + 2 * P - K}{S} + 1
$$
两边各增加P个0后，总的神经元个数变为$M + 2 * P$，去掉一个卷积核的宽度K后，剩下$M + 2 * P - K$可供滑动，每次滑动的步长是$S$，除以S就能得到能滑动几次，也就是需要几个神经元，也就是输出几个神经元，再+1保证N非0即可。

有了这些之后，就可以用卷积层代替全连接层：
![在这里插入图片描述](https://img-blog.csdnimg.cn/2020071021430983.png)
上图的卷积层中，所有同颜色的连接上的权重是相同的，这就是==权重共享==（卷积核只捕捉输入数据中的一种特定的局部特征），如果需要多个特征，就用多个卷积核

一个例子如下（步长为2）：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200710215245238.gif)
上面展示的是对于输入的3个特征图，需要输出2个卷积特征图的情况。
因为输入了3个特征图，所以1组filter中就有3个filter，又因为需要输出2个特征图，所以共有2组filter（1组filter = 1个卷积核）

经过卷积后，特征图的大小改变并不算大，还是容易过拟合，所以会加一些pooling。除了用pooling外，增加步长也可以防止过拟合

## Pooling
**常用的pooling方式**

Max Pooling
输出该区域内的最大值

Mean Pooling
输出该区域的平均值

## 网络结构
常用的卷积网络结构如下：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200710221814953.png)
也就是 卷积+ReLU 很多个，然后加个pooling，这样的结构很多个，最后再用FC把最后输出的所有神经元连起来

现在的流行是：小卷积，大深度，少用pooling（用步长来代替pooling的效果）

还可以在网络结构中：
1. 加1x1的卷积核（升维/降维/depth之间通信）
2. 增加残差（x不经过中间这么多胡卷了直接连到最后的FC）
## textCNN
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200710224845934.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3NpbmF0XzQxNjc5MTIz,size_16,color_FFFFFF,t_70)
用词的词向量作为横向的“像素”

# RNN
## Vallina RNN
考虑时间维度（由于考虑了时间维度，所以正则化用dropout时不能丢弃$h_t$，只能丢弃$x_t$的部分单元）

ref:
https://www.cs.cmu.edu/~bhiksha/courses/deeplearning/Fall.2019/archive/www-bak12-9-2019/
https://www.youtube.com/watch?v=YYNNTrSROa4

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200705202015262.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3NpbmF0XzQxNjc5MTIz,size_16,color_FFFFFF,t_70)

传递的公式为：
$$
    \begin{cases}
      Z_t &= W_x X + W_h h_{t-1} + b_x \\
      h_t &= f_h(Z_t) \\
      y_t &= f_y(W_yh_t + b_y)
    \end{cases} 
$$

前向传播和mlp相同，反向传播略有区别。

如果把初始化的$h_0$当作参数学习时，需要计算Loss function对h的导数，此时需要注意h的去向不止一处，例如下图绿色箭头指着的位置：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200705202044981.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3NpbmF0XzQxNjc5MTIz,size_16,color_FFFFFF,t_70)

注意:
1. ==权重共享==是指：$W_x, W_h, W_y$ 3个权重不随时间变化，各层之间的权重都是相同的。对权重的梯度计算，应该叠加每层的梯度，一直加到t=0
2. 理论上，反向传播要一直算到`t = 0`，但是有时为了提高效率，当输入的序列长度比较大的时候，会使用截断，只计算固定时间间隔内的梯度回传

产生梯度消失的原因：在反向传播过程中，会计算激活函数导数的连乘，所以如果激活函数没选好，则会产生梯度消失问题（详细计算可见mlp中的反向传播计算）


**普通rnn的应用**
ref: [https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-recurrent-neural-networks](https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-recurrent-neural-networks)

Type of RNN | Illustration | Example
--- | --- | ---
one-to-one, $N_x = N_y = 1$ | ![在这里插入图片描述](https://img-blog.csdnimg.cn/20200705202104894.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3NpbmF0XzQxNjc5MTIz,size_16,color_FFFFFF,t_70)| Traditional neural network
1-to-many, $N_x = 1,  N_y > 1$ |![在这里插入图片描述](https://img-blog.csdnimg.cn/20200705202127551.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3NpbmF0XzQxNjc5MTIz,size_16,color_FFFFFF,t_70)|Music generation
many-to-1,  $N_x > 1,  N_y = 1$|![在这里插入图片描述](https://img-blog.csdnimg.cn/20200705202144431.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3NpbmF0XzQxNjc5MTIz,size_16,color_FFFFFF,t_70)|Sentiment classification
many-to-many, $N_x = N_y$|![在这里插入图片描述](https://img-blog.csdnimg.cn/20200705202159930.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3NpbmF0XzQxNjc5MTIz,size_16,color_FFFFFF,t_70)|Name entity recognition
many-to-many, $N_x \neq N_y$ <br>==encoer-decoder==<br>==seq2seq模型==|![在这里插入图片描述](https://img-blog.csdnimg.cn/20200705202206434.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3NpbmF0XzQxNjc5MTIz,size_16,color_FFFFFF,t_70)|Machine translation

## LSTM/Long Short-Term Memory
参考链接：[https://nndl.github.io/](https://nndl.github.io/)


LSTM为了解决梯度爆炸/梯度消失的问题，额外引入了一个序列状态记为$\boldsymbol{c_t}$（序列状态专门用来负责==线性的==循环信息传递，同时==非线性==地输出信息给$h_t$）然后引入了3个门：遗忘门$\boldsymbol{f_t}$、输入门$\boldsymbol{i_t}$、输出门$\boldsymbol{o_t}$

整体的逻辑是：
1. 计算细胞状态：$\boldsymbol{c_t}$：$\boldsymbol{c_t} = \boldsymbol{i_t} \odot tanh(W_c[x_t , h_{t - 1}] + b_c) + \boldsymbol{f_t} \odot \boldsymbol{c_{t-1}}$
2. 计算本轮的输出$\boldsymbol{h_t}$，本轮的输出只和本轮的状态$\boldsymbol{c_t}$有关，用输出门决定本轮状态多有用：$\boldsymbol{h_t} = \boldsymbol{o_t} \odot tanh(\boldsymbol{c_t})$

说明：

1. 上面的式子中，$\odot$是外积，也即叉乘，也即元素相乘
2. 计算细胞状态$\boldsymbol{c_t}$中`[]`，表示向量的拼接，将向量拼接起来计算可以省掉一些参数量
3. 用$tanh(x)$作为细胞状态一部分的激活函数，因为该函数的导数值域较大（当x=0时导数为1），不容易发生梯度消失的问题。当然这里换成relu也可以。
4. 3个门不是固定值，3个门也是计算出来的，计算方式和vallina rnn计算$h_t$完全相同，不过权重矩阵不同，此处用sigmoid，是贪sigmoid输出值域为`(0, 1)`，刚好能起到“门”的作用
$$\begin{aligned}
\boldsymbol{f_t} &= \sigma(W_f [x_t, h_{t - 1}] + b_f) \\
\boldsymbol{i_t} &= \sigma(W_i [x_t, h_{t - 1}] + b_i) \\
\boldsymbol{o_t} &= \sigma(W_o [x_t, h_{t - 1}] + b_o) \\
\end{aligned}$$
5. 总的参数量，即可训练的参数量，和输入的dimension、隐藏层的dimension有关，设`input_dim = n, hidden_unit = m`，则总参数量为：`(input_dim + hidden_unit + 1) * hidden_unit * 4`
原因：可训练的参数量，一共有4组，分别来自3个门以及细胞状态中的`tanh`部分，每组的参数中，首先将$x_t$和$h_{t - 1}$拼接到一起，得到一个`1 x (input_dim + hidden_unit)`的向量，隐藏层的输出是`1 x hidden_unit`，所以w矩阵的大小为`(input_dim + hidden_unit) x hidden_unit`，除此外还有个偏置`b`，`b`的大小是`1 x hidden_uint`，所以一组的参数是`(input_dim + hidden_unit + 1) x hidden_unit`，共有4组，也就是上面的答案了

感性理解：
在vallina RNN中，$h_t$由于每个时刻都被更新，所以可以看作是短期记忆（只和上1个状态有关）。
在LSTM中，额外增加的状态$\boldsymbol{c_t}$其实是个记忆单元，能够捕捉某个时刻某个关键信息，并将关键信息保存一定的时间间隔，所以是一个时间较长的短期记忆，这也就是Long Short-Term Memory名字的由来。
## GRU

LSTM中输入门和遗忘门是冗余的，GRU改进这一点，引入2个门：重置门$\boldsymbol{r_t}$、更新门$\boldsymbol{u_t}$

整体逻辑是：
计算本轮输出$\boldsymbol{h_t}$：$\boldsymbol{h_t} = \boldsymbol{u_t} \odot h_{t - 1} + (1 - \boldsymbol{u_t}) \odot tanh(W_c[x_t, \boldsymbol{r_t} \odot h_{t - 1}] + b_c)$

说明：
1. 和LSTM类似的，重置门$\boldsymbol{r_t}$、更新门$\boldsymbol{u_t}$也不是一直不变的，而是用仿照vallina RNN计算$h_t$的方式计算出来的：
$$\begin{aligned}
\boldsymbol{u_t} &= \sigma(W_u[x_t, h_{t - 1}] + b_u) \\
\boldsymbol{r_t} &= \sigma(W_r[x_t, h_{t - 1}] + b_r) \\
\end{aligned}$$
2. $\boldsymbol{u_t} = 0, \boldsymbol{r_t} = 1$ 时，GRU退化为vallina RNN
3. $\boldsymbol{u_t} = 0, \boldsymbol{r_t} = 0$ 时，本轮$\boldsymbol{h_t}$只和$x_t$有关，不考虑$\boldsymbol{h_{t - 1}}$
4. $\boldsymbol{u_t} = 1$时，$\boldsymbol{h_t}$ 只和 $\boldsymbol{h_{t - 1}}$有关，不考虑$x_t$


---

RNN主要是在时间层面加深了网络深度，在$x_t \rightarrow y_t$层面还是很浅的，只有一层：$x_t \rightarrow h_t \rightarrow y_t$，所以考虑在这方面再搞复杂一些。

相应的复杂一些的网络有：双向RNN、多层RNN

## 双向rnn
![在这里插入图片描述](https://img-blog.csdnimg.cn/2020070520221664.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3NpbmF0XzQxNjc5MTIz,size_16,color_FFFFFF,t_70)
最终输出的$y_t$，来自正向和反向结果的拼接

## 多层RNN
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200711235136327.png)

# Attention/注意力机制
**背景**：NLP做阅读理解（给一个文章作为context，回答和文章相关的问题）时，如果能突出和问题query有关的context，对这部分context用RNN做建模，效果肯定更好
**核心思想**：给重要的信息划重点（ ==实际上就是对普通的数据数据，计算加权和，作为新的输入数据== ）

设有N个输入向量，每个输入向量的维度是D，记为$[x_1, \dots, x_N]$
给定一个和任务相关的查询向量q（查询向量q可以是动态生成的，也可以是可学习的参数）
用注意力变量$z = n$表示选择了第n个向量，其中$z \in [1, N]$

在给定了q和X的情况下，选择第n个向量的概率记为$\alpha_n$，则：
$$\begin{aligned}
\alpha_n &= p(z = n | X, q) \\
&= softmax[s(x_n, q)] \\
&= \frac{e^{s(x_n, q)}}{\sum_{i = 1}^Ne^{s(x_i, q)}}
\end{aligned}$$

注意，由于计算attention score $\alpha_n$的方法是计算softmax，所以所有$\alpha_n$相加的和为1，即$\sum_i^n \alpha_i = 1$

其中$s(x, q)$是注意力打分函数，计算方式包括：
1. 加法模型：$s(x,q) = v^T tanh(Wx + Uq)$
2. 点积模型：$s(x,q) = x^T q$
3. 缩放点积模型：$s(x, q) = \frac{x^T q}{\sqrt{D}}$
4. 双线性模型：$s(x,q) = x^T W q$

其中$v, W, U$均为可学习的参数，较常用的是缩放点积模型（相对于点积模型，改进了softmax函数导数比较小的缺点）

加了注意力后，最终的输入数据X为：
$$
att(X, q) = \sum_{i = 1}^N \alpha_i x_i
$$

画成图即为：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200712201732526.png)

**attention的典型应用**
参考链接：[https://towardsdatascience.com/intuitive-understanding-of-attention-mechanism-in-deep-learning-6c9482aecf4f](https://towardsdatascience.com/intuitive-understanding-of-attention-mechanism-in-deep-learning-6c9482aecf4f)

attention是为了改进机器翻译中seq2seq问题产生的。seq2seq的思想是，用encoder将源语言变为embedding，然后用decoder将embedding变为目标语言，通常用RNN负责做embedding。

其缺点在于，如果sequence长度过长，RNN会丢掉前期的信息，而在翻译中，目标语言最开头的词可能和源语言的开头更相关一些，所以考虑用attention把这块补上。

假设源语言在embedding过程中，产生$h_1, h_2, h_3, h_4$ 这4个hidden layer output，其中$h_4$作为encoder最后一个输出，会作为decoder的第一个输入，以decoder的每个time slot输出作为query， 以$h_1, h_2, h_3, h_4$ 作为key，来计算attention

计算第一个attention时，步骤如下：
1. 计算第一个attention score：（应该使用decoder的第一个输出，但是此时decoder还没有输出，所以用decoder的输入$h_4$代替）用$h_1, h_2, h_3, h_4$ 和 $h_4$ 内积获得attention score，$a_1, a_2, a_3, a_4$ 
2. 对attention score用softmax：从$a_1, a_2, a_3, a_4$ 得到$s_1, s_2, s_3, s_4$ 
3. 用attention score和value加权求和：得到context vector 为$\sum_{i = 1}^4 s_i h_i$
4. decoder的首次输入为context vector和\<START\>拼接成的向量

计算下一步的attention和上面类似，就是把第一步里面计算attention score用到的$h_4$换成上一次decoder的输出$d_1$，把第四步里面的\<START\>同样换成上一次decoder的输出$d_1$

## attention变种
**硬性attention**
直接取让$\alpha_n$最大的x，即：
$$
att(X,q) = \hat{x}_n
$$
其中$\hat{x}_n$是让让$\alpha_n$最大的x

**键值对attention**
key用来计算$\alpha_n$，value基本起到x的作用
$$\begin{aligned}
att(X, q) &= \sum_{n = 1}^N \alpha_n v_n \\
&= \sum_{n = 1}^N \frac{e^{s(x_i, q)}}{\sum_{i = 1}^N{e^{s(x_i, q)}}} v_n
\end{aligned}$$

画图就是：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200712210755952.png)
当K = V = X时，键值对attention就是普通attention了

**multi-head attention/多头注意力**
多个query，并行，最后把得到的结果拼接起来
假设有M个query，即$Q = [q_1, q_2, \dots, q_M]$，则：
$$
att((K, V), Q) = att((K, V), q_1) \oplus att((K, V), q_2) \oplus \dots \oplus att((K, V), q_M)
$$
其中$\oplus$表示向量拼接

**self-attention**
一般使用key-value attention，主要为了评估sequence不同part之间的依赖性。和普通attention的区别在于，query不是外部的sequence，而是自己产生的。

通常，使用的时候，用$X$通过线性变换投影得到query, key, value，即：
$$\begin{aligned}
Q &= W_q X \\
K &= W_k X \\
V &= W_v X
\end{aligned}$$

如果选择缩放点积模型作为注意力打分函数的话，则self-attention输出可以直接简写为：
$$
Attr(X) = \boldsymbol{V} softmax(\frac{\boldsymbol{K^T}\boldsymbol{Q}}{\sqrt{D}})
$$

因为self-attention只需要1个输入，所以经常可以用作神经网络中的1个单独的层。

所以，self-attention和普通attention不同的地方主要在于：
1. query是自己产生的（通过线性变换投影）
2. 可以用作神经网络中的单独层（BERT BASE中用了12个）

更详细的attention和self-attention对比可见：[https://datascience.stackexchange.com/questions/49468/whats-the-difference-between-attention-vs-self-attention-what-problems-does-ea](https://datascience.stackexchange.com/questions/49468/whats-the-difference-between-attention-vs-self-attention-what-problems-does-ea)（基本都总结在上面了）
