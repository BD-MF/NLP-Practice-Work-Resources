## Tanh与Relu等激活函数的适用场景

### 1.什么是激活函数
  如下图，在神经元中，输入inputs通过加权、求和后，还被作用了一个函数——激活函数Activation Function。
   ![image](https://user-images.githubusercontent.com/59279781/120734788-0fd05600-c51c-11eb-9341-516f351e684b.png)

### 2. 为什么要用激活函数 
  * 首先，如果不用激活函数，每一层输出都是上层输入的线性函数，无论神经网路有多少层，输出都是输入的线性组合。与没有隐藏层效果相当，这种情况就是最原始的感知机了。

  * 其次，使用激活函数的话，激活函数给神经元引入了非线性因素，使得神经网络可以任意逼近任何非线性函数，这样神经网络就可以应用到众多的非线性模型中。


### 3. 都有什么激活函数
  （1）sigmoid函数 
    sigmoid函数也叫 logistic 函数，用于隐层神经元输出，取值范围为(0,1)，它可以将一个实数映射到(0,1)的区间，可以用来做二分类，在特征相差比较复杂或是相差不是特别大时效果比较好。 
      ![image](https://user-images.githubusercontent.com/59279781/120735058-7fdedc00-c51c-11eb-9f36-f1a11ce3f7aa.png)
  
    优点：

      输出为0到1之间的连续实值，此输出范围和概率范围一致，因此可以用概率的方式解释输出。-
      将线性函数转变为非线性函数
    缺点：

      容易出现gradient vanishing
      函数输出并不是zero-centered
      幂运算相对来讲比较耗时
        
  ![image](https://user-images.githubusercontent.com/59279781/120735169-b6b4f200-c51c-11eb-9f21-36a08a37ab09.png)
  
  （2）tanh函数（双曲正切）

   tanh函数也称为双切正切函数，取值范围为[-1,1]。tanh在特征相差明显时的效果会很好，在循环过程中会不断扩大特征效果。与 sigmoid 的区别是，tanh 是 0 均值的，因此实际应用中 tanh 会比 sigmoid 更好。
  ![image](https://user-images.githubusercontent.com/59279781/120735242-d64c1a80-c51c-11eb-9f7f-8114b62d246e.png)
   sigmoid函数和tanh函数导数区别：

    考虑相同的输入区间[0,1]，sigmoid函数导数输出范围为[0.20,0.25]，tanh函数导数输出范围为[0.42,1]。

    优点：

       对比sigmoid和tanh两者导数输出可知，tanh函数的导数比sigmoid函数导数值更大，即梯度变化更快，也就是在训练过程中收敛速度更快。
       输出范围为-1到1之间，这样可以使得输出均值为0，这个性质可以提高BP训练的效率。
       将线性函数转变为非线性函数。
    缺点：

       梯度消失（gradient vanishing）。
       幂运算相对来讲比较耗时。
       
（3）ReLU
   Relu函数即线性整流函数，用于隐层神经元输出，输入信号 < 0时，输出都是0；输入 > 0时，输出等于输入，公式如下：
   ![image](https://user-images.githubusercontent.com/59279781/120735347-07c4e600-c51d-11eb-9f2e-2694d0851ba6.png)
  由函数图像可知其导数为分段函数，当x <= 0时，导数为0；当x > 0时，导数为1。

    优点：

      解决了梯度消失（gradient vanishing）问题 (在正区间)。
      ReLU更容易优化，因为其分段线性性质，导致其前传、后传、求导都是分段线性的。而传统的sigmoid函数，由于两端饱和，在传播过程中容易丢失信息。
      ReLU会使一部分神经元输出为0，造成了网络的稀疏性，并且减少了参数的相互依存关系，缓解了过拟合。
      计算速度非常快，只需要判断输入是否大于0。
      收敛速度远快于sigmoid和tanh。
    缺点：

      不是zero-centered。
      某些神经元可能永远不会被激活即训练的时候很”脆弱”，神经元很容易就”die”了。例如：一个非常大的梯度流过一个 ReLU 神经元，更新过参数之后，这个神经元再也不会对任何数据有激活现象了，那么这个神经元的梯度就永远都会是 0，如果 learning rate 很大，那么很有可能网络中的 40% 的神经元都”dead”了。

    对比sigmoid类函数主要变化是： 

      单侧抑制。
      相对宽阔的兴奋边界。
      稀疏激活性。

（4）softmax函数
  Softmax函数用于多分类神经网络的输出。
   ![image](https://user-images.githubusercontent.com/59279781/120735482-4490dd00-c51d-11eb-84ca-edaf769156a4.png)
    
    举个例子来看公式的意思：就是如果某一个 zj 大过其他 z, 那这个映射的分量就逼近于 1,其他就逼近于 0，主要应用就是多分类。那为什么要取指数呢？
　　   1. 模拟 max 的行为，让大的更大。
      2. 需要一个可导函数。

    而Sigmoid 和 Softmax 区别是：

        sigmoid将一个real value映射到（0,1）的区间，用来做二分类；
        softmax 把一个 k 维的real value向量（a1,a2,a3,a4….）映射成一个（b1,b2,b3,b4….）其中 bi 是一个 0～1 的常数，输出神经元之和为 1.0，所以相当于概率值，然后可以根据 bi 的概率大小来进行多分类的任务。

### 适用场景
  在CNN等结构中将原先的sigmoid、tanh换成ReLU可以取得比较好的效果，而在RNN中，将tanh换成ReLU不能取得类似的效果
  
  ![image](https://user-images.githubusercontent.com/59279781/120735657-933e7700-c51d-11eb-922b-0292ab611a10.png)
  ![image](https://user-images.githubusercontent.com/59279781/120735724-b537f980-c51d-11eb-8349-093ce11d7629.png)

