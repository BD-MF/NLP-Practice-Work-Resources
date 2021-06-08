## HMM与CRF的区别
1.HMM是生成模型，CRF是判别模型

2.HMM是概率有向图，CRF是概率无向图

3.HMM求解过程可能是局部最优，CRF可以全局最优

4.CRF概率归一化较合理，HMM则会导致label bias 问题



参考原文：
1.https://www.zhihu.com/question/35866596/answer/236886066
2.https://www.zhihu.com/question/53458773/answer/554436625
