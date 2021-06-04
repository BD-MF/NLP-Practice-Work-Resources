# 图卷积网络GCN知识汇总

## 一、简介
   ![image](https://user-images.githubusercontent.com/59279781/120822306-dc77e080-c588-11eb-9cd4-08e809984779.png)

## 二、CNN中的卷积
   ![image](https://user-images.githubusercontent.com/59279781/120822393-f31e3780-c588-11eb-8bc2-372dc6f38846.png)

## 三、CNN与GCN的区别
   ![image](https://user-images.githubusercontent.com/59279781/120822426-fca79f80-c588-11eb-8155-9341f6a77134.png)

## 四、提取拓扑图特征的方式
   ![image](https://user-images.githubusercontent.com/59279781/120822475-07623480-c589-11eb-9060-0af7cb3bd75d.png)

## 五、图网络的基本概念（偏向基于空间的图卷积网络）
### 5.1 定义
   ![image](https://user-images.githubusercontent.com/59279781/120822544-18ab4100-c589-11eb-9f96-d15b97132ac0.png) 

### 5.2 一个简单的例子
   ![image](https://user-images.githubusercontent.com/59279781/120822666-3bd5f080-c589-11eb-8327-62b434d50aca.png)
   ![image](https://user-images.githubusercontent.com/59279781/120822685-42646800-c589-11eb-91eb-f07489e61b73.png)
   ![image](https://user-images.githubusercontent.com/59279781/120822725-4abca300-c589-11eb-9f73-cf36470aad37.png)
   ![image](https://user-images.githubusercontent.com/59279781/120822757-527c4780-c589-11eb-8747-f235ffc4f93b.png)

### 5.3 特征聚合
   ![image](https://user-images.githubusercontent.com/59279781/120822818-62942700-c589-11eb-95c9-06485aedd356.png)
   ![image](https://user-images.githubusercontent.com/59279781/120822850-67f17180-c589-11eb-8159-6ce2be574647.png)

## 六、基于谱的图卷积网络

### 6.1 拉普拉斯矩阵
   ![image](https://user-images.githubusercontent.com/59279781/120822947-7b9cd800-c589-11eb-8219-a12791803657.png)
   ![image](https://user-images.githubusercontent.com/59279781/120822997-86576d00-c589-11eb-9ba0-256d404c6729.png)

### 6.2 拉普拉斯矩阵的谱分解（特征分解）
   ![image](https://user-images.githubusercontent.com/59279781/120823038-92dbc580-c589-11eb-8de2-bfd529c4dd59.png)
   
### 6.3 GCN使用拉普拉斯矩阵的原因
   ![image](https://user-images.githubusercontent.com/59279781/120823087-a25b0e80-c589-11eb-8509-698238569cc9.png)

### 6.4 Graph上的傅里叶变换及卷积
   ![image](https://user-images.githubusercontent.com/59279781/120823130-aedf6700-c589-11eb-98ea-ba6ce85a806e.png)

### 6.5 三种经典的谱图卷积网络
   ![image](https://user-images.githubusercontent.com/59279781/120823181-c0c10a00-c589-11eb-82ae-9dd5943a9071.png)

   ![image](https://user-images.githubusercontent.com/59279781/120823203-c6b6eb00-c589-11eb-9742-0c8bf1e3bd82.png)

   




   
