# TenXunGuangGao
2020年腾讯广告算法比赛 **赛题链接**：[Ten Xun Guang Gao](https://algo.qq.com/user.html)

- 数据：用户在在长度为 91 天（3 个月）的时间窗口内的广告点击历史记录作为训练数据集 ,求参赛者预测测试数据集中出现的用户的年龄和性别 
  - 特征选择方法：经纬度统计特征、基于`Word2Vec`的特征编码（轨迹序列Embedding向量）、`TfidfVectorizer`和`CountVectorizer`转化后使用`TruncatedSVD`的截断特征
  - 模型：`LightGBM`/`catboost`单模+5折交叉验证

- 评估方式：准确率(Accuracy)

- 排名：1.363222止步初赛