---
layout: post
title: 机器学习|决策树
categories: [Machine Learning]
description: 决策树
keywords: 决策树
mermaid: false
sequence: false
flow: false
mathjax: true
mindmap: true
mindmap2: false
---

决策树是一类常见的机器学习方法，我们通过树状的结构进行分类最终得到决策。一般的，一棵决策树包含一个根结点、若干个内部结点和若干个叶结点。

叶结点对应于决策结果，其他每个结点则对应于一个属性测试；每个结点包含的样本集合根据属性测试的结果被划分到子结点中；根结点包含样本全集.从根结点到每个叶结点的路径对应了一个判定测试序列.决策树学习的目的是为了产生一棵泛化能力强，即处理未见示例能力强的决策树，其基本流程遵循简单且直观的“分而治之"(divide-and-conquer)策略。

## 基本流程


![decision-tree](/images/ml/decisiontree/1.png)：决策树的生成流程
