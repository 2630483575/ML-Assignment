# Dimensionality Reduction and Clustering Analysis

## 概述

本模块为NER项目添加了降维和聚类分析功能,用于可视化和分析词嵌入(Word Embeddings)以及BERT表示。这是课程评分标准中**DimRed/Clustering**部分的实现,可获得5分。

## 功能特性

### 降维方法
- **PCA (Principal Component Analysis)**: 线性降维,保留最大方差
- **t-SNE (t-Distributed Stochastic Neighbor Embedding)**: 非线性降维,擅长保留局部结构
- **UMAP (Uniform Manifold Approximation and Projection)**: 现代降维方法,比t-SNE更快且保留全局结构

### 聚类方法
- **K-means聚类**: 将嵌入向量聚类,分析不同实体类型的分布模式

### 可视化功能
- 2D嵌入可视化(带标签和颜色编码)
- 聚类结果对比(K-means vs 真实标签)
- PCA方差解释图
- 多种降维方法对比图

## 使用方法

### 前提条件

确保已安装所有依赖:
```bash
pip install -r requirements.txt
```

### 快速开始

#### 1. 分析BiLSTM嵌入

```bash
python main.py --mode analyze --model bilstm
```

#### 2. 分析BERT嵌入

```bash
python main.py --mode analyze --model bert
```

#### 3. 同时分析两种模型

```bash
python main.py --mode analyze --model both
```

### 高级用法

直接运行分析脚本以获得更多控制:

```bash
python scripts/analyze_embeddings.py --model both --max_samples 100 --output_dir outputs --analysis_dir outputs/analysis
```

参数说明:
- `--model`: 分析哪个模型 (`bilstm`, `bert`, `both`)
- `--max_samples`: 每种实体类型的最大样本数 (默认: 100)
- `--output_dir`: 训练模型的输出目录 (默认: `outputs`)
- `--analysis_dir`: 分析结果保存目录 (默认: `outputs/analysis`)

## 输出文件

分析完成后,会在 `outputs/analysis/` 目录下生成以下可视化图表:

### BiLSTM分析
- `bilstm_pca.png` - PCA降维结果
- `bilstm_tsne.png` - t-SNE降维结果
- `bilstm_pca_variance.png` - PCA方差解释图
- `bilstm_clustering.png` - K-means聚类结果对比
- `bilstm_comparison.png` - 多种降维方法对比

### BERT分析
- `bert_pca.png` - PCA降维结果
- `bert_tsne.png` - t-SNE降维结果
- `bert_pca_variance.png` - PCA方差解释图
- `bert_clustering.png` - K-means聚类结果对比
- `bert_comparison.png` - 多种降维方法对比

## 技术实现细节

### 嵌入提取

**BiLSTM嵌入**:
- 从训练好的BiLSTM模型的embedding层提取词嵌入
- 维度: 通常为100 (由`config.bilstm.embedding_dim`控制)

**BERT嵌入**:
- 从BERT的最后一层隐藏状态提取上下文化词嵌入
- 维度: 768 (bert-base-cased)
- 对于子词标记,取平均值得到完整词的嵌入

### 降维流程

1. **标准化**: 使用StandardScaler对嵌入进行标准化
2. **降维**: 应用PCA/t-SNE/UMAP将高维嵌入降至2D
3. **可视化**: 绘制2D散点图,用颜色区分实体类型

### 聚类分析

1. **K-means聚类**: 聚类数量 = 实体类型数量 (PER, ORG, LOC, MISC, O)
2. **对比可视化**: 
   - 左图: K-means聚类结果
   - 右图: 真实实体类型分布
3. **分析**: 观察无监督聚类是否能发现实体类型的自然分组

## 代码模块

### `utils/dimensionality_reduction.py`

核心工具函数模块:

- `extract_embeddings_from_model()` - 从BiLSTM提取嵌入
- `extract_bert_embeddings()` - 从BERT提取嵌入
- `perform_pca()` - PCA降维
- `perform_tsne()` - t-SNE降维
- `perform_umap()` - UMAP降维
- `perform_kmeans()` - K-means聚类
- `plot_2d_embeddings()` - 绘制2D嵌入
- `plot_clustering_results()` - 绘制聚类结果
- `plot_pca_variance()` - 绘制PCA方差图
- `compare_reduction_methods()` - 对比多种降维方法

### `scripts/analyze_embeddings.py`

主分析脚本:

1. 加载训练好的模型
2. 从数据集中采样实体
3. 提取嵌入向量
4. 执行降维和聚类
5. 生成所有可视化图表

## 示例输出解读

### PCA方差解释图
- **左图**: 每个主成分解释的方差比例
- **右图**: 累积方差曲线
- **解读**: 前2个主成分通常解释10-30%的总方差

### t-SNE可视化
- **散点图**: 每个点代表一个词
- **颜色**: 不同颜色代表不同实体类型
- **解读**: 相同类型的实体应该聚集在一起

### 聚类对比图
- **左侧**: K-means发现的聚类
- **右侧**: 真实的实体类型标签
- **解读**: 如果聚类与真实标签一致,说明嵌入质量较高

## 注意事项

1. **计算时间**: t-SNE和UMAP可能需要几分钟,特别是样本量大时
2. **UMAP依赖**: 如果安装UMAP失败,程序会跳过UMAP分析但继续其他分析
3. **样本数量**: 默认每种实体类型采样100个,可通过`--max_samples`调整
4. **模型要求**: 必须先训练相应的模型(BiLSTM或BERT),否则会跳过该模型的分析

## 常见问题

**Q: 提示找不到模型文件?**
A: 请先训练模型:
```bash
python main.py --model bilstm --mode train
python main.py --model bert --mode train
```

**Q: UMAP安装失败?**
A: UMAP是可选的,如果安装失败,分析会自动跳过UMAP但继续PCA和t-SNE分析。

**Q: 内存不足?**
A: 减少样本数量: `--max_samples 50`

**Q: 如何提高可视化质量?**
A: 
- 调整t-SNE的perplexity参数(在代码中修改)
- 增加训练迭代次数
- 尝试不同的随机种子

## 在报告中使用

这些可视化图表可以直接用于项目报告的以下部分:

1. **数据探索**: 展示实体类型的嵌入空间分布
2. **模型分析**: 对比BiLSTM和BERT的嵌入质量
3. **降维方法对比**: 展示不同降维方法的效果差异
4. **聚类分析**: 证明模型学到了有意义的实体表示

建议在报告中加入:
- 可视化图表
- 对聚类效果的定性分析
- PCA方差解释的定量分析
- 不同降维方法的优缺点讨论
