# 使用特征提取实现文本分类的示例

本目录包含了如何使用 Hugot 框架进行文本特征提取并实现文本分类的完整示例。

## 三种实现方式

### 方式1: 特征提取 + 余弦相似度分类（Few-shot Learning）

适用于**少样本学习**场景，只需要为每个类别提供几个示例文本。

**工作原理:**
1. 使用特征提取模型（如 `all-MiniLM-L6-v2`）提取文本的嵌入向量
2. 为每个类别计算示例文本的平均向量
3. 对于新文本，计算其向量与各类别向量的余弦相似度
4. 选择相似度最高的类别作为分类结果

**优点:**
- 不需要训练分类模型
- 可以快速适应新的分类任务
- 适合类别较少的场景

**适用场景:**
- 情感分析（正面/负面/中性）
- 主题分类（少量类别）
- 意图识别（少量意图）

### 方式2: 特征提取 + 传统机器学习分类器

适用于**有足够训练数据**的场景，需要训练一个专门的分类器。

**工作流程:**
1. 使用特征提取模型提取所有训练文本的嵌入向量
2. 将特征向量和标签保存为训练数据
3. 使用传统机器学习算法（SVM、逻辑回归等）训练分类器
4. 对新文本提取特征后，使用训练好的分类器进行分类

**优点:**
- 可以处理多类别、多标签分类
- 分类准确率通常更高
- 适合复杂的分类任务

**适用场景:**
- 多类别文本分类
- 多标签分类
- 大规模文本分类任务

**推荐的 Go 机器学习库:**
- `gonum.org/v1/gonum`: 数值计算和优化
- `github.com/sjwhitworth/golearn`: 完整的机器学习库
- 或者将特征导出到 Python，使用 scikit-learn

### 方式3: 直接使用 TextClassificationPipeline（推荐）

适用于**有预训练分类模型**的场景，这是最简单高效的方式。

**优点:**
- 最简单易用
- 性能最佳
- 支持单标签和多标签分类
- 开箱即用

**适用场景:**
- 情感分析
- 文本分类（有现成模型）

## 运行示例

```bash
cd examples
go run feature_extraction_text_classification.go
```

**注意:** 首次运行会自动下载模型，需要网络连接。

## 模型选择

### 特征提取模型推荐

- **轻量级**: `sentence-transformers/all-MiniLM-L6-v2` (384维)
- **平衡**: `sentence-transformers/all-mpnet-base-v2` (768维，更准确)
- **高性能**: `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` (多语言支持)

### 文本分类模型推荐

- **情感分析**: `distilbert-base-uncased-finetuned-sst-2-english`
- **多标签分类**: `roberta-base-go_emotions`
- **自定义模型**: 使用 Hugging Face 训练并导出为 ONNX 格式

## 性能建议

1. **批量处理**: 尽量批量处理文本（推荐批次大小 16-32）
2. **后端选择**: 
   - CPU: 使用 `NewGoSession()` (纯 Go，简单)
   - 性能: 使用 `NewORTSession()` (需要编译标签 `-tags ORT`)
   - 训练: 使用 `NewXLASession()` (需要编译标签 `-tags XLA`)

## 进阶使用

### 特征归一化

使用 `WithNormalization()` 选项对特征向量进行归一化，这对余弦相似度计算很重要：

```go
Options: []hugot.FeatureExtractionOption{
    pipelines.WithNormalization(),
}
```

### 自定义相似度度量

除了余弦相似度，还可以使用：
- 欧氏距离: `math.Sqrt(sum((a[i]-b[i])^2))`
- 曼哈顿距离: `sum(|a[i]-b[i]|)`
- 点积: `sum(a[i]*b[i])` (归一化后等同于余弦相似度)

### 实时分类系统

可以预先提取类别的参考向量，然后对新输入的文本实时分类：

```go
// 初始化时提取类别向量
categoryVectors := extractCategoryVectors(pipeline, categoryExamples)

// 实时分类
func classifyText(pipeline *pipelines.FeatureExtractionPipeline, 
                  text string, 
                  categoryVectors map[string][]float32) string {
    result, _ := pipeline.RunPipeline([]string{text})
    textVector := result.Embeddings[0]
    return findMostSimilar(textVector, categoryVectors)
}
```

## 常见问题

**Q: 特征向量应该归一化吗？**
A: 对于余弦相似度计算，是的，应该归一化。对于欧氏距离，归一化也是推荐的做法。

**Q: 如何选择合适的特征提取模型？**
A: 取决于你的需求：
- 速度快、资源少：选择 MiniLM 系列
- 准确率优先：选择 mpnet 或 BERT 系列
- 多语言支持：选择 multilingual 版本

**Q: 如何提高分类准确率？**
A: 
1. 增加类别示例数量
2. 使用更准确的特征提取模型
3. 使用专门的分类模型（方式3）
4. 对特征进行后处理（如降维）

## 参考资料

- [Hugot 官方文档](../README.md)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [Sentence Transformers](https://www.sbert.net/)

