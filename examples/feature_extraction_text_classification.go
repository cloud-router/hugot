package main

import (
	"encoding/json"
	"fmt"
	"math"

	"github.com/knights-analytics/hugot"
	"github.com/knights-analytics/hugot/pipelines"
)

// check 辅助函数，用于错误检查
func check(err error) {
	if err != nil {
		panic(err.Error())
	}
}

// 方式1: 使用特征提取 + 余弦相似度进行分类
// 这种方法适用于少样本学习（few-shot learning）或基于相似度的分类

// ClassificationBySimilarity 使用特征提取和余弦相似度进行分类
func ClassificationBySimilarity() {
	fmt.Println("=== 方式1: 使用特征提取 + 余弦相似度进行分类 ===\n")

	// 创建会话
	session, err := hugot.NewGoSession()
	// 如果使用 ORT 后端（需要编译时标签 ORT）：
	// session, err := hugot.NewORTSession()
	check(err)
	defer func(session *hugot.Session) {
		err := session.Destroy()
		check(err)
	}(session)

	// 下载或使用已存在的特征提取模型
	// 这里使用 all-MiniLM-L6-v2，这是一个轻量级的句子嵌入模型
	options := hugot.NewDownloadOptions()
	options.OnnxFilePath = "onnx/model.onnx"
	// options.ExternalDataPath = "onnx/model.onnx.data"
	modelPath, err := hugot.DownloadModel(
		"sentence-transformers/all-MiniLM-L6-v2",
		"./models/",
		options,
	)
	check(err)

	// 创建特征提取管道
	config := hugot.FeatureExtractionConfig{
		ModelPath:    modelPath,
		Name:         "featureExtraction",
		OnnxFilename: "model.onnx",
		Options: []hugot.FeatureExtractionOption{
			pipelines.WithNormalization(), // 归一化向量，便于计算余弦相似度
		},
	}
	pipeline, err := hugot.NewPipeline(session, config)
	check(err)

	// 定义分类的示例文本（few-shot learning）
	// 每个类别提供几个示例文本
	classExamples := map[string][]string{
		"正面": {
			"This movie is excellent!",
			"I love this product.",
			"Great service, highly recommended!",
		},
		"负面": {
			"This is terrible.",
			"I hate this product.",
			"Poor quality, very disappointed.",
		},
		"code": {
			"開發一個PHP腳本，用於終止與MySQL資料庫伺服器的活躍連接。",
			"給定一個以十六進制格式表示的數字列表，請編寫一個程式來識別所有可以被15整除的數字。myHexList = ['A', 'B', 'C', 'D', 'E', 'F', '10', '11', '12', '13', '14', '15']",
			"編寫一個遞迴程序，不僅要反向輸出初始化的字元序列的各個元素，還要將所有字母轉換為奇偶數對應的 ASCII 值。若為奇數，則保持原樣；若為偶數，則加一。特定字符串為 'Hello World'。",
		},
		"math": {
			"求所有满足以下条件的$p$的值，使得对于每个$q>0$，我们有 $$\frac{3(pq^2+p^2q+3q^2+3pq)}{p+q}>2p^2q?$$ 用区间表示法以十进制形式表达你的答案。",
			"小明有 5 个苹果，他想分享给他的 3 个朋友。他每个朋友应该得到多少个苹果？",
			"每个立方体的面都被涂成红色或蓝色，每种颜色的概率都是1/2。每个面的颜色是独立确定的。涂色的立方体可以放在水平表面上，使得四个竖直面都是同一种颜色的概率是多少？",
			"一个园丁在一排中种植了三棵枫树、四棵橡树和五棵桦树。他将它们随机排列，每种排列方式的可能性相同。设 $\frac{m}{n}$ 为最简分数形式的概率，表示没有两棵桦树相邻。求 $m+n$。",
		},
		"medicine": {
			"头上长疙瘩的主要表现是什么？",
			"感冒频繁打喷嚏怎么办？",
			"使用清咽饮茶后可能会出现什么不良反应?",
		},
		"中性": {
			"This is okay.",
			"The product is average.",
			"It's neither good nor bad.",
		},
	}

	// 提取每个类别的示例向量
	classVectors := make(map[string][]float32)
	for class, examples := range classExamples {
		// 提取所有示例的嵌入向量
		result, err := pipeline.RunPipeline(examples)
		check(err)

		// 计算类别的平均向量（也可以使用其他聚合方法）
		avgVector := averageVector(result.Embeddings)
		classVectors[class] = avgVector
		fmt.Printf("类别 '%s' 的向量维度: %d\n", class, len(avgVector))
	}

	// 待分类的文本
	testTexts := []string{
		"This is a wonderful movie!",
		"The product quality is very poor.",
		"I think it's acceptable.",
		"小明有5根笔，小红有3根笔，他们两个一共有几根笔？",
		"從名為'orders'的表中，構造一個SQL查詢來獲取每個客戶的個別名稱以及他們的訂單總數。",
	}

	// 对每个测试文本进行分类
	for _, text := range testTexts {
		// 提取测试文本的嵌入向量
		result, err := pipeline.RunPipeline([]string{text})
		check(err)
		textVector := result.Embeddings[0]

		// 计算与每个类别的余弦相似度
		bestClass := ""
		bestScore := float32(-1.0)
		scores := make(map[string]float32)

		for class, classVector := range classVectors {
			similarity := cosineSimilarity(textVector, classVector)
			scores[class] = similarity
			if similarity > bestScore {
				bestScore = similarity
				bestClass = class
			}
		}

		// 输出结果
		fmt.Printf("\n文本: \"%s\"\n", text)
		fmt.Printf("分类结果: %s (相似度: %.4f)\n", bestClass, bestScore)
		fmt.Println("所有类别得分:")
		for class, score := range scores {
			fmt.Printf("  %s: %.4f\n", class, score)
		}
	}
}

// 方式2: 使用特征提取获取嵌入，然后使用传统的机器学习分类器
// 这种方法需要先提取所有文本的特征，然后训练一个分类器（如 SVM、逻辑回归等）
// 这里展示如何提取特征，实际的分类器训练需要使用外部库（如 gonum 或 goml）

// ExtractFeaturesForTraining 提取特征用于后续训练分类器
func ExtractFeaturesForTraining() {
	fmt.Println("\n\n=== 方式2: 使用特征提取准备训练数据 ===\n")

	session, err := hugot.NewGoSession()
	check(err)
	defer func(session *hugot.Session) {
		err := session.Destroy()
		check(err)
	}(session)

	// 下载特征提取模型
	options := hugot.NewDownloadOptions()
	options.OnnxFilePath = "onnx/model.onnx"
	modelPath, err := hugot.DownloadModel(
		"sentence-transformers/all-MiniLM-L6-v2",
		"./models/",
		options,
	)
	check(err)

	// 创建特征提取管道
	config := hugot.FeatureExtractionConfig{
		ModelPath:    modelPath,
		Name:         "featureExtraction",
		OnnxFilename: "model.onnx",
		Options: []hugot.FeatureExtractionOption{
			pipelines.WithNormalization(),
		},
	}
	pipeline, err := hugot.NewPipeline(session, config)
	check(err)

	// 模拟训练数据：文本和对应的标签
	trainingData := []struct {
		Text  string
		Label string
	}{
		{"This is great!", "正面"},
		{"I love it!", "正面"},
		{"Amazing product!", "正面"},
		{"This is terrible.", "负面"},
		{"I hate this.", "负面"},
		{"Poor quality.", "负面"},
		{"It's okay.", "中性"},
		{"Not bad.", "中性"},
	}

	// 提取所有文本的特征向量
	type TrainingExample struct {
		Features []float32 `json:"features"`
		Label    string    `json:"label"`
		Text     string    `json:"text"`
	}

	var examples []TrainingExample
	texts := make([]string, len(trainingData))
	for i, data := range trainingData {
		texts[i] = data.Text
	}

	// 批量提取特征（更高效）
	result, err := pipeline.RunPipeline(texts)
	check(err)

	// 组织训练示例
	for i, embedding := range result.Embeddings {
		examples = append(examples, TrainingExample{
			Features: embedding,
			Label:    trainingData[i].Label,
			Text:     trainingData[i].Text,
		})
	}

	// 将特征保存为 JSON（可用于训练分类器）
	jsonData, err := json.MarshalIndent(examples, "", "  ")
	check(err)

	fmt.Println("提取的特征向量示例:")
	fmt.Printf("特征维度: %d\n", len(examples[0].Features))
	fmt.Printf("\n第一个训练示例:\n")
	firstExampleJSON, err := json.MarshalIndent(examples[0], "", "  ")
	check(err)
	fmt.Println(string(firstExampleJSON))
	fmt.Printf("\n训练样本总数: %d, JSON 大小: %d 字节\n", len(examples), len(jsonData))

	// 保存到文件（可选）
	// os.WriteFile("training_features.json", jsonData, 0644)
	// fmt.Println("\n特征已保存到 training_features.json")

	fmt.Println("\n提示: 您可以使用这些特征向量配合以下工具训练分类器:")
	fmt.Println("  - gonum.org/v1/gonum: 用于 SVM 或逻辑回归")
	fmt.Println("  - github.com/sjwhitworth/golearn: 完整的机器学习库")
	fmt.Println("  - 或者使用 Python 的 scikit-learn 处理提取的特征")
}

// 方式3: 直接使用 TextClassificationPipeline（最简单的方式）
// 如果您有预训练的分类模型，这是推荐的方式

func DirectTextClassification() {
	fmt.Println("\n\n=== 方式3: 直接使用 TextClassificationPipeline ===\n")

	session, err := hugot.NewGoSession()
	check(err)
	defer func(session *hugot.Session) {
		err := session.Destroy()
		check(err)
	}(session)

	// 下载文本分类模型
	modelPath, err := hugot.DownloadModel(
		"distilbert-base-uncased-finetuned-sst-2-english",
		"./models/",
		hugot.NewDownloadOptions(),
	)
	check(err)

	// 创建文本分类管道
	config := hugot.TextClassificationConfig{
		ModelPath: modelPath,
		Name:      "textClassification",
		Options: []hugot.TextClassificationOption{
			pipelines.WithSoftmax(),
			pipelines.WithSingleLabel(), // 单标签分类
		},
	}
	pipeline, err := hugot.NewPipeline(session, config)
	check(err)

	// 待分类的文本
	testTexts := []string{
		"This movie is disgustingly good!",
		"The director tried too much",
		"This is a neutral comment",
	}

	// 进行分类
	result, err := pipeline.RunPipeline(testTexts)
	check(err)

	// 输出结果
	for i, text := range testTexts {
		fmt.Printf("文本: \"%s\"\n", text)
		if len(result.ClassificationOutputs[i]) > 0 {
			output := result.ClassificationOutputs[i][0]
			fmt.Printf("  类别: %s, 得分: %.4f\n", output.Label, output.Score)
		}
		fmt.Println()
	}
}

// 辅助函数：计算向量平均值
func averageVector(vectors [][]float32) []float32 {
	if len(vectors) == 0 {
		return nil
	}
	dim := len(vectors[0])
	avg := make([]float32, dim)
	for _, v := range vectors {
		for i := range v {
			avg[i] += v[i]
		}
	}
	for i := range avg {
		avg[i] /= float32(len(vectors))
	}
	return avg
}

// 辅助函数：计算余弦相似度
func cosineSimilarity(a, b []float32) float32 {
	if len(a) != len(b) {
		return 0
	}
	var dotProduct, normA, normB float32
	for i := range a {
		dotProduct += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}
	if normA == 0 || normB == 0 {
		return 0
	}
	return dotProduct / float32(math.Sqrt(float64(normA))*math.Sqrt(float64(normB)))
}

func main() {
	// 方式1: 使用特征提取 + 余弦相似度进行分类（few-shot learning）
	ClassificationBySimilarity()

	// 方式2: 提取特征用于训练分类器
	// ExtractFeaturesForTraining()

	// 方式3: 直接使用预训练的分类模型（推荐用于生产环境）
	// DirectTextClassification()
}
