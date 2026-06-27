# LiCoMemory 项目复现学习笔记

## 项目概述

LiCoMemory (Lightweight and Cognitive Agentic Memory) 是一个用于LLM的长期记忆框架，支持多会话对话的记忆存储和检索。

### 核心特性

1. **多会话记忆**: 高效存储和检索跨多个对话会话的信息
2. **动态图构建**: 支持实时图更新和检索增强
3. **层次化认知图谱 (Cognigraph)**: 维护多层轻量级图结构
4. **时间感知检索**: 统一重排序实现实体级、会话级和时间级相似度平衡

## 项目结构

```
LiCoMemory/
├── main.py                 # 主入口
├── config/                 # 配置文件
│   └── Memory.yaml        # 主配置文件
├── base/                   # 基础组件
│   ├── embeddings.py      # Embedding管理器
│   ├── llm.py             # LLM管理器
│   └── utils.py           # 工具函数
├── coregraph/              # 核心图谱组件
│   ├── dynamic_memory.py  # 动态记忆图
│   ├── entity_extractor.py # 实体提取器
│   ├── dialogue_extractor.py # 对话提取器
│   └── session_summarizer.py # 会话摘要器
├── query/                  # 查询组件
│   ├── query_processor.py # 查询处理器
│   ├── retriever.py       # 检索器
│   └── triple_reranker.py # 三元组重排序
├── evaluation/             # 评估组件
│   ├── evaluator.py       # 评估器
│   └── llm_evaluator.py   # LLM评估器
├── dataset/                # 数据集处理
│   ├── locomo.py          # LOCOMO数据集处理
│   └── longmem.py         # LongmemEval数据集处理
└── utils/                  # 工具类
    ├── cost_manager.py    # 成本管理
    ├── time_statistic.py  # 时间统计
    └── final_report.py    # 最终报告
```

## 核心流程

### 1. 图构建流程 (Graph Building)

1. **会话摘要生成** (Stage 1)
   - 为每个对话会话生成摘要
   - 使用LLM提取关键信息

2. **文档分块** (Chunking)
   - 将对话文档分割成chunks
   - 支持对话格式的特殊处理

3. **实体和关系提取** (Stage 2)
   - 从每个chunk中提取实体和关系
   - 使用LLM进行信息抽取
   - 支持对话模式的特殊提示词

4. **图构建**
   - 使用NetworkX构建图
   - 实体作为节点，关系作为边
   - 支持增量更新

### 2. 查询流程 (Query Processing)

1. **查询理解**
   - 解析用户问题
   - 提取查询意图

2. **检索**
   - 实体检索：找到相关实体
   - 关系检索：找到相关三元组
   - 会话摘要检索：找到相关会话

3. **重排序**
   - CogniRank：统一重排序算法
   - 考虑实体、会话、时间三个维度

4. **答案生成**
   - 使用检索到的上下文
   - 调用LLM生成答案

## 配置说明

### LLM配置

```yaml
llm:
  api_type: "openai"  # API类型
  model: 'qwen3:14b'  # 模型名称
  base_url: "http://localhost:11434/v1"  # API地址
  api_key: "ollama"   # API密钥
  max_token: 8192     # 最大token数
  temperature: 0.0    # 温度参数
  enable_concurrent: True  # 启用并发
  max_concurrent: 8   # 最大并发数
```

### Embedding配置

```yaml
embedding:
  api_type: "openai"  # 支持"openai"或"hf"
  model: "nomic-embed-text:latest"  # 模型名称
  dimensions: 768     # 向量维度
  max_token_size: 8102  # 最大token数
  embed_batch_size: 128  # 批处理大小
```

### 图配置

```yaml
graph:
  graph_type: dynamic_memory  # 图类型
  force: True  # 强制重建
  add: False   # 增量添加
  entity_merge_threshold: 0.85  # 实体合并阈值
  relationship_merge_threshold: 1  # 关系合并阈值
```

### 检索配置

```yaml
retriever:
  top_k: 5  # 返回的top-k结果数
  top_k_triples: 20  # 返回的三元组数
  top_chunks: 15  # 返回的chunk数
  enable_summary: True  # 启用摘要检索
  enable_CogniRank: False  # 启用CogniRank
```

## 数据集格式

### LOCOMO数据集

LOCOMO数据集包含多会话对话和问答对：

```json
{
  "conversation": {
    "session_1": [...],
    "session_1_date_time": "2:30 pm on 15 March, 2024",
    "session_2": [...],
    "session_2_date_time": "10:15 am on 20 March, 2024"
  },
  "qa": [
    {
      "question": "...",
      "answer": "...",
      "evidence": ["D1:3"],
      "category": "1"
    }
  ]
}
```

处理后的格式：

**Corpus.json** (每行一个JSON):
```json
{
  "session_time": "2024/03/15",
  "context": "\"Alice\": \"Hi...\"\"Bob\": \"Hello...\"",
  "session_id": "D1"
}
```

**Question.json** (每行一个JSON):
```json
{
  "question": "...",
  "answer": "...",
  "question_type": "1",
  "origin": "D1",
  "label": "..."
}
```

## 复现过程中的问题与解决方案

### 问题1: HuggingFace模型下载失败

**现象**: 网络不可达，无法下载HuggingFace模型

**解决方案**: 
- 使用Ollama的本地模型
- 配置`api_type: "openai"`，使用Ollama的OpenAI兼容API

### 问题2: LLM API调用失败

**现象**: `llama runner process has terminated`

**原因**: qwen3:14b模型太大，内存不足

**解决方案**:
- 使用更小的模型（如qwen2.5:0.5b）
- 或减少`max_concurrent`参数

### 问题3: 路径安全问题

**现象**: `insecure path` 错误

**原因**: Ollama对路径安全检查

**解决方案**:
- 将模型文件复制到Ollama允许的目录
- 或使用Ollama pull下载模型

## 学习要点

### 1. LLM+KG架构

LiCoMemory展示了如何将LLM与知识图谱结合：
- 使用LLM进行信息抽取（实体、关系）
- 使用图结构存储知识
- 使用检索增强生成（RAG）回答问题

### 2. 层次化图结构

Cognigraph的设计：
- 实体层：存储命名实体
- 关系层：存储实体间关系
- 会话层：存储会话摘要
- 时间层：时间戳信息

### 3. 时间感知检索

考虑时间因素的检索：
- 时间相近的对话权重更高
- 支持时间推理问题

### 4. 统一重排序 (CogniRank)

多维度相似度融合：
- 实体相似度
- 会话相似度
- 时间相似度
- 加权融合

## 下一步计划

1. ✅ 创建虚拟环境
2. ✅ 安装依赖
3. ✅ 配置API
4. ✅ 准备数据集
5. ⏳ 运行完整实验
6. ⏳ 分析结果
7. ⏳ 记录详细笔记

## 参考资料

- [LiCoMemory GitHub](https://github.com/jasonchen505/LiCoMemory)
- [LOCOMO数据集](https://github.com/snap-research/locomo)
- [LongmemEval数据集](https://github.com/xiaowu0162/LongMemEval)
