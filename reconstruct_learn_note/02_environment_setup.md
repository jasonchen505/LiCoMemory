# LiCoMemory 复现环境配置指南

## 1. 环境要求

### 系统要求
- Python 3.10+
- CUDA (可选，用于GPU加速)
- 足够的GPU内存（建议24GB+）

### 依赖包
主要依赖：
- openai>=1.0.0 (LLM API)
- torch>=2.0.0 (深度学习框架)
- networkx>=3.0 (图数据库)
- sentence-transformers>=2.2.0 (Embedding模型)
- pandas, numpy, matplotlib等

## 2. 虚拟环境配置

### 使用uv创建虚拟环境
```bash
cd /home/chenyizhou/LiCoMemory
uv venv .venv --python 3.10
source .venv/bin/activate
```

### 安装依赖
```bash
uv pip install -r requirements.txt
```

## 3. API配置

### LLM配置选项

#### 选项1: Ollama本地服务
```yaml
llm:
  api_type: "openai"
  model: 'qwen3:14b'
  base_url: "http://localhost:11434/v1"
  api_key: "ollama"
```

#### 选项2: vLLM服务
```yaml
llm:
  api_type: "openai"
  model: 'Qwen/Qwen3-8B-AWQ'
  base_url: "http://localhost:8910/v1"
  api_key: "EMPTY"
```

#### 选项3: OpenAI API
```yaml
llm:
  api_type: "openai"
  model: 'gpt-4'
  base_url: "https://api.openai.com/v1"
  api_key: "your-api-key"
```

### Embedding配置选项

#### 选项1: HuggingFace本地模型
```yaml
embedding:
  api_type: "hf"
  model: "BAAI/bge-large-zh-v1.5"
  cache_dir: "/path/to/cache"
  dimensions: 1024
```

#### 选项2: Ollama Embedding
```yaml
embedding:
  api_type: "openai"
  model: "nomic-embed-text:latest"
  base_url: "http://localhost:11434/v1"
  api_key: "ollama"
  dimensions: 768
```

#### 选项3: OpenAI Embedding
```yaml
embedding:
  api_type: "openai"
  model: "text-embedding-ada-002"
  base_url: "https://api.openai.com/v1"
  api_key: "your-api-key"
  dimensions: 1536
```

## 4. 数据集准备

### LOCOMO数据集
1. 下载数据集：
   - 从Hugging Face: `snap-research/locomo`
   - 或从GitHub: https://github.com/snap-research/locomo

2. 处理数据集：
```bash
python dataset/locomo.py --input locomo10.json --outdir dataset/locomo
```

3. 数据格式：
   - `group_X/Corpus.json`: 会话数据（NDJSON格式）
   - `group_X/Question.json`: 问答数据（NDJSON格式）

### LongmemEval数据集
参考 `dataset/longmem.py` 进行处理。

## 5. 运行实验

### 基本运行命令
```bash
python main.py \
  -opt config/Memory_local.yaml \
  -dataset_name group_1 \
  -root experiment_1 \
  -query 1
```

### 参数说明
- `-opt`: 配置文件路径
- `-dataset_name`: 数据集名称
- `-root`: 结果保存目录
- `-query`: 是否运行查询和评估（1=是，0=否）

## 6. 常见问题解决

### 问题1: HuggingFace模型下载失败
**原因**: 网络不可达
**解决方案**:
- 使用Ollama或vLLM的本地模型
- 配置HuggingFace镜像

### 问题2: LLM API调用失败
**原因**: 模型太大，GPU内存不足
**解决方案**:
- 使用更小的模型（如Qwen3-8B而非14B）
- 减少`max_concurrent`参数
- 使用量化模型（AWQ、GPTQ等）

### 问题3: OpenAI API版本不兼容
**原因**: openai>=1.0.0 API变更
**解决方案**:
- 更新代码使用新API
- 或降级openai版本: `pip install openai==0.28`

### 问题4: 实体提取失败
**原因**: LLM输出格式不符合预期
**解决方案**:
- 检查LLM模型能力
- 调整提示词
- 使用更强的模型

## 7. 性能优化建议

### GPU内存优化
- 使用量化模型（AWQ、GPTQ）
- 减少并发数
- 使用更小的模型

### 推理速度优化
- 使用vLLM等推理框架
- 启用批处理
- 使用更快的硬件

### 准确率优化
- 使用更强的LLM
- 调整提示词
- 优化实体合并阈值
- 启用CogniRank重排序

## 8. 代码修改记录

### 修改1: 修复OpenAI Embedding API
文件: `base/embeddings.py`
原因: openai>=1.0.0不再支持`openai.Embedding`
修改: 使用新的`openai.AsyncOpenAI`客户端

### 修改2: 添加base_url支持
文件: `base/embeddings.py`
原因: 支持Ollama等本地服务
修改: 从config中读取base_url参数

## 9. 实验结果分析

### 图构建统计
- 会话摘要生成时间
- 实体关系提取时间
- 图构建时间
- Token消耗

### 查询统计
- 实体检索时间
- 三元组检索时间
- 摘要检索时间
- 答案生成时间
- 总Token消耗

### 评估指标
- LLM评估准确率
- 精确匹配准确率
- 会话匹配分数
- 分类型准确率

## 10. 下一步计划

1. 修复评估器配置
2. 优化实体提取
3. 测试更大规模数据集
4. 对比不同LLM效果
5. 分析CogniRank效果
