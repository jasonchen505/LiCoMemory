# LiCoMemory 复现问题解决方案

## 1. 环境配置问题

### 问题1.1: HuggingFace模型下载失败

**现象**:
```
'[Errno 101] Network is unreachable' thrown while requesting HEAD https://huggingface.co/...
```

**原因**:
- 网络不可达
- HuggingFace被墙
- 防火墙限制

**解决方案**:

#### 方案1: 使用本地模型
```yaml
embedding:
  api_type: "hf"
  model: "/path/to/local/model"
  cache_dir: "/path/to/cache"
```

#### 方案2: 使用Ollama
```yaml
embedding:
  api_type: "openai"
  model: "nomic-embed-text:latest"
  base_url: "http://localhost:11434/v1"
  api_key: "ollama"
```

#### 方案3: 配置镜像
```python
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
```

#### 方案4: 使用代理
```python
import os
os.environ['HTTP_PROXY'] = 'http://proxy:port'
os.environ['HTTPS_PROXY'] = 'http://proxy:port'
```

### 问题1.2: OpenAI API版本不兼容

**现象**:
```
You tried to access openai.Embedding, but this is no longer supported in openai>=1.0.0
```

**原因**:
- openai库版本太新（>=1.0.0）
- API接口变更

**解决方案**:

#### 方案1: 更新代码（推荐）
修改 `base/embeddings.py`:
```python
# 旧代码
response = await self.client.Embedding.acreate(
    input=texts,
    model=self.model_name
)
embeddings = [data.embedding for data in response.data]

# 新代码
import openai
client = openai.AsyncOpenAI(
    api_key=self.api_key,
    base_url=self.base_url
)
response = await client.embeddings.create(
    input=texts,
    model=self.model_name
)
embeddings = [item.embedding for item in response.data]
```

#### 方案2: 降级openai版本
```bash
pip install openai==0.28
```

### 问题1.3: GPU内存不足

**现象**:
```
CUDA out of memory
model failed to load, this may be due to resource limitations
```

**原因**:
- 模型太大
- GPU显存不足
- 其他进程占用显存

**解决方案**:

#### 方案1: 使用更小的模型
```yaml
llm:
  model: 'Qwen/Qwen3-8B-AWQ'  # 而不是14B
```

#### 方案2: 使用量化模型
```yaml
llm:
  model: 'Qwen/Qwen3-8B-AWQ'  # AWQ量化
```

#### 方案3: 减少并发数
```yaml
llm:
  max_concurrent: 1  # 而不是8
```

#### 方案4: 使用CPU模式
```yaml
embedding:
  device: "cpu"
```

#### 方案5: 清理GPU显存
```bash
# 查看GPU使用情况
nvidia-smi

# 杀死占用显存的进程
kill -9 <pid>
```

## 2. LLM调用问题

### 问题2.1: LLM API调用失败

**现象**:
```
LLM API call failed: Error code: 500 - {'error': {'message': 'llama runner process has terminated: %!w(<nil>)'}}
```

**原因**:
- 模型加载失败
- 内存不足
- 模型文件损坏

**解决方案**:

#### 方案1: 检查模型状态
```bash
ollama list
ollama show <model_name>
```

#### 方案2: 重新下载模型
```bash
ollama pull <model_name>
```

#### 方案3: 使用其他模型
```bash
ollama pull qwen2.5:0.5b  # 更小的模型
```

#### 方案4: 检查Ollama日志
```bash
journalctl -u ollama -f
```

### 问题2.2: LLM输出格式错误

**现象**:
- 输出不是JSON格式
- 输出包含额外文本
- 输出格式不一致

**原因**:
- 提示词不够清晰
- 模型能力不足
- 输出约束不明确

**解决方案**:

#### 方案1: 优化提示词
```
请严格按照以下JSON格式返回结果，不要添加任何其他文本：
{
  "entities": [...],
  "relationships": [...]
}
```

#### 方案2: 增加输出约束
```
只返回JSON，不要返回其他任何内容。
```

#### 方案3: 使用更强模型
```yaml
llm:
  model: 'Qwen/Qwen3-14B-AWQ'
```

#### 方案4: 增加后处理
```python
import json
import re

def extract_json(text):
    # 尝试提取JSON
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except:
            pass
    return None
```

### 问题2.3: LLM响应超时

**现象**:
```
TimeoutError: LLM request timed out
```

**原因**:
- 模型推理慢
- 网络延迟
- 服务器负载高

**解决方案**:

#### 方案1: 增加超时时间
```yaml
llm:
  timeout: 1200  # 20分钟
```

#### 方案2: 减少输入长度
```yaml
chunk:
  chunk_token_size: 600  # 而不是1200
```

#### 方案3: 使用更快的模型
```yaml
llm:
  model: 'Qwen/Qwen3-8B-AWQ'  # 更小的模型
```

#### 方案4: 使用更快的推理框架
- vLLM
- TensorRT-LLM
- TGI

## 3. 数据处理问题

### 问题3.1: 数据格式错误

**现象**:
```
Error loading dataset: JSON decode error
```

**原因**:
- JSON格式错误
- 文件编码问题
- 字段缺失

**解决方案**:

#### 方案1: 验证JSON格式
```python
import json

with open('data.json', 'r') as f:
    try:
        data = json.load(f)
        print("JSON格式正确")
    except json.JSONDecodeError as e:
        print(f"JSON格式错误: {e}")
```

#### 方案2: 检查文件编码
```bash
file -i data.json
```

#### 方案3: 检查必需字段
```python
required_fields = ['question', 'answer', 'question_type']
for item in data:
    for field in required_fields:
        if field not in item:
            print(f"缺少字段: {field}")
```

### 问题3.2: 数据集路径错误

**现象**:
```
FileNotFoundError: No such file or directory
```

**原因**:
- 路径配置错误
- 文件不存在
- 权限问题

**解决方案**:

#### 方案1: 检查路径配置
```yaml
data_root: /home/chenyizhou/LiCoMemory/dataset/locomo
dataset_name: group_1
```

#### 方案2: 验证文件存在
```bash
ls -la /home/chenyizhou/LiCoMemory/dataset/locomo/group_1/
```

#### 方案3: 检查文件权限
```bash
chmod 644 *.json
```

## 4. 图构建问题

### 问题4.1: 实体提取失败

**现象**:
- 提取实体数为0
- 图为空

**原因**:
- LLM输出格式错误
- 解析逻辑错误
- 提示词问题

**解决方案**:

#### 方案1: 调试LLM输出
```python
# 保存LLM原始输出
with open('debug_output.txt', 'w') as f:
    f.write(llm_output)
```

#### 方案2: 优化提示词
```
从以下对话中提取实体和关系。

实体类型：人物、地点、组织、时间、事件、概念
关系类型：属于、发生在、参与、包含、位于、相关

请以JSON格式返回：
{
  "entities": [
    {"name": "实体名", "type": "类型"}
  ],
  "relationships": [
    {"source": "源实体", "target": "目标实体", "relation": "关系"}
  ]
}

对话内容：
{dialogue}
```

#### 方案3: 增加示例
```
示例输入：
"Alice和Bob讨论了项目进展。"

示例输出：
{
  "entities": [
    {"name": "Alice", "type": "人物"},
    {"name": "Bob", "type": "人物"},
    {"name": "项目", "type": "概念"}
  ],
  "relationships": [
    {"source": "Alice", "target": "项目", "relation": "讨论"},
    {"source": "Bob", "target": "项目", "relation": "讨论"}
  ]
}
```

### 问题4.2: 关系合并失败

**现象**:
- 重复关系
- 关系冲突
- 图不一致

**原因**:
- 合并阈值不当
- 去重逻辑错误
- 冲突解决策略不当

**解决方案**:

#### 方案1: 调整合并阈值
```yaml
graph:
  entity_merge_threshold: 0.9  # 更严格的阈值
  relationship_merge_threshold: 0.95
```

#### 方案2: 优化合并算法
```python
def merge_entities(entity1, entity2, threshold=0.85):
    similarity = calculate_similarity(entity1, entity2)
    if similarity > threshold:
        return combine_entities(entity1, entity2)
    return entity1, entity2
```

#### 方案3: 增加冲突解决
```python
def resolve_conflict(relation1, relation2):
    # 保留最新的关系
    if relation1['timestamp'] > relation2['timestamp']:
        return relation1
    return relation2
```

## 5. 查询问题

### 问题5.1: 检索结果为空

**现象**:
- 检索到0个结果
- 答案为"上下文信息不足"

**原因**:
- 图为空
- 相似度阈值太高
- 检索参数不当

**解决方案**:

#### 方案1: 检查图状态
```python
print(f"图节点数: {G.number_of_nodes()}")
print(f"图边数: {G.number_of_edges()}")
```

#### 方案2: 降低相似度阈值
```yaml
retriever:
  top_k: 10  # 增加检索数量
  top_k_triples: 50
```

#### 方案3: 优化检索策略
```yaml
retriever:
  enable_full: True  # 启用全图检索
  enable_summary: True  # 启用摘要检索
```

### 问题5.2: 答案质量差

**现象**:
- 答案不准确
- 答案不完整
- 答案不相关

**原因**:
- 检索结果不相关
- 上下文不足
- 提示词问题

**解决方案**:

#### 方案1: 优化检索
```yaml
retriever:
  top_chunks: 20  # 增加上下文数量
  enable_CogniRank: True  # 启用重排序
```

#### 方案2: 优化提示词
```
基于以下信息，详细回答问题。如果信息不足，请明确指出。

相关三元组：
{triples}

相关文本：
{chunks}

问题：{question}

请提供准确、完整的答案：
```

#### 方案3: 增加验证
```python
def validate_answer(answer, context):
    # 检查答案是否基于上下文
    if not is_based_on_context(answer, context):
        return "信息不足，无法回答"
    return answer
```

## 6. 评估问题

### 问题6.1: 评估器配置错误

**现象**:
```
The model `qwen3:14b` does not exist.
```

**原因**:
- 评估器使用了错误的模型配置
- 模型名称不匹配

**解决方案**:

#### 方案1: 更新配置
```yaml
evaluation:
  eval_model: 'Qwen/Qwen3-8B-AWQ'
  eval_base_url: "http://localhost:8910/v1"
  eval_api_key: "EMPTY"
```

#### 方案2: 修改代码
在评估器中添加配置支持：
```python
class LLMEvaluator:
    def __init__(self, config, ...):
        self.model = config.evaluation.eval_model
        self.base_url = config.evaluation.eval_base_url
        self.api_key = config.evaluation.eval_api_key
```

### 问题6.2: 评估结果不准确

**现象**:
- 准确率异常
- 评估结果不稳定
- 评估标准不一致

**原因**:
- 评估模型能力不足
- 评估提示词问题
- 评估标准不明确

**解决方案**:

#### 方案1: 使用更强的评估模型
```yaml
evaluation:
  eval_model: 'Qwen/Qwen3-14B-AWQ'
```

#### 方案2: 优化评估提示词
```
请判断以下答案是否正确。

问题：{question}
标准答案：{reference}
模型答案：{prediction}

判断标准：
1. 答案是否包含关键信息
2. 答案是否语义一致
3. 答案是否准确无误

请只回答"正确"或"错误"。
```

#### 方案3: 增加评估样本
```yaml
evaluation:
  num_samples: 100  # 增加评估样本数
```

## 7. 性能问题

### 问题7.1: 处理速度慢

**现象**:
- 图构建耗时长
- 查询响应慢
- 整体效率低

**原因**:
- 模型推理慢
- 并发度低
- 算法效率低

**解决方案**:

#### 方案1: 使用更快的推理框架
- vLLM
- TensorRT-LLM
- TGI

#### 方案2: 增加并发数
```yaml
llm:
  max_concurrent: 8  # 增加并发数
```

#### 方案3: 优化算法
- 使用索引
- 缓存结果
- 批处理
- 增量处理

### 问题7.2: 内存占用高

**现象**:
- 内存溢出
- 系统变慢
- 交换频繁

**原因**:
- 数据量大
- 缓存过多
- 内存泄漏

**解决方案**:

#### 方案1: 流式处理
```python
for chunk in large_dataset:
    process(chunk)
    gc.collect()
```

#### 方案2: 限制缓存大小
```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def expensive_function(x):
    return x * 2
```

#### 方案3: 使用生成器
```python
def process_data(data):
    for item in data:
        yield process(item)
```

## 8. 调试技巧

### 8.1 日志记录
```python
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='debug.log'
)
```

### 8.2 中间结果保存
```python
def save_intermediate_results(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
```

### 8.3 性能分析
```python
import cProfile

def profile_function(func):
    def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        profiler.enable()
        result = func(*args, **kwargs)
        profiler.disable()
        profiler.print_stats()
        return result
    return wrapper
```

### 8.4 断言检查
```python
def validate_data(data):
    assert isinstance(data, list), "数据应该是列表"
    assert len(data) > 0, "数据不能为空"
    for item in data:
        assert 'question' in item, "缺少question字段"
        assert 'answer' in item, "缺少answer字段"
```

## 9. 最佳实践

### 9.1 开发流程
1. 先在小数据集上测试
2. 逐步增加复杂度
3. 详细记录日志
4. 定期保存中间结果
5. 使用版本控制

### 9.2 测试策略
1. 单元测试
2. 集成测试
3. 性能测试
4. 回归测试

### 9.3 部署考虑
1. 环境隔离
2. 依赖管理
3. 配置管理
4. 监控告警

## 10. 参考资源

### 10.1 官方文档
- LiCoMemory GitHub
- NetworkX文档
- OpenAI API文档
- HuggingFace文档

### 10.2 社区资源
- GitHub Issues
- Stack Overflow
- 技术博客
- 论坛讨论

### 10.3 学习资源
- 相关论文
- 技术教程
- 在线课程
- 实践案例
