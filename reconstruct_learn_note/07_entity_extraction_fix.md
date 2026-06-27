# LiCoMemory 实体提取问题修复记录

## 1. 问题描述

### 症状
- 图构建阶段提取了0个实体和0个关系
- 查询阶段返回"上下文信息不足"
- 准确率为0%

### 影响
- 整个系统无法正常工作
- 无法进行有效的问答

## 2. 问题诊断

### 2.1 添加调试代码

在`coregraph/dialogue_extractor.py`的`_parse_dialogue_response`方法中添加调试代码：

```python
# Debug: save raw response for analysis
import os
debug_dir = "./debug_outputs"
os.makedirs(debug_dir, exist_ok=True)
debug_file = os.path.join(debug_dir, "llm_responses.txt")
with open(debug_file, "a", encoding="utf-8") as f:
    f.write(f"{'='*80}\n")
    f.write(f"Response length: {len(response)}\n")
    f.write(f"Response:\n{response[:2000]}\n")
    f.write(f"{'='*80}\n\n")
```

### 2.2 分析LLM输出

运行调试后，发现LLM返回的内容格式如下：

```
<think>
Okay, let's tackle this step by step. The user provided a multi-turn chat transcript...
[很长的思考过程]
</think>
```

**问题根源**：Qwen3-8B-AWQ模型启用了thinking模式，返回的内容全部在`<think>`标签中，没有输出实际的实体关系格式。

### 2.3 期望的输出格式

根据提示词，期望的输出格式是：

```
("entity"|Alice|person)##
("entity"|Bob|person)##
("relationship"|2024-03-15|D1|Bob|Project|working on|8)##
##END##
```

## 3. 解决方案

### 3.1 方案1: 禁用thinking模式（采用）

在提示词开头添加`/no_think`指令：

```python
LOCOMO_EXTRACTION_PROMPT = """/no_think
You are a helpful assistant trying to extract entities and relations...
```

**优点**：
- 简单有效
- 不需要修改模型配置
- 兼容性好

**缺点**：
- 依赖模型对`/no_think`指令的支持
- 可能影响模型的推理能力

### 3.2 方案2: 修改API参数（备选）

在API调用时添加参数禁用thinking：

```python
response = await client.chat.completions.create(
    model=model_name,
    messages=[{"role": "user", "content": prompt}],
    extra_body={"enable_thinking": False}
)
```

**缺点**：
- 需要修改LLM调用代码
- 可能不被所有vLLM版本支持

### 3.3 方案3: 后处理提取（备选）

从thinking内容中提取实体关系：

```python
import re

def extract_from_thinking(response):
    # 尝试从thinking内容中提取实体关系
    # 移除think标签
    if '<think>' in response:
        match = re.search(r'<think>.*?</think>\s*(.*)', response, re.DOTALL)
        if match:
            return match.group(1).strip()
    return response
```

**缺点**：
- 可靠性差
- 需要复杂的解析逻辑

## 4. 实施修复

### 4.1 修改提示词文件

文件：`prompt/entity_prompt.py`

**修改1**: LOCOMO_EXTRACTION_PROMPT
```python
LOCOMO_EXTRACTION_PROMPT = """/no_think
You are a helpful assistant trying to extract entities and relations...
```

**修改2**: DIALOGUE_EXTRACTION_PROMPT
```python
DIALOGUE_EXTRACTION_PROMPT = """/no_think
You are a helpful assistant trying to extract entities and relations...
```

### 4.2 添加后处理逻辑（可选）

在`_parse_dialogue_response`方法中添加thinking标签处理：

```python
# Try to handle different response formats
# Check if response contains think tags (Qwen3 thinking mode)
if '<think>' in response and '</think>' in response:
    # Extract content after think tags
    import re
    think_match = re.search(r'<think>.*?</think>\s*(.*)', response, re.DOTALL)
    if think_match:
        response = think_match.group(1).strip()
        logger.debug(f"Extracted content after think tags: {response[:200]}")
```

## 5. 验证修复

### 5.1 测试结果

修复后运行测试：

```
Stage 2: Dialogue Entity & Relationship Extraction: 100%|██████████| 6/6 [00:04<00:00, 1.33calls/s]
Extracted 23 entities and 15 relationships from 6 dialogue chunks
Dialogue mode - After deduplication: 11 entities, 15 relationships
```

**成功**：
- 提取了23个实体和15个关系
- 去重后得到11个实体和15个关系
- 图构建成功

### 5.2 调试输出示例

修复后的LLM输出：

```
("entity"|Alice|person)##
("entity"|Bob|person)##
("entity"|Project|object)##
("relationship"|2024-03-15 00:00:00|D1|Bob|Project|working on|8)##
##END##
```

## 6. 相关问题

### 6.1 Embedding超时问题

**现象**：`Failed to get embeddings: Request timed out`

**原因**：Ollama的nomic-embed-text服务不稳定

**解决方案**：
1. 增加超时时间配置
2. 在代码中添加超时控制
3. 使用其他embedding服务

### 6.2 评估器配置错误

**现象**：`The model 'qwen3:14b' does not exist`

**原因**：评估器使用了旧的模型配置

**解决方案**：更新评估器配置使用vLLM服务

## 7. 经验总结

### 7.1 关键发现
1. **Qwen3模型默认启用thinking模式**：会返回`<think>`标签内容
2. **`/no_think`指令有效**：可以禁用thinking模式
3. **调试代码很重要**：帮助快速定位问题

### 7.2 最佳实践
1. **添加调试日志**：保存LLM原始输出
2. **测试不同模型**：不同模型行为可能不同
3. **验证输出格式**：确保解析逻辑正确
4. **处理边界情况**：如thinking模式、空输出等

### 7.3 预防措施
1. **在提示词中明确格式要求**
2. **添加输出格式验证**
3. **实现降级处理逻辑**
4. **记录详细的错误日志**

## 8. 代码变更记录

### 文件1: prompt/entity_prompt.py
- 在LOCOMO_EXTRACTION_PROMPT开头添加`/no_think`
- 在DIALOGUE_EXTRACTION_PROMPT开头添加`/no_think`

### 文件2: coregraph/dialogue_extractor.py
- 添加调试代码保存LLM输出
- 添加thinking标签处理逻辑

### 文件3: base/embeddings.py
- 添加超时控制
- 改进错误处理

### 文件4: config/Memory_local.yaml
- 增加embedding超时配置
- 减少批处理大小

## 9. 测试建议

### 9.1 单元测试
- 测试不同格式的LLM输出
- 测试边界情况（空输出、格式错误等）
- 测试thinking模式处理

### 9.2 集成测试
- 测试完整的图构建流程
- 测试查询流程
- 测试评估流程

### 9.3 性能测试
- 测试不同并发数下的稳定性
- 测试超时处理
- 测试错误恢复

## 10. 参考资料

### 10.1 Qwen3模型文档
- thinking模式说明
- `/no_think`指令用法
- API参数配置

### 10.2 vLLM文档
- thinking模式配置
- API参数说明
- 错误处理

### 10.3 LiCoMemory代码
- 提示词设计
- 解析逻辑
- 错误处理
