# LiCoMemory 复现最终结果报告

## 1. 复现状态

### ✅ 图构建成功

**构建时间**: 33.93秒

**统计信息**:
- 实体提取时间: 3.89秒
- 摘要生成时间: 15.07秒
- 图构建时间: 0.00秒
- 总Token消耗: 10,691

### ✅ 知识图谱构建结果

**实体数量**: 11个

| 实体名称 | 实体类型 |
|---------|---------|
| Alice | person |
| Bob | person |
| Project | object |
| Client | person |
| Presentation | event |
| Proposal | object |
| AI model | object |
| Technology | concept |
| 70B parameters | concept |
| GPT-4 | object |
| Performance metrics | concept |

**关系数量**: 13个

| 源实体 | 目标实体 | 关系 |
|--------|---------|------|
| Alice | Presentation | attended |
| Alice | AI model | discussed |
| Alice | Project | mentioned |
| Alice | Bob | asked about |
| Bob | Project | mentioned |
| Bob | Presentation | attended |
| Bob | Client | presented to |
| Bob | Proposal | presented |
| Bob | Technology | talked about |
| Bob | AI model | provided information about |
| Bob | GPT-4 | outperforms |
| Bob | Performance metrics | discussed |
| AI model | 70B parameters | has |

## 2. 修复的关键问题

### 问题1: 实体提取失败

**根本原因**: Qwen3模型默认启用thinking模式，返回`<think>`标签内容

**解决方案**: 在提示词开头添加`/no_think`指令

**修复文件**:
- `prompt/entity_prompt.py`: 修改LOCOMO_EXTRACTION_PROMPT和DIALOGUE_EXTRACTION_PROMPT

**验证结果**: 
- 修复前: 0个实体, 0个关系
- 修复后: 23个实体, 15个关系（去重后11个实体, 15个关系）

### 问题2: Embedding服务不可用

**根本原因**: Ollama的nomic-embed-text服务异常

**解决方案**: 
1. 添加超时控制（30秒）
2. 超时后跳过embedding预计算
3. 查询阶段使用字符串匹配fallback

**修复文件**:
- `coregraph/dynamic_memory.py`: 添加embedding超时处理
- `base/embeddings.py`: 改进错误处理

### 问题3: 查询参数处理错误

**根本原因**: `-query 0`参数被错误解析为True

**解决方案**: 修改参数判断逻辑

**修复文件**:
- `main.py`: 修改`if args.query:`为`if args.query and args.query != "0":`

## 3. 系统架构验证

### 3.1 图构建流程

```
输入文档 → 会话摘要生成 → 文档分块 → 实体关系提取 → 图构建
                                                    ↓
                                              11个实体, 13个关系
```

**验证状态**: ✅ 成功

### 3.2 查询流程（待验证）

```
用户查询 → 查询理解 → 实体检索 → 关系检索 → 答案生成
```

**验证状态**: ⏳ 待解决Embedding问题

### 3.3 评估流程（待验证）

```
模型输出 → LLM评估 → 准确率计算
```

**验证状态**: ⏳ 待运行完整实验

## 4. 技术要点总结

### 4.1 LLM+KG架构

- **LLM用途**: 实体提取、关系提取、会话摘要、答案生成
- **知识图谱**: NetworkX存储实体和关系
- **检索增强**: 基于图的检索和重排序

### 4.2 Qwen3模型特性

- **Thinking模式**: 默认启用，需要`/no_think`指令禁用
- **输出格式**: 需要明确的格式要求
- **推理能力**: 适合复杂的信息抽取任务

### 4.3 Embedding技术

- **服务选择**: Ollama本地服务 vs HuggingFace API
- **超时处理**: 需要合理的超时设置和fallback机制
- **降级策略**: 字符串匹配作为embedding的备选方案

### 4.4 错误处理

- **超时控制**: asyncio.wait_for或httpx timeout
- **降级机制**: embedding失败时使用字符串匹配
- **日志记录**: 详细的错误日志和调试信息

## 5. 实验数据

### 5.1 数据集

- **类型**: LOCOMO模拟数据集
- **规模**: 2个对话组，5个会话，8个问答对
- **格式**: NDJSON格式

### 5.2 配置参数

```yaml
# LLM配置
llm:
  model: Qwen/Qwen3-8B-AWQ
  base_url: http://localhost:8910/v1
  max_concurrent: 2

# 图配置
graph:
  force: True
  entity_merge_threshold: 0.85

# Embedding配置
embedding:
  timeout: 10
```

### 5.3 性能指标

- **图构建时间**: 33.93秒
- **实体提取时间**: 3.89秒
- **摘要生成时间**: 15.07秒
- **Token消耗**: 10,691

## 6. 待解决问题

### 6.1 Embedding服务

**问题**: Ollama的nomic-embed-text服务不可用

**影响**: 无法进行语义搜索，只能使用字符串匹配

**解决方案**:
1. 重启Ollama服务
2. 使用其他Embedding服务
3. 使用本地HuggingFace模型

### 6.2 查询阶段

**问题**: 查询阶段因为Embedding超时而阻塞

**影响**: 无法完成完整实验

**解决方案**:
1. 修复Embedding服务
2. 优化超时处理
3. 使用更快的Embedding模型

### 6.3 评估阶段

**问题**: 评估器使用了错误的模型配置

**影响**: 无法正确评估答案质量

**解决方案**: 更新评估器配置使用vLLM服务

## 7. 后续优化建议

### 7.1 短期优化

1. 修复Embedding服务
2. 运行完整查询实验
3. 评估答案质量
4. 优化提示词

### 7.2 中期优化

1. 使用更强的LLM模型
2. 优化实体合并算法
3. 改进检索策略
4. 增加更多评估指标

### 7.3 长期优化

1. 支持更多数据集
2. 实现增量学习
3. 优化整体性能
4. 发布复现报告

## 8. 学习笔记索引

### 8.1 基础文档

1. `01_project_overview.md` - 项目概述
2. `02_environment_setup.md` - 环境配置指南
3. `03_technical_architecture.md` - 技术架构学习笔记
4. `04_experiment_results.md` - 实验结果分析
5. `05_troubleshooting.md` - 问题解决方案
6. `06_summary.md` - 复现总结

### 8.2 问题修复文档

7. `07_entity_extraction_fix.md` - 实体提取问题修复记录
8. `08_final_results.md` - 最终结果报告（本文件）

## 9. 代码变更记录

### 9.1 提示词修改

**文件**: `prompt/entity_prompt.py`
- 添加`/no_think`指令禁用thinking模式
- 修改LOCOMO_EXTRACTION_PROMPT
- 修改DIALOGUE_EXTRACTION_PROMPT

### 9.2 图构建修改

**文件**: `coregraph/dynamic_memory.py`
- 添加embedding超时处理
- 超时后跳过embedding预计算

### 9.3 Embedding修改

**文件**: `base/embeddings.py`
- 添加timeout属性读取
- 改进错误处理
- 使用httpx设置超时

### 9.4 主程序修改

**文件**: `main.py`
- 修复`-query 0`参数处理
- 修改判断逻辑

## 10. 经验总结

### 10.1 成功经验

1. **调试技巧**: 保存LLM原始输出有助于快速定位问题
2. **降级机制**: embedding失败时使用字符串匹配fallback
3. **超时控制**: 合理设置超时时间避免长时间阻塞
4. **日志记录**: 详细的日志帮助追踪问题

### 10.2 教训

1. **模型特性**: 不同LLM模型有不同的输出格式要求
2. **服务依赖**: 外部服务（如Ollama）可能不稳定
3. **参数处理**: 命令行参数需要仔细验证
4. **错误处理**: 需要完善的错误处理和降级机制

### 10.3 最佳实践

1. **先验证图构建**: 确保基础功能正常
2. **逐步测试**: 分阶段验证各个组件
3. **保存中间结果**: 便于调试和复现
4. **记录变更**: 详细记录代码修改

## 11. 致谢

感谢以下资源和工具：
- LiCoMemory项目团队
- Qwen3模型
- vLLM推理框架
- NetworkX图数据库
- 所有参考文档和教程

## 12. 附录

### 12.1 完整运行命令

```bash
# 图构建
python main.py -opt config/Memory_local.yaml -dataset_name group_1 -root graph_only -query 0

# 完整实验（需要Embedding服务）
python main.py -opt config/Memory_local.yaml -dataset_name group_1 -root full_run -query 1
```

### 12.2 配置文件示例

```yaml
# config/Memory_local.yaml
llm:
  model: Qwen/Qwen3-8B-AWQ
  base_url: http://localhost:8910/v1
  api_key: EMPTY

embedding:
  model: nomic-embed-text:latest
  base_url: http://localhost:11434/v1
  timeout: 10
```

### 12.3 结果文件位置

- 图文件: `results/graph_only2/None.pkl`
- 日志文件: `results/graph_only2/dynamic_memory.log`
- 摘要文件: `results/graph_only2/session_summaries.json`
