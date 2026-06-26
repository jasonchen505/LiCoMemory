# LiCoMemory 增量学习笔记

> 基于前两轮学习（INTERVIEW_GUIDE.md, INTERVIEW_FIVE_CATEGORIES.md）和之前复现经验（reconstruct_learn_note/），
> 本轮新增的深度理解和发现

---

## 一、对比前两轮学习的新增知识点

### 1.1 从"理解架构"到"理解设计决策"

| 维度 | 第一轮学习 (复现阶段) | 第二轮学习 (面试准备) | 本轮新增 |
|------|---------------------|---------------------|---------|
| **CogniRank** | 知道公式 | 理解为什么用中位数 | 理解 k 参数的物理意义：k=0.1 时衰减极平缓，即使时间差10倍中位数仍有0.28权重 |
| **实体抽取** | 知道用 LLM | 理解联合 vs 流水线 | 发现 Qwen3 的 thinking 模式会破坏输出格式，需要 `/no_think` |
| **去重策略** | 知道用 Jaccard | 理解阈值选择依据 | 发现 Jaccard 对短字符串不友好：'Coffee Shop' vs 'The Coffee Shop' 只有 0.5 |
| **存储选择** | 知道用 Pickle | 理解 Pickle vs JSON | 理解 Pickle 的安全隐患：反序列化可执行任意代码 |

### 1.2 代码层面的新发现

#### 发现1: Qwen3 的 `/no_think` 指令

**来源**: `reconstruct_learn_note/07_entity_extraction_fix.md`

```python
# 原始 prompt (entity_prompt.py)
DIALOGUE_EXTRACTION_PROMPT = """You are a helpful assistant..."""

# 修复后
DIALOGUE_EXTRACTION_PROMPT = """/no_think
You are a helpful assistant..."""
```

**深层理解**:
- Qwen3 默认启用 "thinking" 模式，会在输出中添加 `<think>...</think>` 标签
- 这些标签会被实体解析逻辑误解析，导致提取 0 个实体
- `/no_think` 是 Qwen3 的特殊指令，禁用 thinking 模式
- **这是 LLM 特定行为的工程适配**，不是通用问题

#### 发现2: Embedding 服务的降级策略

**来源**: `coregraph/dynamic_memory.py:605-628` + `reconstruct_learn_note/08_final_results.md`

```python
# 动态记忆中的 embedding 预计算
async def _precompute_embeddings(self, entities, relationships):
    try:
        await self._precompute_entity_embeddings(entities)
        await self._precompute_relationship_embeddings(relationships)
        await self._precompute_summary_embeddings()
    except Exception as e:
        logger.error(f"Error during embedding precomputation: {e}")
        # 降级：跳过预计算，查询时使用字符串匹配
```

**新增理解**:
- Embedding 失败不会阻断图构建流程
- 查询阶段有 fallback：`_find_similar_entities_fallback` 使用字符串匹配
- 这是一个**优雅降级**的设计：核心功能（图构建）不依赖 Embedding

#### 发现3: 实体抽取的并发控制

**来源**: `base/llm.py:159-238`, `coregraph/entity_extractor.py:28-77`

```python
# LLMManager 的批量处理
async def batch_generate(self, prompts, progress_bar=None):
    # 使用 Semaphore 控制并发
    tasks = [generate_with_progress(prompt, i) for i, prompt in enumerate(prompts)]
    await asyncio.gather(*tasks, return_exceptions=True)

# EntityExtractor 的并发抽取
async def extract_from_chunks(self, chunks, progress_bar=None):
    if hasattr(self.llm, 'enable_concurrent') and self.llm.enable_concurrent:
        # 并发模式：一次处理所有 chunks
        chunk_entities_list = await self.llm.batch_extract_entities(texts, progress_bar=progress_bar)
    else:
        # 串行模式：逐个处理
        for chunk in chunks:
            chunk_entities = await self.extract_entities(text)
```

**新增理解**:
- 并发控制在两个层级：LLM 调用级 (Semaphore) 和 任务级 (asyncio.gather)
- `progress_bar` 参数的设计：每个请求完成时更新进度条，而非全部完成后一次性更新
- `return_exceptions=True`：单个任务失败不影响其他任务

---

## 二、从复现失败中学到的教训

### 2.1 实体提取失败的根本原因

**现象**: 图构建成功但 0 个实体、0 个关系

**排查过程**:
1. 检查 LLM 返回值 → 发现包含 `<think>` 标签
2. 检查解析逻辑 → `_parse_entity_extraction_response` 无法处理 `<think>` 标签
3. 查找 Qwen3 文档 → 发现默认启用 thinking 模式

**解决方案**: 在 prompt 开头添加 `/no_think`

**深层教训**:
- **LLM 的输出格式是不可控的**：不同模型、不同版本可能有不同的输出格式
- **解析逻辑需要健壮性**：应该能处理意外格式（如 `<think>` 标签）
- **需要保存 LLM 原始输出**：便于调试

### 2.2 Embedding 服务不稳定

**现象**: Ollama 的 nomic-embed-text 服务频繁超时

**排查过程**:
1. 检查 Ollama 服务状态 → 服务正常
2. 检查网络连接 → 本地调用，网络正常
3. 检查模型加载 → 模型已加载
4. 测试单独调用 → 偶尔成功，偶尔超时

**解决方案**:
1. 使用本地 HuggingFace 模型替代 Ollama
2. 添加超时控制（30秒）
3. 超时后跳过 embedding 预计算

**深层教训**:
- **外部服务不可靠**：即使是本地服务也可能不稳定
- **需要超时和降级机制**：不能让一个组件的失败阻断整个流程
- **本地模型 vs API 模型的权衡**：
  - 本地模型：稳定、无网络依赖、但占用 GPU
  - API 模型：不占 GPU、但依赖网络和服务稳定性

### 2.3 参数解析的边界情况

**现象**: `-query 0` 被解析为 True

**原因**: Python argparse 的行为：
```python
parser.add_argument("-query", type=str, default=None)
# "-query 0" → args.query = "0" (字符串)
# if args.query: → True (非空字符串为真)
```

**解决方案**:
```python
# 修改前
if args.query:
# 修改后
if args.query and args.query != "0":
```

**深层教训**:
- **命令行参数需要仔细验证**：字符串 "0" 在 Python 中是真值
- **类型转换要显式**：`type=str` 意味着 "0" 是字符串而非数字
- **需要边界测试**：测试 "0", "1", "", None 等情况

---

## 三、技术原理的深层理解

### 3.1 CogniRank 的物理意义

**之前理解**: 公式 R = S_sem * exp(-(delta_tau/median)^k)

**本轮新增理解**:

```
k 参数的物理意义:
- k 控制衰减曲线的"陡峭程度"
- k=0.1: 衰减极平缓，几乎不区分新旧
- k=1.0: 标准指数衰减
- k=2.0: 高斯衰减，对时间非常敏感

实际效果 (假设 median=30天):
| delta_tau | k=0.1 | k=0.5 | k=1.0 | k=2.0 |
|-----------|-------|-------|-------|-------|
| 1天       | 0.97  | 0.82  | 0.97  | 0.99  |
| 30天      | 0.37  | 0.37  | 0.37  | 0.37  |
| 90天      | 0.28  | 0.13  | 0.05  | 0.002 |
| 365天     | 0.22  | 0.04  | 0.0001| ~0    |

设计意图:
- k=0.1 意味着即使 1 年前的信息，权重仍有 0.22
- 这是为了长期记忆场景：远期信息不应该被完全忽略
- 如果是实时推荐场景，应该用更大的 k (如 2.0)
```

### 3.2 实体去重的边界情况

**之前理解**: Jaccard 相似度，阈值 0.85

**本轮新增理解**:

```python
# Jaccard 相似度计算
def _calculate_similarity(self, text1, text2):
    set1 = set(text1.split())
    set2 = set(text2.split())
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0

# 问题案例:
"Coffee Shop" vs "The Coffee Shop"
set1 = {"Coffee", "Shop"} → 2 words
set2 = {"The", "Coffee", "Shop"} → 3 words
intersection = {"Coffee", "Shop"} → 2 words
union = {"The", "Coffee", "Shop"} → 3 words
Jaccard = 2/3 = 0.67 < 0.85 → 不会被合并!

"Coffee" vs "Coffee Shop"
set1 = {"Coffee"} → 1 word
set2 = {"Coffee", "Shop"} → 2 words
intersection = {"Coffee"} → 1 word
union = {"Coffee", "Shop"} → 2 words
Jaccard = 1/2 = 0.5 < 0.85 → 不会被合并!
```

**改进方向**:
1. 使用编辑距离 (Levenshtein Distance) 作为补充
2. 增加子串匹配：如果一个是另一个的子串，也算相似
3. 使用 Embedding 相似度作为最终判断

### 3.3 图的增量更新策略

**之前理解**: `add_entities_and_relationships_incrementally` 支持增量添加

**本轮新增理解**:

```python
# 增量更新的三个关键操作:

# 1. 实体合并 (不是覆盖)
def _merge_entity_attributes(self, existing, new):
    merged = existing.copy()
    for key, value in new.items():
        if value and (key not in merged or not merged[key]):
            merged[key] = value  # 只填充空属性
    return merged

# 2. 关系合并 (累加 chunk_ids 和 session_ids)
def _merge_relationship_data(self, existing, new):
    # chunk_ids: 合并去重
    existing_chunks = existing.get('chunk_ids', [])
    existing_chunks.extend(new.get('chunk_ids', []))
    existing['chunk_ids'] = list(set(existing_chunks))
    
    # strength: 取最大值 (关系随时间强化)
    existing['strength'] = max(existing.get('strength', 1), new.get('strength', 1))

# 3. 孤立节点清理
def _remove_isolated_nodes(self):
    isolated = [n for n in self.graph.nodes() if self.graph.degree(n) == 0]
    self.graph.remove_nodes_from(isolated)
```

**设计意图**:
- 实体合并：保留所有已知信息，新信息只补充空缺
- 关系合并：记录关系出现的所有 session 和 chunk，强度取最大值
- 孤立节点清理：删除没有连接的实体，保持图的紧凑性

---

## 四、工程实践的新增理解

### 4.1 vLLM 部署的最佳实践

**从复现中学到**:

```bash
# 推荐的 vLLM 启动命令
CUDA_VISIBLE_DEVICES=0,1 python -m vllm.entrypoints.openai.api_server \
  --model /home/chenyizhou/models/Qwen3-8B-AWQ \
  --host 0.0.0.0 \
  --port 8910 \
  --tensor-parallel-size 2 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.9

# 关键参数:
# --tensor-parallel-size: 使用多少张 GPU 做张量并行
# --max-model-len: 最大序列长度，影响显存占用
# --gpu-memory-utilization: GPU 显存使用率，0.9 表示使用 90%
```

**新增理解**:
- 张量并行 (Tensor Parallelism)：将模型的权重矩阵分摊到多张 GPU
- `--gpu-memory-utilization 0.9`：预留 10% 显存用于 CUDA 运行时
- `--max-model-len`：影响 KV Cache 大小，越大占用显存越多

### 4.2 HuggingFace 模型的本地加载

```python
# EmbeddingManager 的本地加载
from sentence_transformers import SentenceTransformer

# 方式1: 使用 cache_dir
self.client = SentenceTransformer(
    self.model_name, 
    cache_folder=self.cache_dir, 
    device=str(self.device)
)

# 方式2: 直接指定本地路径
self.client = SentenceTransformer(
    "/home/chenyizhou/models/bge-large-en-v1.5",
    device="cuda:2"
)
```

**新增理解**:
- `cache_folder` 会创建 `model_name` 子目录
- 直接指定路径更可控，避免目录结构问题
- `device` 参数指定使用哪张 GPU

### 4.3 异步编程的实际陷阱

**从代码中发现的模式**:

```python
# 模式1: Semaphore 控制并发
self.semaphore = asyncio.Semaphore(max_concurrent)

async def generate(self, prompt):
    async with self.semaphore:  # 自动获取和释放
        return await self._generate_internal(prompt)

# 模式2: 进度条更新
async def generate_with_progress(prompt, index):
    try:
        result = await self.generate(prompt)
        results_list[index] = result
    finally:
        if progress_bar:
            progress_bar.update(1)  # 无论成功失败都更新

# 模式3: 异常不传播
tasks = [generate_with_progress(prompt, i) for i, prompt in enumerate(prompts)]
await asyncio.gather(*tasks, return_exceptions=True)  # 单个失败不影响其他
```

**新增理解**:
- `async with self.semaphore`：比手动 acquire/release 更安全
- `results_list[index] = result`：通过索引写入，避免结果顺序错乱
- `return_exceptions=True`：将异常作为返回值而非抛出

---

## 五、评估体系的深入理解

### 5.1 多层评估的意义

**之前理解**: 有 Exact Match 和 LLM Eval 两种方式

**本轮新增理解**:

```
评估的三层结构:

Layer 1: 答案准确性 (Answer Accuracy)
  - Exact Match: 字符串匹配 (严格但漏判多)
  - LLM Eval: 语义匹配 (灵活但可能误判)
  - 两种方式交叉验证，分歧案例最有分析价值

Layer 2: 检索质量 (Retrieval Quality)
  - Session Matching: 检索到的 session 是否正确
  - 独立于生成模块，帮助定位问题在检索还是生成

Layer 3: 效率指标 (Efficiency Metrics)
  - 时间: 每个阶段的耗时
  - 成本: Token 消耗和 API 费用
  - 资源: GPU 使用率、内存占用
```

### 5.2 评估指标的局限性

| 指标 | 局限性 | 改进方向 |
|------|--------|---------|
| Exact Match | 对表述差异过于严格 | 增加同义词匹配、数字归一化 |
| LLM Eval | 成本高、速度慢、可能有 bias | 用更小的模型、多次评估取平均 |
| Session Matching | 只检查 session 级别，不检查具体内容 | 增加 chunk 级别的匹配 |

---

## 六、部署架构的新增理解

### 6.1 单机部署 vs 分布式部署

**当前架构 (单机)**:
```
User → main.py → GraphRAG → QueryProcessor → LLM API
                           ↓
                    DynamicMemory (NetworkX)
                           ↓
                    Pickle 文件存储
```

**生产架构 (分布式)**:
```
                    Load Balancer
                         |
         +---------------+---------------+
         |               |               |
    Query Service   Graph Service   Storage Service
    (FastAPI)       (FastAPI)       (Neo4j + Milvus)
```

**新增理解**:
- NetworkX 是内存图，单用户几千节点没问题，多用户需要图数据库
- Pickle 文件无法支持并发读写，需要数据库
- 向量检索需要专门的向量数据库 (Milvus) 支持 ANN

### 6.2 8卡3090 的最优分配

```
推荐配置:

Card 0-1: vLLM (Qwen3-8B-AWQ, tensor_parallel=2)
  - 处理所有 LLM 调用 (抽取、摘要、查询、评估)
  - 支持 16 并发

Card 2: BGE-Large Embedding
  - 处理所有 Embedding 计算
  - 支持 64 批处理

Card 3: 预留 (评估用独立LLM 或 更大模型)

Card 4-7: 预留 (并行实验 或 更大模型)
  - 可以同时运行 2-3 个不同配置的实验
  - 或运行 Qwen3-14B-AWQ (需要 2-3 张卡)
```

---

## 七、待深入研究的问题

### 7.1 多跳检索的实现

**当前实现**: 只做 1-hop 检索 (找到实体 -> 找到相邻边)

**待研究**:
- 如何实现 2-hop 或 3-hop 检索?
- 如何控制检索深度避免爆炸?
- 如何在多跳过程中保持语义相关性?

### 7.2 指代消解的集成

**当前实现**: 不支持指代消解 ("他"、"那个咖啡店" 无法链接)

**待研究**:
- 如何集成 spaCy 或 Stanza 的共指消解?
- 如何在 Prompt 中引导 LLM 做指代消解?
- 性能和准确性的权衡?

### 7.3 增量 Embedding 更新

**当前实现**: 每次新增实体都要重新计算所有 Embedding

**待研究**:
- 如何实现增量计算?
- 使用 annoy 或 faiss 做增量索引?
- 如何保证增量更新后的一致性?

---

## 八、学习路径总结

```
第一轮 (复现阶段):
  目标: 跑通流程
  收获: 环境配置、数据处理、基本运行
  失败: 实体提取失败、Embedding 不稳定

第二轮 (面试准备):
  目标: 理解原理
  收获: CogniRank 公式、多阶段检索、评估体系
  产出: INTERVIEW_GUIDE.md, INTERVIEW_FIVE_CATEGORIES.md

第三轮 (本轮):
  目标: 深入细节 + 实际落地
  收获: 
    - Qwen3 的 /no_think 指令
    - Embedding 降级策略
    - 并发控制的实现细节
    - 8卡3090 的最优分配
  产出: REPRODUCTION_PLAN.md, INCREMENTAL_LEARNING_NOTES.md (本文件)
```

---

## 九、关键代码位置速查 (新增)

| 功能 | 文件:行号 | 本轮新增理解 |
|------|----------|-------------|
| Qwen3 /no_think | `prompt/entity_prompt.py:1` | 需要在 prompt 开头添加 |
| Embedding 降级 | `coregraph/dynamic_memory.py:605-628` | 失败时跳过预计算 |
| 字符串匹配 fallback | `query/query_processor.py:426-454` | Embedding 不可用时的备选 |
| Semaphore 并发控制 | `base/llm.py:31,43-46` | async with 自动管理 |
| 进度条更新 | `base/llm.py:182-193` | 每个请求完成时更新 |
| 实体合并 | `coregraph/graph_builder.py:162-174` | 只填充空属性 |
| 关系合并 | `coregraph/graph_builder.py:176-228` | chunk_ids 去重、strength 取 max |
| 孤立节点清理 | `coregraph/graph_builder.py:90-95` | 删除 degree=0 的节点 |
| 参数解析修复 | `main.py:292` | `args.query != "0"` |

---

> 文档生成时间: 基于第三轮深度学习
> 核心价值: 将前两轮的理论理解与复现实践中的工程教训结合，形成可落地的知识
