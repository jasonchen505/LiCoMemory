# LiCoMemory 完整复现计划

> 基于 8卡 RTX 3090 (24GB显存/卡) 的算力评估与详细复现方案
> 结合前两轮学习理解和之前的复现经验，制定可落地的全流程复现计划

---

## 一、算力资源评估

### 1.1 硬件资源

| 资源 | 规格 | 可用情况 |
|------|------|---------|
| GPU | 8x RTX 3090 (24GB) | ✅ 充足 |
| CPU | AMD EPYC 7402 24-Core | ✅ 充足 |
| 显存总计 | 8 * 24GB = 192GB | ✅ 可并行多模型 |

### 1.2 算力需求分析

#### LLM 推理需求

```
项目需要的 LLM 调用:
1. 实体关系抽取: 每个 chunk 1次调用 (约1000 tokens prompt + 500 tokens completion)
2. 会话摘要生成: 每个 session 1次调用 (约1500 tokens prompt + 800 tokens completion)
3. 查询答案生成: 每个 query 1次调用 (约2000 tokens prompt + 200 tokens completion)
4. LLM评估: 每个答案 1次调用 (约500 tokens)

LOCOMO 数据集规模:
- 约 10-20 个对话组
- 每组 5-10 个 session
- 每组 10-30 个问题
- 总计约 200 个 session, 500 个问题

总 LLM 调用估算:
- 图构建: 200 session * (1 summary + 5 chunks * 1 extraction) = 1200 次调用
- 查询: 500 次调用
- 评估: 500 次调用
- 总计: ~2200 次 LLM 调用
```

#### 模型选择与显存需求

| 模型 | 参数量 | 显存需求 (FP16) | 显存需求 (INT4/AWQ) | 推荐配置 |
|------|--------|----------------|-------------------|---------|
| Qwen3-8B | 8B | ~16GB | ~5GB | 1卡3090 |
| Qwen3-14B | 14B | ~28GB | ~8GB | 2卡3090 |
| Qwen2.5-72B | 72B | ~144GB | ~40GB | 4卡3090 |
| Llama3-8B | 8B | ~16GB | ~5GB | 1卡3090 |
| BGE-Large | 335M | ~1.3GB | N/A | 1卡可共享 |

**结论**: 8卡3090 完全足够运行 LiCoMemory 全流程

### 1.3 推荐模型配置方案

#### 方案A: 高质量配置 (追求最佳效果)

```yaml
# LLM: 使用 2张3090 运行 Qwen3-14B-AWQ
llm:
  model: Qwen/Qwen3-14B-AWQ
  base_url: "http://localhost:8910/v1"
  max_concurrent: 8

# Embedding: 使用 1张3090 运行 BGE-Large
embedding:
  api_type: "hf"
  model: "BAAI/bge-large-en-v1.5"
  device: "cuda:2"

# 评估LLM: 复用主LLM或使用更小模型
evaluation:
  eval_model: "Qwen/Qwen3-8B-AWQ"  # 共用主LLM
```

#### 方案B: 高效率配置 (追求速度)

```yaml
# LLM: 使用 4张3090 并行运行 4个 Qwen3-8B 实例
llm:
  model: Qwen/Qwen3-8B-AWQ
  base_url: "http://localhost:8910/v1"
  max_concurrent: 32  # 4实例 * 8并发

# Embedding: 使用 2张3090 运行 BGE-Large
embedding:
  api_type: "hf"
  model: "BAAI/bge-large-en-v1.5"
  device: "cuda:4"  # 或使用 DataParallel
```

#### 方案C: 均衡配置 (推荐)

```yaml
# LLM: 使用 2张3090 运行 Qwen3-8B-AWQ (vLLM)
llm:
  model: Qwen/Qwen3-8B-AWQ
  base_url: "http://localhost:8910/v1"
  max_concurrent: 16

# Embedding: 使用 1张3090 运行 BGE-Large
embedding:
  api_type: "hf"
  model: "BAAI/bge-large-en-v1.5"
  device: "cuda:2"

# 预留 5张3090 用于:
# - 并行运行多个实验
# - 运行更大的模型 (如需要)
# - 运行评估用的独立LLM
```

---

## 二、复现阶段规划

### Phase 0: 环境准备 (Day 1)

#### 0.1 虚拟环境配置

```bash
cd /home/chenyizhou/LiCoMemory

# 使用已有的 .venv 或重新创建
source .venv/bin/activate

# 确认依赖
pip list | grep -E "torch|networkx|sentence-transformers|openai"
```

#### 0.2 模型下载

```bash
# 下载 Qwen3-8B-AWQ (用于 LLM)
# 方式1: 使用 ModelScope (国内镜像)
pip install modelscope
python -c "from modelscope import snapshot_download; snapshot_download('Qwen/Qwen3-8B-AWQ', cache_dir='/home/chenyizhou/models')"

# 方式2: 使用 HuggingFace
huggingface-cli download Qwen/Qwen3-8B-AWQ --local-dir /home/chenyizhou/models/Qwen3-8B-AWQ

# 下载 BGE-Large (用于 Embedding)
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-large-en-v1.5', cache_folder='/home/chenyizhou/models')"
```

#### 0.3 推理服务部署

```bash
# 启动 vLLM 服务 (LLM)
# 使用 2张GPU 运行 Qwen3-8B-AWQ
CUDA_VISIBLE_DEVICES=0,1 python -m vllm.entrypoints.openai.api_server \
  --model /home/chenyizhou/models/Qwen3-8B-AWQ \
  --host 0.0.0.0 \
  --port 8910 \
  --tensor-parallel-size 2 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.9

# 在另一个终端测试
curl http://localhost:8910/v1/models
```

### Phase 1: 数据准备 (Day 1-2)

#### 1.1 下载 LOCOMO 数据集

```bash
# 方式1: 从 HuggingFace
pip install datasets
python -c "from datasets import load_dataset; ds = load_dataset('snap-research/locomo', cache_dir='/home/chenyizhou/datasets')"

# 方式2: 从 GitHub
git clone https://github.com/snap-research/locomo.git /home/chenyizhou/datasets/locomo_raw
```

#### 1.2 数据预处理

```bash
# 使用项目自带的脚本处理
cd /home/chenyizhou/LiCoMemory

# 处理 LOCOMO 数据集
python dataset/locomo.py \
  --input /home/chenyizhou/datasets/locomo_raw/locomo10.json \
  --outdir dataset/locomo

# 验证数据格式
ls -la dataset/locomo/
head -1 dataset/locomo/group_1/Corpus.json
head -1 dataset/locomo/group_1/Question.json
```

#### 1.3 创建配置文件

```yaml
# config/Memory_3090.yaml
index_name: "licomemory_3090"
data_type: "LOCOMO"
data_root: "/home/chenyizhou/LiCoMemory/dataset"
dataset_name: "locomo"
working_dir: "/home/chenyizhou/LiCoMemory/results"

# Embedding 配置 (使用本地 HuggingFace 模型)
embedding:
  api_type: "hf"
  model: "BAAI/bge-large-en-v1.5"
  cache_dir: "/home/chenyizhou/models"
  dimensions: 1024
  max_token_size: 512
  embed_batch_size: 64
  embedding_func_max_async: 4

# LLM 配置 (使用 vLLM 服务)
llm:
  api_type: "openai"
  model: "Qwen/Qwen3-8B-AWQ"
  base_url: "http://localhost:8910/v1"
  api_key: "EMPTY"
  max_token: 8192
  temperature: 0.0
  enable_concurrent: true
  max_concurrent: 16
  timeout: 120

# 查询LLM (可以与主LLM相同)
query_llm:
  api_type: "openai"
  model: "Qwen/Qwen3-8B-AWQ"
  base_url: "http://localhost:8910/v1"
  api_key: "EMPTY"
  max_token: 4096
  temperature: 0.1
  timeout: 120

# Chunk 配置
chunk:
  chunk_token_size: 1200
  chunk_overlap_token_size: 100
  dialogue_input: true

# 对话配置
dialog:
  enable_turn_pairing: true
  include_session_metadata: true
  validate_dialog_format: true

# 图配置
graph:
  graph_type: "dynamic_memory"
  force: true
  add: false
  entity_merge_threshold: 0.85
  relationship_merge_threshold: 1

# 检索配置
retriever:
  top_k: 5
  top_k_triples: 20
  top_chunks: 15
  enable_summary: true
  top_summary: 2
  enable_visual: false
  enable_full: true
  enable_sessiontime: true
  enable_CogniRank: true
  summary_weight: 0.2
  rerank_k: 0.1

# 评估配置
evaluation:
  enable_llm_eval: true
  eval_model: "Qwen/Qwen3-8B-AWQ"
  eval_temperature: 0.0
  eval_max_tokens: 10
```

### Phase 2: 小规模验证 (Day 2-3)

#### 2.1 单 Session 测试

```bash
# 先用单个 group 测试图构建
python main.py \
  -opt config/Memory_3090.yaml \
  -dataset_name group_1 \
  -root test_single_group \
  -query 0

# 检查构建结果
ls -la results/test_single_group/
cat results/test_single_group/session_summaries.json | python -m json.tool | head -50
```

#### 2.2 图构建验证

```bash
# 检查 pkl 文件
python -c "
import pickle
with open('results/test_single_group/licomemory_3090.pkl', 'rb') as f:
    data = pickle.load(f)
    graph = data['graph']
    print(f'Nodes: {len(graph.nodes)}')
    print(f'Edges: {len(graph.edges)}')
    print(f'Chunk storage: {len(data.get(\"chunk_storage\", {}))}')
"
```

#### 2.3 单问题查询测试

```bash
# 运行查询
python main.py \
  -opt config/Memory_3090.yaml \
  -dataset_name group_1 \
  -root test_query \
  -query 1

# 检查结果
cat results/test_query/results/results.json | python -m json.tool | head -100
```

### Phase 3: 完整实验 (Day 3-5)

#### 3.1 全量图构建

```bash
# 遍历所有 group 构建图
for group in group_1 group_2 group_3 group_4 group_5; do
  echo "Processing $group..."
  python main.py \
    -opt config/Memory_3090.yaml \
    -dataset_name $group \
    -root full_experiment/$group \
    -query 0
done
```

#### 3.2 全量查询与评估

```bash
# 对所有 group 运行查询
for group in group_1 group_2 group_3 group_4 group_5; do
  echo "Querying $group..."
  python main.py \
    -opt config/Memory_3090.yaml \
    -dataset_name $group \
    -root full_experiment/$group \
    -query 1
done
```

#### 3.3 收集结果

```bash
# 汇总所有结果
python -c "
import json
import os

results_dir = 'full_experiment'
all_metrics = {}

for group in os.listdir(results_dir):
    metrics_path = os.path.join(results_dir, group, 'results', 'metrics.json')
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)
            all_metrics[group] = metrics
            print(f'{group}: accuracy={metrics.get(\"accuracy\", 0):.3f}')

# 计算平均
avg_acc = sum(m.get('accuracy', 0) for m in all_metrics.values()) / len(all_metrics)
print(f'Average accuracy: {avg_acc:.3f}')
"
```

### Phase 4: 消融实验 (Day 5-7)

#### 4.1 CogniRank vs SimpleRank

```bash
# 实验1: 启用 CogniRank
# 修改 config: enable_CogniRank: true
python main.py -opt config/Memory_3090_CogniRank.yaml -dataset_name group_1 -root ablation/cognirank_on -query 1

# 实验2: 禁用 CogniRank
# 修改 config: enable_CogniRank: false
python main.py -opt config/Memory_3090_SimpleRank.yaml -dataset_name group_1 -root ablation/cognirank_off -query 1
```

#### 4.2 Summary 消融

```bash
# 实验3: 启用 Summary
# 修改 config: enable_summary: true
python main.py -opt config/Memory_3090_Summary.yaml -dataset_name group_1 -root ablation/summary_on -query 1

# 实验4: 禁用 Summary
# 修改 config: enable_summary: false
python main.py -opt config/Memory_3090_NoSummary.yaml -dataset_name group_1 -root ablation/summary_off -query 1
```

#### 4.3 不同 LLM 对比

```bash
# 实验5: 使用 Qwen3-8B
python main.py -opt config/Memory_3090_Qwen8B.yaml -dataset_name group_1 -root ablation/qwen8b -query 1

# 实验6: 使用 Qwen3-14B (需要更多GPU)
# 需要修改 vLLM 配置使用 2-4 张卡
python main.py -opt config/Memory_3090_Qwen14B.yaml -dataset_name group_1 -root ablation/qwen14b -query 1
```

### Phase 5: 结果分析与优化 (Day 7-10)

#### 5.1 结果可视化

```python
# analyze_results.py
import json
import matplotlib.pyplot as plt
import pandas as pd

# 加载所有实验结果
experiments = {
    'CogniRank': 'ablation/cognirank_on/results/metrics.json',
    'SimpleRank': 'ablation/cognirank_off/results/metrics.json',
    'With Summary': 'ablation/summary_on/results/metrics.json',
    'No Summary': 'ablation/summary_off/results/metrics.json',
}

results = []
for name, path in experiments.items():
    with open(path) as f:
        metrics = json.load(f)
        results.append({
            'experiment': name,
            'accuracy': metrics.get('accuracy', 0),
            'matching': metrics.get('average_matching_score', 0),
        })

df = pd.DataFrame(results)

# 绘制对比图
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
df.plot(x='experiment', y='accuracy', kind='bar', ax=axes[0], title='Accuracy')
df.plot(x='experiment', y='matching', kind='bar', ax=axes[1], title='Session Matching')
plt.tight_layout()
plt.savefig('ablation/comparison.png')
plt.show()
```

#### 5.2 按问题类型分析

```python
# 分析不同问题类型的准确率
def analyze_by_type(results_path):
    with open(results_path) as f:
        results = json.load(f)
    
    type_stats = {}
    for item in results:
        q_type = item.get('question_type', 'unknown')
        correct = item.get('correct', False)
        
        if q_type not in type_stats:
            type_stats[q_type] = {'total': 0, 'correct': 0}
        
        type_stats[q_type]['total'] += 1
        if correct:
            type_stats[q_type]['correct'] += 1
    
    for q_type, stats in type_stats.items():
        acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        print(f'{q_type}: {stats["correct"]}/{stats["total"]} ({acc:.2%})')
```

---

## 三、关键技术点与注意事项

### 3.1 前次复现的关键发现

根据 `reconstruct_learn_note/08_final_results.md` 的记录:

| 问题 | 根因 | 解决方案 |
|------|------|---------|
| 实体提取失败 (0个实体) | Qwen3 默认启用 thinking 模式 | 在 prompt 开头添加 `/no_think` |
| Embedding 服务超时 | Ollama 服务不稳定 | 使用本地 HuggingFace 模型 |
| 参数解析错误 | `-query 0` 被错误解析 | 修改判断逻辑 |

### 3.2 Qwen3 模型特殊处理

```python
# prompt/entity_prompt.py 已修改
# 在每个 prompt 开头添加 /no_think
DIALOGUE_EXTRACTION_PROMPT = """/no_think
You are a helpful assistant trying to extract entities and relations...
"""
```

### 3.3 Embedding 服务稳定性

```python
# 方案1: 使用本地 HuggingFace (推荐)
# 优点: 稳定、无网络依赖、GPU加速
# 缺点: 占用GPU显存
embedding:
  api_type: "hf"
  model: "BAAI/bge-large-en-v1.5"

# 方案2: 使用 OpenAI API
# 优点: 不占用本地GPU
# 缺点: 需要网络、有成本
embedding:
  api_type: "openai"
  model: "text-embedding-3-small"
  api_key: "sk-xxx"
```

### 3.4 并行化策略

```
8卡3090 并行策略:

Card 0-1: vLLM 服务 (Qwen3-8B-AWQ, tensor_parallel=2)
Card 2: BGE-Large Embedding (本地)
Card 3: 预留 (评估用LLM 或 更大模型)
Card 4-7: 预留 (并行实验 或 更大模型)

并行实验:
- 可以同时运行 2-3 个不同配置的实验
- 每个实验使用不同的 GPU
```

### 3.5 成本与时间估算

| 阶段 | LLM调用次数 | 预估时间 | Token消耗 |
|------|------------|---------|----------|
| 图构建 (单group) | ~150次 | ~10分钟 | ~150K |
| 查询 (单group) | ~30次 | ~3分钟 | ~60K |
| 评估 (单group) | ~30次 | ~2分钟 | ~15K |
| **全流程 (5 groups)** | **~1050次** | **~75分钟** | **~1.1M** |

---

## 四、验证清单

### 4.1 图构建验证

- [ ] 实体数量 > 0
- [ ] 关系数量 > 0
- [ ] Session summaries 生成成功
- [ ] pkl 文件保存成功
- [ ] Embedding 预计算成功 (或超时跳过)

### 4.2 查询验证

- [ ] 实体检索返回结果
- [ ] 三元组召回 > 0
- [ ] 答案生成非空
- [ ] 时间统计完整

### 4.3 评估验证

- [ ] LLM 评估运行成功
- [ ] 准确率计算正确
- [ ] Session matching 计算正确
- [ ] 分类型准确率可获取

---

## 五、风险与备选方案

### 5.1 潜在风险

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| vLLM 启动失败 | 无法运行LLM | 使用 Ollama 或 直接调用 API |
| HuggingFace 下载慢 | 环境配置耗时 | 使用 ModelScope 镜像 |
| GPU 显存不足 | 无法运行大模型 | 使用量化模型 (AWQ/GPTQ) |
| 实体提取质量差 | 图结构不完整 | 优化 prompt 或使用更强模型 |

### 5.2 备选模型方案

```
如果 Qwen3-8B 效果不好:
1. 尝试 Qwen3-14B-AWQ (需要更多GPU)
2. 尝试 Llama3-8B-Instruct
3. 尝试 Mistral-7B-Instruct
4. 使用 API: GPT-4o-mini (成本低、效果好)

如果 BGE-Large 效果不好:
1. 尝试 GTE-Large
2. 尝试 E5-Large
3. 使用 API: text-embedding-3-small
```

---

## 六、时间线总览

```
Day 1:   环境准备 + 模型下载 + 服务部署
Day 2:   数据准备 + 配置文件创建 + 小规模验证
Day 3:   单group全流程测试 + 问题修复
Day 4-5: 全量实验 (5 groups)
Day 6-7: 消融实验 (CogniRank, Summary, LLM对比)
Day 8-9: 结果分析 + 可视化 + 优化
Day 10:  总结报告 + 代码整理
```

---

## 七、快速启动命令

```bash
# 1. 激活环境
cd /home/chenyizhou/LiCoMemory
source .venv/bin/activate

# 2. 启动 vLLM (新终端)
CUDA_VISIBLE_DEVICES=0,1 python -m vllm.entrypoints.openai.api_server \
  --model /home/chenyizhou/models/Qwen3-8B-AWQ \
  --host 0.0.0.0 --port 8910 --tensor-parallel-size 2

# 3. 测试单group
python main.py -opt config/Memory_3090.yaml -dataset_name group_1 -root test_run -query 1

# 4. 查看结果
cat results/test_run/results/metrics.json
```

---

> 文档生成时间: 基于 8卡3090 算力评估
> 建议: 先用 Phase 2 小规模验证，确认流程无误后再进行全量实验
