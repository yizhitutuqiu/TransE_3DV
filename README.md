# ZSTP Final (3D Vision KG + TransE)

本目录提供从知识图谱 TransE 模型进行推理/评测的本地 HTTP API。API 默认加载 `checkpoints/transe_v1/best.pt`，并使用 `data/preprocessed/final/` 下的 `entity2id.txt / relation2id.txt / train2id.txt / test2id.txt`。

## Python API（给脚本直接调用）

默认配置文件是 `zstp_final/api.yaml`，默认加载：
- `checkpoints/transe_v1/best.pt`
- `data/preprocessed/final/`

### 快速开始

```python
from zstp_final.api import load_api

api = load_api("zstp_final/api.yaml")
print(api.model_info())
```

### 可用方法

```python
api.relations()
api.search_entities(q="nerf", entity_type="Method", limit=20)

api.score(h="Paper:2403.04765", r="paper_proposes_method", t="Method:NeRF")
api.predict_tail(h="Paper:2403.04765", r="paper_proposes_method", k=10, candidate_type="Method", filtered=True)
api.predict_head(t="Method:NeRF", r="paper_proposes_method", k=10, candidate_type="Paper", filtered=True)
api.neighbors(entity="Method:NeRF", k=10)

api.evaluate(split="test", batch_size=256)
```
