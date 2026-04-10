# MDT Call Module

本目录包含 MDT 分诊与追问模块，用于根据 head doctor 识别出来的待澄清不确定点，生成后续追问任务并收集各专科补充意见。

主要文件：

- `mdt_call_agent.py`：MDT 追问协调器，负责：
  - 初始分诊任务生成
  - 根据 `uncertainty_review` 生成追问任务
  - 调用各专科 clarify_case 处理追问
- `mdt_call_kb.json`：MDT 逻辑相关的知识库。
- `mdt_kb_retriever.py`：MDT 任务生成检索器。
- `mdt_patient_mapper_agent.py`：MDT 请求上下文构建模块。
- `run_mdt_call_pipeline_demo.py.py`：本地演示脚本，注意文件名重复，应谨慎使用。

使用说明：

1. 该模块通常由 `head_doctor` 调用，不建议单独作为最终决策模块使用。
2. `mdt_call` 负责将 head doctor 的疑问分发给对应专科进行补充。