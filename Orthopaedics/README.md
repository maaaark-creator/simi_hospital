# Orthopaedics Agent

这是一个按当前仓库 `Cardiology` / `Anesthesia` 接口风格整理后的骨科子 agent，可直接被主干 `head_doctor -> mdt_call -> 前端可视化` 流程加载。

核心目标：

- 复用 `KB + mapper + retriever + decision + demo` 的分层方式。
- 保持与现有专科 agent 一致的模块命名和加载接口。
- 从骨科视角综合考虑主诉、创伤机制、影像、神经血管风险、感染、肿瘤/病理骨折、家族史、基础疾病和围手术期风险。

## 文件

- `orthopaedics_kb.json`
- `patient_mapper_agent.py`
- `kb_retriever.py`
- `orthopaedics_decision_agent.py`
- `run_orthopaedics_pipeline_demo.py`

## 主要输出

骨科决策输出包含：

- `primary_condition` / `primary_condition_id`
- `primary_plan` / `primary_plan_id`
- `backup_plans` / `backup_plan_ids`
- `triage_level`
- `recommended_workup`
- `recommended_management`
- `perioperative_considerations`
- `potential_complications`
- `risk_flags`
- `need_more_info`
- `reasoning_trace`

## 运行示例

```bash
python run_orthopaedics_pipeline_demo.py --mode local --decision-mode heuristic
```

如需 API 版：

```bash
python run_orthopaedics_pipeline_demo.py --mode api --decision-mode api
```

## 接入说明

- 当前目录已经改成和主仓库其它专科一致的命名方式。
- `head_doctor/head_doctor_agent.py` 已注册 `orthopaedics` 专科。
- `MDT_Call` 已加入骨科分诊关键词和 API prompt 允许项。
- 前端实时进度面板会自动显示 `orthopaedics/...` 的结构化、检索、决策和追问步骤。
