# Neurosurgery Module

本目录包含神经外科专科智能体相关代码，用于处理神经外科病例的结构化、检索和诊疗决策。

主要文件：

- `neurosurgery_decision_agent.py`：神经外科决策智能体，负责生成手术及术前评估建议。
- `neurosurgery_kb_retriever.py`：神经外科知识检索模块。
- `neurosurgery_kb.json`：神经外科知识库数据。
- `neurosurgery_patient_mapper_agent.py`：患者信息映射模块。
- `neurosurgery_shared.py`：神经外科模块共享工具。
- `check_neurosurgery_encoding.py`：编码检查脚本。
- `run_neurosurgery_pipeline_demo.py`：本地演示脚本。

使用说明：

1. 运行 `run_neurosurgery_pipeline_demo.py` 查看本模块单独评估结果。
2. 该目录在 `head_doctor` 的总流程中负责产生神经外科意见。