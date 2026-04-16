# Cardiology Module

本目录包含心内科专科智能体相关代码，用于处理心血管病例的结构化、知识检索和决策生成。

主要文件：

- `cardiology_decision_agent.py`：心内科决策智能体，生成心内科诊疗建议。
- `cardiovascular_kb.json`：心内科知识库数据文件。
- `kb_retriever.py`：心内科知识检索模块。
- `patient_mapper_agent.py`：患者信息映射模块，用于标准化心内科输入。
- `run_cardiology_pipeline_demo.py`：本地演示脚本，可单独运行心内科评估流程。

使用说明：

1. 运行 `run_cardiology_pipeline_demo.py` 测试本模块行为。
2. 该目录的功能在 `head_doctor` 总流程中被集成调用。