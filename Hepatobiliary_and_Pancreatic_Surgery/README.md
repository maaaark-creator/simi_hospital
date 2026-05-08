# Hepatobiliary and Pancreatic Surgery Module

本目录包含肝胆胰外科专科智能体相关代码，用于处理肝胆胰病例的结构化、知识检索和决策建议。

主要文件：

- `hbp_decision_agent.py`：肝胆胰外科决策智能体。
- `hbp_kb_retriever.py`：肝胆胰知识检索模块。
- `hepatobiliary_decision_agent.py`：肝胆胰具体决策实现。
- `hepatobiliary_kb.json`：肝胆胰知识库数据文件。
- `kb_retriever.py`：辅助检索模块。
- `patient_mapper_agent.py`：患者映射模块。
- `run_hepatobiliary_pipeline_demo.py`：单独演示脚本。

使用说明：

1. 运行 `run_hepatobiliary_pipeline_demo.py` 验证本模块结果。
2. 该目录也会被 `head_doctor` 集成调用。