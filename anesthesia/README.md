# Anesthesia Module

本目录包含麻醉科专科智能体相关代码，用于处理患者信息结构化、知识检索和麻醉决策。

主要文件：

- `anesthesia_decision_agent.py`：麻醉科决策智能体，负责基于结构化病例和检索结果生成麻醉方案。
- `anesthesia_kb.json`：麻醉科知识库数据文件。
- `kb_retriever.py`：知识检索模块，用于从麻醉知识库中检索相关规则和信息。
- `patient_mapper_agent.py`：患者信息映射模块，用于将原始病例文本或结构化输入转为麻醉科可处理的数据。
- `run_anesthesia_pipeline_demo.py`：本地演示脚本，可单独运行麻醉科评估流程。

使用说明：

1. 运行 `run_anesthesia_pipeline_demo.py` 查看本模块诊断流程。
2. 该目录主要被 `head_doctor` 的总控流程调用。