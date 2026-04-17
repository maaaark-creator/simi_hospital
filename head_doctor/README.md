# Head Doctor Module

本目录包含头部医生（总控）智能体相关代码，负责协调各专科评估、识别不确定点、触发 MDT 追问，并最终整合各模型审阅结果。

主要文件：

- `head_doctor_agent.py`：核心总控逻辑，负责：
  - 调用各专科 agent 执行初始评估
  - 收集不确定点并触发 `mdt_call` 追问
  - 整合多个模型的 head doctor 审阅结果，生成最终推荐
- `run_head_doctor_demo.py`：运行 `head_doctor` 总控流程的脚本。

使用说明：

1. 运行 `run_head_doctor_demo.py` 或者根目录的完整演示脚本查看总流程。
2. 该模块依赖各专科目录和 `mdt_call` 模块进行协同工作。