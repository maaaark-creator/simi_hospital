# simi_hospital: Anesthesia Agent Demo

一个面向麻醉场景的轻量级 Agent Demo，用于演示如何将患者信息结构化、映射到麻醉知识库、检索候选方案，并生成初步麻醉决策建议。

该项目更适合作为教学、课程设计、原型验证或 Agent pipeline 示例，不应直接用于真实临床决策。

## 1. 项目目标

本项目尝试搭建一个最小可运行的麻醉决策支持流程：

1. 输入患者原始信息，可以是结构化字典，也可以是自然语言病例描述。
2. 将患者信息标准化为统一字段。
3. 将关键信息映射到麻醉知识库中的概念和规则。
4. 检索候选麻醉方案、相关药物和安全规则。
5. 输出一个可解释的初步麻醉建议。

整体上，这是一个典型的医疗 Agent Pipeline：

`Patient Input -> Mapper Agent -> KB Retriever -> Decision Agent -> Structured Recommendation`

## 2. 核心功能

### 2.1 患者信息标准化

由 [patient_mapper_agent.py](/d:/hemu_STU/BME/simi_hospital/patient_mapper_agent.py) 实现，支持：

- 将中文或英文别名字段统一映射到标准字段名。
- 将原始患者字典转换为统一的 `normalized_profile`。
- 支持年龄、性别、体重、手术名称、ASA 分级、合并症、禁食情况、气道评估、检验、生命体征等字段。

标准化后的关键字段包括：

- `age`
- `sex`
- `weight_kg`
- `height_cm`
- `procedure_name`
- `procedure_site`
- `asa_hint`
- `allergies`
- `comorbidities`
- `fasting_status`
- `airway_notes`
- `urgency`
- `labs`
- `vitals`

### 2.2 文本病例结构化

`PatientProfileMapperAgent` 支持调用外部大模型 API，将自然语言病例描述抽取为 JSON：

- `call_patient_structuring_api()`

例如，可将“67岁男性，拟行下肢骨折内固定，ASA III，已禁食8小时……”转换成统一结构化字段。

### 2.3 医学实体归一化

Mapper 还提供：

- `call_medical_entity_linking_api()`

该步骤用于把病例中的临床表达进一步整理为适合知识库检索的概念，例如：

- 标准化术式关键词
- 风险关键词
- ASA 解释
- 药物相关关键词

### 2.4 知识库检索

由 [kb_retriever.py](/d:/hemu_STU/BME/simi_hospital/kb_retriever.py) 实现，主要负责：

- 根据 ASA 提示映射到知识库中的 ASA 分类
- 根据术式、禁食情况、气道信息推断候选麻醉方案
- 根据合并症和风险信息触发安全规则
- 汇总相关药物
- 识别病例中的缺失信息

输出包括：

- `asa_match`
- `candidate_plans`
- `matched_safety_rules`
- `related_drugs`
- `retrieval_notes`
- `missing_information`

### 2.5 麻醉方案决策

由 [anesthesia_decision_agent.py](/d:/hemu_STU/BME/simi_hospital/anesthesia_decision_agent.py) 实现，支持两种模式：

- `decide()`：基于本地启发式规则做决策
- `decide_with_api()`：调用大模型生成最终 JSON 决策结果

决策结果包含：

- `patient_summary`
- `primary_plan`
- `backup_plans`
- `risk_flags`
- `need_more_info`
- `reasoning_trace`

## 3. 方法设计

### 3.1 整体思路

这个项目采用了“规则 + 知识库 + 大模型”的混合方法：

- 规则负责最基础、最可控的字段标准化与条件判断。
- 知识库负责承载麻醉方案、药物、安全规则等领域知识。
- 大模型负责处理自然语言输入、实体归一化和更灵活的方案排序。

这种设计的优点是：

- 比纯大模型输出更可控
- 比纯规则系统更灵活
- 方便后续扩展更多知识条目和临床规则

### 3.2 Pipeline 分层

#### 第一层：输入映射层

负责把用户输入转换为统一的患者画像：

- 解决中英文字段名不一致的问题
- 对列表类字段做清洗
- 保留原始输入 `raw_input` 便于追溯

#### 第二层：知识检索层

负责把患者画像与知识库进行连接：

- 匹配 ASA 分类
- 推断候选麻醉方案
- 触发安全规则
- 收集候选药物

#### 第三层：决策生成层

负责从候选方案中选出首选方案，并给出解释：

- 提炼患者摘要
- 标注风险点
- 给出备选方案
- 指出仍需补充的信息

### 3.3 当前决策逻辑示例

项目中的本地启发式决策大致遵循以下思路：

- 若存在误吸风险、未充分禁食或疑似困难气道，则优先考虑气管插管全麻。
- 若手术部位偏向下肢、下腹、会阴等区域，则优先考虑椎管内麻醉。
- 若为短小、表浅操作，则可考虑 MAC。
- 若关键信息缺失，则在结果中显式提示补充信息。

这部分目前是原型逻辑，适合演示 Agent 推理流程，不代表完整临床规范。

## 4. 项目结构

```text
simi_hospital/Anesthesia
├─ README.md
├─ LICENSE
├─ anesthesia_kb.json
├─ patient_mapper_agent.py
├─ kb_retriever.py
├─ anesthesia_decision_agent.py
└─ run_anesthesia_pipeline_demo.py
```

各文件职责如下：

- [anesthesia_kb.json](/d:/hemu_STU/BME/simi_hospital/anesthesia_kb.json)：麻醉知识库，包含 ASA 分类、药物、麻醉方案、安全规则等。
- [patient_mapper_agent.py](/d:/hemu_STU/BME/simi_hospital/patient_mapper_agent.py)：患者信息标准化、文本结构化、实体归一化、知识库匹配。
- [kb_retriever.py](/d:/hemu_STU/BME/simi_hospital/kb_retriever.py)：知识检索模块。
- [anesthesia_decision_agent.py](/d:/hemu_STU/BME/simi_hospital/anesthesia_decision_agent.py)：麻醉决策模块。
- [run_anesthesia_pipeline_demo.py](/d:/hemu_STU/BME/simi_hospital/run_anesthesia_pipeline_demo.py)：演示入口，串联完整 pipeline。

## 5. 知识库设计

[anesthesia_kb.json](/d:/hemu_STU/BME/simi_hospital/anesthesia_kb.json) 目前包含以下几类内容：

### 5.1 元数据

- 知识库名称
- schema 版本
- 语言
- 更新时间
- 使用场景说明
- 安全警告

### 5.2 参考来源注册表

用于记录知识条目的外部参考来源，例如：

- ASA Physical Status Classification
- Basic Anesthetic Monitoring
- Preoperative Fasting Guidelines
- 部分药品说明书来源

### 5.3 ASA 分级

包含：

- `ASA_I` 到 `ASA_VI`
- `ASA_E`

### 5.4 术前规则

目前包括：

- 术前禁食参考
- 基础监测要求

### 5.5 药物条目

当前示例药物包括：

- `propofol`
- `ketamine`
- `rocuronium`
- `succinylcholine`

每个药物条目包含：

- 中文名
- 药物类别
- 常用给药途径
- 常见用途
- 标签参考信息
- 临床推断说明
- 来源引用

### 5.6 麻醉方案模板

目前包含 3 个示例方案：

- `general_anesthesia_ett`
- `spinal_anesthesia`
- `monitored_anesthesia_care`

每个方案包括适应场景、关键步骤或检查项、监测要求、候选药物等。

### 5.7 安全规则

目前示例规则包括：

- `mh_trigger_avoidance`
- `aspiration_risk_review`
- `residual_blockade_review`

这些规则用于在检索和决策时标出高风险点。

## 6. 运行方式

### 6.1 环境要求

- Python 3.10+

当前代码主要依赖 Python 标准库，不依赖复杂第三方包。

### 6.2 本地规则演示

在项目目录下运行：

```bash
python run_anesthesia_pipeline_demo.py --mode local --decision-mode heuristic
```

含义：

- `--mode local`：使用本地字典作为患者输入
- `--decision-mode heuristic`：使用本地启发式决策逻辑

### 6.3 API 演示

```bash
python run_anesthesia_pipeline_demo.py --mode api --decision-mode api
```

含义：

- `--mode api`：将自然语言病例交给大模型先做结构化
- `--decision-mode api`：由大模型生成最终决策 JSON

注意：

- 当前代码中 `patient_mapper_agent.py` 内置了 API 地址和若干模型配置。
- 若外部 API 不可用，`api` 模式将无法运行。
- 如果用于课程展示，建议优先使用 `local + heuristic`，可复现性更高。

## 7. 输出示例

典型输出由三部分组成：

### 7.1 标准化患者画像

```json
{
  "age": 67,
  "sex": "男",
  "weight_kg": 72,
  "procedure_name": "下肢骨折内固定",
  "asa_hint": "III",
  "comorbidities": ["高血压", "糖尿病"]
}
```

### 7.2 检索结果

```json
{
  "asa_match": "...",
  "candidate_plans": ["..."],
  "matched_safety_rules": ["..."],
  "related_drugs": ["..."],
  "missing_information": ["..."]
}
```

### 7.3 决策结果

```json
{
  "patient_summary": "...",
  "primary_plan": "...",
  "backup_plans": ["..."],
  "risk_flags": ["..."],
  "need_more_info": ["..."],
  "reasoning_trace": ["..."]
}
```

## 8. 项目特点

- 结构清晰，适合展示 Agent pipeline。
- 同时支持规则路径和大模型路径。
- 使用知识库承载领域知识，便于扩展。
- 输出是结构化 JSON，适合后续接前端、评测或数据库。
- 适合做课程作业、毕业设计原型或医疗 AI demo。

## 9. 当前局限

目前这个项目仍然是原型版本，主要局限包括：

- 知识库规模较小，药物、规则和方案覆盖有限。
- 检索逻辑主要基于关键词与启发式规则，尚未引入更强的语义检索。
- 决策逻辑偏演示性质，未覆盖复杂临床场景。
- 缺少系统化测试、评估集和错误分析流程。
- 不应替代麻醉医生的临床判断和机构规范。

## 10. 可扩展方向

后续可以从以下方向继续完善：

- 扩展更多手术场景、并发症和麻醉方案模板。
- 增加禁忌证、药物相互作用、围术期事件处理规则。
- 引入向量检索或 RAG，提高知识召回能力。
- 增加病例评测集，对推荐结果做自动评估。
- 接入前端页面，形成可交互的麻醉决策支持演示系统。
- 增加日志、可解释性追踪和审计记录。

## 11. 免责声明

本项目仅用于教学、研究和原型演示。

其中的知识库、规则、药物信息和方案建议仅作为结构化参考示例，不能替代：

- 麻醉医生的现场评估
- 医疗机构正式流程与规范
- 官方指南与药品说明书
- 真实临床中的个体化决策

如需用于更严肃的医疗场景，必须补充临床审核、来源校验、权限控制、日志审计和安全合规设计。

