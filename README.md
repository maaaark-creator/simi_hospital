# Anesthesia Agent

一个面向麻醉场景的原型 Agent，用于演示患者信息结构化、知识库检索和麻醉方案建议生成的基础流程。

当前版本定位为独立 Demo / 临时分支原型，目标是先把核心链路跑通，后续再与更大的系统进行集成与合并。

## 1. 模块定位

`Anesthesia Agent` 负责围绕单个围术期患者，完成以下任务：

1. 接收原始患者信息。
2. 将输入整理为统一的结构化患者画像。
3. 根据麻醉知识库检索候选方案、药物和安全规则。
4. 输出一个结构化的麻醉决策建议结果。

它本质上是一个轻量级的决策支持链路，而不是完整的临床系统。

## 2. 主要想法

当前 Agent 采用“规则标准化 + 知识库检索 + 决策输出”的分层设计。

### 2.1 设计目标

- 先把麻醉场景中的核心信息流打通。
- 输出结构化结果，方便后续接前端、评测、接口层或主系统。
- 保留一定可解释性，而不是只返回自由文本。
- 在不依赖复杂工程框架的前提下，尽量清晰展示 Agent pipeline。

### 2.2 基本流程

整体流程如下：

`Raw Patient Input -> Patient Mapper -> Knowledge Retriever -> Decision Agent -> Structured Output`

其中：

- `Patient Mapper` 负责患者信息标准化、文本结构化和实体归一化。
- `Knowledge Retriever` 负责从麻醉知识库中召回候选方案、药物和规则。
- `Decision Agent` 负责生成首选方案、备选方案、风险提示和解释信息。

### 2.3 当前方法特点

- 支持结构化输入和自然语言输入两条路径。
- 兼容本地启发式决策与模型 API 决策两种模式。
- 以 JSON 作为核心输入输出格式，便于后续系统集成。
- 以知识库为中间层，避免把所有逻辑都写死在 prompt 或规则中。

## 3. 文件结构

```text
simi_hospital/
├─ README.md
├─ LICENSE
├─ anesthesia_kb.json
├─ patient_mapper_agent.py
├─ kb_retriever.py
├─ anesthesia_decision_agent.py
└─ run_anesthesia_pipeline_demo.py
```

各文件职责如下：

- [README.md](/d:/hemu_STU/BME/simi_hospital/README.md)：当前模块说明文档。
- [LICENSE](/d:/hemu_STU/BME/simi_hospital/LICENSE)：许可证文件。
- [anesthesia_kb.json](/d:/hemu_STU/BME/simi_hospital/anesthesia_kb.json)：麻醉知识库，保存 ASA 分类、药物、麻醉方案和安全规则。
- [patient_mapper_agent.py](/d:/hemu_STU/BME/simi_hospital/patient_mapper_agent.py)：患者输入标准化、病例结构化、实体归一化和知识库匹配逻辑。
- [kb_retriever.py](/d:/hemu_STU/BME/simi_hospital/kb_retriever.py)：知识检索逻辑。
- [anesthesia_decision_agent.py](/d:/hemu_STU/BME/simi_hospital/anesthesia_decision_agent.py)：麻醉决策逻辑。
- [run_anesthesia_pipeline_demo.py](/d:/hemu_STU/BME/simi_hospital/run_anesthesia_pipeline_demo.py)：本地 demo 入口。

## 4. 模块组成

### 4.1 Patient Mapper

主要职责：

- 统一中英文患者字段。
- 将原始输入转换为标准化 `normalized_profile`。
- 可选调用外部模型 API，对自然语言病例做结构化抽取。
- 可选调用外部模型 API，做医学实体归一化。

核心接口：

- `normalize_patient_input(raw_patient)`
- `build_case_record(raw_patient)`
- `build_case_record_from_text(raw_patient_text)`
- `call_patient_structuring_api(raw_patient_text)`
- `call_medical_entity_linking_api(normalized_patient)`

### 4.2 Knowledge Retriever

主要职责：

- 映射 ASA 分类。
- 推断候选麻醉方案。
- 触发安全规则。
- 汇总相关药物。
- 标记当前病例缺失的信息。

核心接口：

- `retrieve(normalized_patient, entity_linking=None)`

### 4.3 Decision Agent

主要职责：

- 从候选方案中选择首选方案。
- 输出备选方案。
- 标记风险提示。
- 给出需要补充的信息。
- 生成简要解释链路。

核心接口：

- `decide(normalized_patient, retrieval_result)`
- `decide_with_api(normalized_patient, retrieval_result)`

## 5. 输入

当前 Agent 支持两类输入。

### 5.1 结构化输入

适用于本地 demo、测试用例和后续服务接口直接调用。

示例：

```json
{
  "年龄": 67,
  "性别": "男",
  "体重": 72,
  "拟行手术": "下肢骨折内固定",
  "ASA分级": "III",
  "过敏史": "无",
  "基础疾病": ["高血压", "糖尿病"],
  "禁食情况": "已禁食 8 小时",
  "气道评估": "张口尚可，暂未提示困难气道",
  "急诊": "择期"
}
```

### 5.2 自然语言输入

适用于后续和上游问诊、病历摘要或多 Agent 系统对接。

示例：

```text
患者，男，67岁，72kg。拟行下肢骨折内固定术。既往有高血压、糖尿病。ASA III级。已禁食8小时。气道评估暂未提示困难气道。过敏史无特殊。
```

### 5.3 标准化后的内部输入格式

进入检索和决策模块前，输入会统一为如下字段：

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
- `raw_input`

## 6. 输出

当前 Agent 的输出是结构化 JSON，便于日志记录、前端展示或后续接口集成。

### 6.1 中间输出

在完整 pipeline 中，会依次产生以下中间结果：

- `structured_patient`
- `normalized_profile`
- `entity_linking`
- `retrieval_result`

### 6.2 最终输出

决策模块最终返回的核心字段包括：

- `patient_summary`
- `primary_plan`
- `backup_plans`
- `risk_flags`
- `need_more_info`
- `reasoning_trace`

### 6.3 输出示例

```json
{
  "patient_summary": "67岁，男，拟行下肢骨折内固定，ASA III",
  "primary_plan": {
    "id": "spinal_anesthesia",
    "name_zh": "椎管内麻醉"
  },
  "backup_plans": [
    {
      "id": "general_anesthesia_ett",
      "name_zh": "全身麻醉-气管插管"
    }
  ],
  "risk_flags": [
    "存在基础疾病，需进一步评估"
  ],
  "need_more_info": [
    "缺少身高信息",
    "缺少实验室检查摘要"
  ],
  "reasoning_trace": [
    "已参考术式信息",
    "已纳入 ASA 提示"
  ]
}
```

## 7. 依赖知识库

当前 Agent 强依赖本地知识库文件 [anesthesia_kb.json](/d:/hemu_STU/BME/simi_hospital/anesthesia_kb.json)。

### 7.1 知识库包含内容

目前知识库主要包含以下几类信息：

- `metadata`
- `source_registry`
- `asa_physical_status`
- `preop_rules`
- `drugs`
- `anesthesia_plans`
- `safety_rules`
- `placeholders_for_future`

### 7.2 当前知识库承担的作用

- 提供 ASA 分类映射目标。
- 提供候选麻醉方案模板。
- 提供药物条目与相关说明。
- 提供安全规则和高风险提示条件。
- 作为后续统一知识维护的入口。

### 7.3 当前知识库边界

当前知识库仍是原型级 scaffold，覆盖范围有限：

- 药物数量较少。
- 麻醉方案模板较少。
- 安全规则仍偏示例性质。
- 缺少更细的并发症、禁忌证、术前评估和围术期事件处理规则。

## 8. Demo 运行方式

### 8.1 环境要求

- Python 3.10+

当前代码主要使用 Python 标准库。

### 8.2 本地规则模式

在项目目录下运行：

```bash
python run_anesthesia_pipeline_demo.py --mode local --decision-mode heuristic
```

说明：

- `--mode local`：直接使用本地患者字典。
- `--decision-mode heuristic`：使用本地启发式规则输出决策结果。

适合：

- 跑通最小 demo
- 做课程展示
- 做本地调试
- 避免依赖外部 API

### 8.3 API 模式

```bash
python run_anesthesia_pipeline_demo.py --mode api --decision-mode api
```

说明：

- `--mode api`：先将自然语言病例发送给模型 API 做结构化。
- `--decision-mode api`：再由模型 API 生成最终决策输出。

适合：

- 演示自然语言输入链路
- 验证模型增强流程

注意：

- 当前 API 地址和模型配置写在 [patient_mapper_agent.py](/d:/hemu_STU/BME/simi_hospital/patient_mapper_agent.py) 中。
- 若外部模型服务不可用，则 `api` 模式无法运行。

## 9. 当前接口形态

当前分支中的接口主要是 Python 类方法，尚未封装为统一服务 API。

### 9.1 当前已具备的内部接口

- `PatientProfileMapperAgent.normalize_patient_input`
- `PatientProfileMapperAgent.call_patient_structuring_api`
- `PatientProfileMapperAgent.call_medical_entity_linking_api`
- `KnowledgeRetriever.retrieve`
- `AnesthesiaDecisionAgent.decide`
- `AnesthesiaDecisionAgent.decide_with_api`

### 9.2 当前 demo 串联入口

- [run_anesthesia_pipeline_demo.py](/d:/hemu_STU/BME/simi_hospital/run_anesthesia_pipeline_demo.py)

这个脚本负责串联：

1. 患者输入
2. 结构化与映射
3. 知识检索
4. 决策输出

## 10. 后续待集成接口

由于当前只是临时 branch，后续建议在与主系统合并时重点补齐以下接口层。

### 10.1 上游输入接口

建议预留：

- 病历摘要输入接口
- 问诊 Agent 输出接入接口
- HIS / EMR 结构化字段接入接口
- 术前评估表单接入接口

### 10.2 下游输出接口

建议预留：

- 标准 JSON 返回接口
- 前端展示接口
- 决策解释接口
- 审计日志接口
- 人工复核接口

### 10.3 知识库管理接口

建议预留：

- 知识库版本管理接口
- 规则热更新接口
- 药物条目更新接口
- 手术模板扩展接口

### 10.4 模型服务接口

建议预留：

- 患者文本结构化接口
- 实体归一化接口
- 候选方案排序接口
- 决策结果生成接口

### 10.5 评测与监控接口

建议预留：

- 用例回放接口
- 结果评测接口
- 错误分析接口
- 线上日志与告警接口

## 11. 当前分支状态与合并建议

当前模块适合作为一个独立可运行原型保留，后续合并时建议按以下方向整理：

- 保留 `mapper / retriever / decision` 三层结构。
- 将知识库路径、模型配置和 API 地址外置到配置文件或环境变量。
- 增加统一的 service 层入口，而不是直接依赖 demo 脚本。
- 增加标准 request / response schema。
- 增加最小测试用例和样例输入输出。
- 清理临时硬编码内容，尤其是模型配置与 API 相关信息。

## 12. 已知限制

当前版本仍然是原型，因此有以下限制：

- 规则和知识覆盖有限。
- 缺少统一服务化封装。
- 缺少标准接口文档。
- 缺少自动化测试。
- 缺少完整异常处理与日志机制。
- 不适合直接用于真实临床场景。

## 13. 免责声明

本项目仅用于教学、研究、课程设计和原型演示。

当前知识库、规则和输出结果仅作为结构化参考示例，不能替代：

- 麻醉医生的专业判断
- 医疗机构正式规范
- 官方指南与药品说明书
- 真实围术期场景中的个体化决策

