# Disobind 新人上手指南（中文）

> 面向首次接触 Disobind 代码库的同学：先看整体，再按“能跑通 → 能读懂 → 能修改”逐步深入。

## 1. 代码库整体结构

Disobind 仓库大体可以按“**推理主入口 / 数据构建 / 模型训练 / 结果分析 / 预训练权重与配置**”来理解：

- `run_disobind.py`：**对外主入口**，读取用户输入（蛋白片段对），拉取 UniProt 序列，生成 embedding，调用模型，输出接触图/界面位点预测。  
- `dataset/`：**数据工程流水线**，包含从多个数据库整理原始样本、构建二元复合物、去冗余划分、生成 embedding 等脚本。  
- `src/`：**训练与模型核心**，包括训练循环、损失函数、指标计算、数据加载、网络结构定义。  
- `analysis/`：**评估与论文分析**，整合 Disobind 与 AF2/AF3 的结果，计算指标与出图。  
- `params/`：**模型配置文件**（yml），训练和推理都会依赖这些超参数设置。  
- `models/`：**已训练权重**（当前以 `Epsilon_3` 版本为主）。  
- `database/`：**数据来源与中间产物**（原始下载文件、输入索引文件等）。  
- `example/`：**示例输入与样例结构文件**，用于快速验证运行流程。

## 2. 你需要先理解的关键内容

### 2.1 业务与任务定义（先统一概念）

Disobind的核心是：给定一对蛋白片段（其中第一个通常是 IDR），预测：

1. **interaction/contact map**（残基对-残基对是否接触）
2. **interface residues**（每条链上哪些位点参与相互作用）

并支持不同 coarse-grained（CG）尺度（1/5/10）。

### 2.2 输入输出契约（最容易踩坑）

`run_disobind.py` 支持两种 CSV 行格式：

- 仅 Disobind：`UniProt_ID1,start1,end1,UniProt_ID2,start2,end2`
- Disobind + AF2：额外附上结构路径、PAE json、chain 与 offset 信息

输出包括：

- 结果 CSV（可读结果）
- `Predictions.npy`（程序化复用的嵌套字典）

### 2.3 模型执行主链路（建议按调用顺序读）

1. `run_disobind.py` 解析输入与任务配置（是否预测 contact map、CG 级别等）
2. 创建/读取 embedding（当前默认 T5）
3. 从 `analysis/params.py` 提供的参数路径加载模型配置与权重
4. 按 task（interaction/interface × CG）逐批推理并汇总保存

### 2.4 训练框架关键点（在 `src/`）

- `src/build_model.py`：`Trainer` 封装训练/验证/测试流程，包含 loss、metrics、校准逻辑
- `src/dataset_loaders.py`：负责 `.npy` 数据集加载与 DataLoader 组织
- `src/models/`：按版本管理模型结构（`Epsilon_3.py` 等）
- `src/utils.py`：任务输入重组（interaction/interface 的 target 变换、binning 处理）

### 2.5 数据流水线关键点（在 `dataset/`）

数据制作是多脚本串联流程：

1. 汇总多数据库条目
2. 下载结构并构建二元复合物
3. 合并复合物并去冗余
4. 划分训练/测试与 OOD 集
5. 生成 embedding

如果你后续想复现实验，`dataset/README.md` 是最直接执行清单。

## 3. 给新人的学习路线（推荐 2~3 周）

### 第 0 阶段：先跑通（半天）

- 安装依赖，直接跑：`python run_disobind.py -f ./example/test.csv`
- 明确输出目录里每个文件含义。

### 第 1 阶段：理解推理（1~2 天）

- 通读 `run_disobind.py`，重点看：输入解析、task 选择、批处理、模型调用。
- 画一张“输入 CSV → embedding → model → postprocess → 输出”的流程图。

### 第 2 阶段：理解训练（3~5 天）

- 读 `src/README.md` 和 `src/build_model.py`
- 把 `prepare_input`（`src/utils.py`）里 interaction/interface 的 target 变换自己手算一遍
- 用小规模样本跑 1 个 epoch，确认你能解释 loss/metric 的变化

### 第 3 阶段：理解数据与评估（3~5 天）

- 跑 `dataset/README.md` 和 `analysis/README.md` 里的最小子流程（可少量数据）
- 重点理解 OOD 测试集与 AF2/AF3 对比是如何拼装到一起的

### 第 4 阶段：开始做改动（持续）

优先改“低风险高收益”模块：

- 新增日志/参数校验
- 把硬编码路径改为配置
- 增加 smoke test（例如 1~2 条样本的端到端运行）

## 4. 新人最常见问题与建议

- **脚本命名不一致**：README 与实际脚本有时存在轻微差异，遇到报错先核对文件名。  
- **路径耦合较高**：多个脚本在构造函数里写死路径，迁移环境前先统一路径配置。  
- **数据/模型体量较大**：首次运行尽量从 `example/` 与小批量开始。  
- **多任务 + 多分辨率**：不要一次改太多维度，建议固定 objective 和 CG 后逐步扩展。

## 5. 建议的“入门检查清单”

在你提交第一段代码前，建议确认：

- [ ] 能解释 input CSV 两种格式和各字段意义
- [ ] 能说明 interaction vs interface 的输出差异
- [ ] 知道 CG=1/5/10 对应什么
- [ ] 知道推理入口、训练入口、分析入口分别是哪几个脚本
- [ ] 能独立跑通 example 并定位输出文件

---

如果你是导师：建议让新人先完成一个很小的任务（如增加输入校验并附带示例），比直接改模型结构更能建立上下文与信心。
