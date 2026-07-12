# MatPlotAgent 本地化运行与修改说明

原始项目：[THUNLP/MatPlotAgent](https://github.com/thunlp/MatPlotAgent)

对应论文：[MatPlotAgent: Method and Evaluation for LLM-Based Agentic Scientific Data Visualization](https://arxiv.org/abs/2402.11453)

本文记录原项目下载到本地后遇到的问题、Windows 本地化过程、千问模型接入方式，以及针对视觉反馈退化问题所做的改进。

## 1. 项目原本是做什么的

MatPlotAgent 是一个面向科学数据可视化的多 Agent 研究项目。它希望让大语言模型根据自然语言要求和数据文件自动生成绘图代码，并通过视觉模型检查生成图片，再根据反馈修改代码。

论文中的主要流程是：

1. Query Expansion：扩写用户的简单绘图要求。
2. Code Generation：生成 Python 数据处理和绘图代码。
3. Iterative Debugging：执行代码，根据报错继续修改。
4. Visual Feedback：让多模态模型检查生成图片。
5. Visual Refinement：根据视觉反馈重新生成代码和图片。
6. Evaluation：将结果与 MatPlotBench 标准图片比较并评分。

仓库还包含 MatPlotBench，其中有100个绘图任务、测试数据和标准答案图片。

## 2. 为什么原项目下载后不能直接运行

原仓库更接近论文实验代码，而不是面向普通用户发布的完整应用。它保存了作者实验环境中的一些假设，因此换到本地 Windows 电脑后会遇到以下问题。

### 2.1 数据路径写死在作者电脑上

`workflow.py`、`one_time_generate.py` 等脚本使用了类似下面的绝对路径：

```python
data_path = '/home/zhoupeng/project/LLM/agent/plotagent/.../benchmark_data/'
```

这个目录只存在于作者的 Linux 服务器，本地电脑无法访问。

### 2.2 使用了 Linux 文件复制命令

原流程通过下面的方式复制测试数据：

```python
os.system(f'cp -r {input_path}/* {directory_path}')
```

Windows 默认没有 `cp` 命令，因此即使改对数据路径，复制数据仍可能失败。

### 2.3 API Key 和模型配置需要修改源码

原项目要求在下面的文件里直接填写 Key：

```text
agents/config/openai.py
```

这种方式容易误提交密钥，也不方便在 OpenAI、千问和其他兼容接口之间切换。

### 2.4 使用了旧的模型名称

原项目依赖 `gpt-4-vision-preview`、`gpt-4`、`gpt-3.5-turbo` 等论文实验时期的模型。随着服务端模型更新，部分名称已不适合作为当前默认配置。

### 2.5 本地开源模型配置仍是占位符

`models/model_config.py` 中包含：

```python
"model": "path/to/model"
```

如果想使用 Magicoder、DeepSeek Coder 或 CodeLlama，需要自行下载模型、填写路径、安装 vLLM，并启动对应端口的 OpenAI 兼容服务。

### 2.6 依赖列表偏向论文服务器环境

原始 `requirements.txt` 同时包含 PyTorch、Transformers、vLLM 和多种绘图库。对于只想通过在线 API 运行项目的用户，这些依赖过重，而且 vLLM 通常不适合直接在普通 Windows 环境安装。

### 2.7 生成的代码会直接在本地执行

项目会把模型回答保存成 `.py` 文件并直接运行。这是 Agent 完成绘图任务的必要步骤，但也意味着不能执行来源不可信的提示词或代码。原项目没有提供完善的容器或沙箱隔离。

### 2.8 “脚本执行成功”不等于“图片正确”

原流程主要检查：

- Python 是否抛出异常；
- PNG 文件是否存在。

但 Matplotlib 可以成功执行语义错误的代码。例如，把金额四分位数画到直方图的频数轴上不会触发异常，却会让纵轴范围被拉高，导致真正的柱形被压缩在底部。

### 2.9 视觉反馈可能让结果变差

原项目通常直接采用视觉模型给出的修改意见，没有可靠比较修改前后的图片。如果视觉模型只看出“图片异常”，却没有定位到具体代码错误，第二次生成可能比第一次更差。

## 3. 第一步修改：增加跨平台本地入口

为了保留原论文代码，同时提供一个容易运行的版本，新增了：

```text
local_run.py
```

没有直接重写原来的 `workflow.py`，这样仍然可以对照论文实验实现。

新的本地入口完成了以下工作：

- 使用 `pathlib.Path` 处理 Windows、macOS 和 Linux 路径；
- 使用 `shutil.copy2()` 复制测试数据，不再调用 Linux 的 `cp`；
- 自动创建工作目录；
- 支持通过 `--example` 运行 MatPlotBench 样例；
- 支持通过 `--prompt` 和 `--data` 绘制用户自己的数据；
- 将请求、生成代码、执行日志和图片全部保存在独立工作目录；
- 使用 `subprocess.run()` 执行生成代码，并增加超时控制；
- 默认设置 Matplotlib 非交互式后端；
- 生成失败时把执行日志反馈给模型并自动修复一次。

同时增加了轻量依赖文件：

```text
requirements-local.txt
```

它只安装通过在线 API 绘图所需的 OpenAI SDK、Pandas、NumPy、Matplotlib、Pillow 和 Seaborn，不要求安装 vLLM。

## 4. 第二步修改：使用环境变量管理模型

新增了 `.env.example` 作为配置示例，不再要求把密钥写进源码。

OpenAI 兼容配置为：

```powershell
$env:OPENAI_API_KEY="your-key"
$env:OPENAI_BASE_URL="https://api.openai.com/v1"
$env:MATPLOT_MODEL="gpt-4.1-mini"
```

真实 `.env` 已加入 `.gitignore`，避免误提交密钥。

## 5. 第三步修改：接入阿里云百炼千问

本机已经配置 `DASHSCOPE_API_KEY`，因此本地入口增加了千问自动检测。

当存在 `DASHSCOPE_API_KEY`，或者显式设置：

```powershell
$env:MATPLOT_PROVIDER="qwen"
```

程序会使用：

```text
Base URL: https://dashscope.aliyuncs.com/compatible-mode/v1
Default model: qwen-vl-max
```

选择 `qwen-vl-max` 是因为该流程既需要生成 Python 代码，也需要读取生成图片并给出视觉反馈。

显式配置示例：

```powershell
$env:MATPLOT_PROVIDER="qwen"
$env:DASHSCOPE_API_KEY="your-dashscope-key"
$env:MATPLOT_MODEL="qwen-vl-max"
```

阿里云百炼提供 OpenAI 兼容接口，因此项目仍然可以使用 OpenAI Python SDK，只需替换 API Key、Base URL 和模型名称。相关说明参见：[阿里云百炼 OpenAI 兼容接口](https://help.aliyun.com/zh/model-studio/compatibility-of-openai-with-dashscope)。

## 6. 第一次测试暴露的图表错误

第一次使用千问运行 MatPlotBench 样例76时，API 调用、代码生成、代码执行和图片保存都成功了，但图表本身不正确。

任务要求绘制箱线图和直方图，并标出 Q1、中位数和 Q3。千问生成了：

```python
ax2.hist(women_data)

for q in [Q1, Q2, Q3]:
    ax2.axhline(y=q)
```

普通竖直直方图的坐标语义是：

```text
x轴 = 数据值
y轴 = 频数
```

四分位数属于数据值，所以正确方向应该是：

```python
ax2.axvline(x=q)
```

错误代码把大约2400至3200的金额值放到了频数轴上，而真实频数只有几十。Matplotlib 为容纳这些横线把纵轴拉高到3000以上，真正的直方图因此被压缩成底部的一条细线。

这个错误不会产生 Python 异常，所以只检查退出码和文件存在无法发现它。

## 7. 为什么视觉反馈没有修好

原来的视觉检查主要接收用户要求和图片，但没有充分结合：

- 当前完整绘图代码；
- CSV 的实际列名和数据范围；
- 每个子图的数据位于 x 轴还是 y 轴；
- `axhline` 与 `axvline` 的代码语义。

千问能够看出直方图异常，却把原因误判为数据读取或分箱问题，并继续建议使用 `axhline()`。项目随后无条件相信这条反馈，所以 refinement 生成的最终图片比初始图片更差。

因此问题不是单纯的“千问不行”或“原项目代码完全错误”，而是：

1. 模型第一次生成时发生坐标方向错误；
2. 视觉模型没有把视觉异常准确映射到具体代码；
3. Agent 流程缺少代码语义检查、修改前后比较和退化回滚。

## 8. 第四步修改：加强视觉检查上下文

`inspect_plot()` 现在会同时向视觉模型提供：

- 原始绘图要求；
- 当前生成图片；
- 当前完整 Python 代码；
- CSV 文件名、表头和有限的数据预览；
- 本地静态检查产生的警告。

提示词要求模型对每个子图明确回答：

1. 绘制的变量是什么；
2. 数据值位于 x 轴还是 y 轴；
3. 另一条轴表示什么；
4. `axhline` 或 `axvline` 是否符合数据轴方向；
5. 造成问题的具体代码表达式是什么。

这样可以减少只根据图片猜测原因的情况。

## 9. 第五步修改：基于上一版代码定点修复

refinement 不再只接收一段自然语言反馈，而是同时接收上一版完整代码。

提示词明确要求：

- 修改上一版代码，而不是完全重新设计；
- 只采用能被需求、数据、图片和代码共同支持的反馈；
- 保留已经正确的数据列名、统计方法和图表类型；
- 不直接复制视觉反馈中可能错误的示例代码；
- 修改前检查每个子图的数据轴方向。

## 10. 第六步修改：增加坐标语义静态检查

程序增加了基础 Matplotlib 规则，用于识别常见方向冲突。

当前规则包括：

| 图表形式 | 数据所在轴 | 数据值参考线 |
|---|---|---|
| `hist(data)` | x轴 | `axvline` |
| `hist(data, orientation="horizontal")` | y轴 | `axhline` |
| `boxplot(data, vert=True)` | y轴 | `axhline` |
| `boxplot(data, vert=False)` | x轴 | `axvline` |

例如检测到：

```python
ax2.hist(data)
ax2.axhline(y=q)
```

程序会向视觉模型提供警告，提示检查是否应该改成 `ax2.axvline()`。

静态检查不是为了代替模型，而是为明显的代码语义错误提供确定性保护。

## 11. 第七步修改：新旧图片择优与回退

新流程不再直接用 refinement 覆盖最终图片，而是保留：

```text
initial.png
refined_candidate.png
```

然后把两张图片、两份代码和原始需求一起交给评审模型。评审模型必须在第一行返回：

```text
CHOICE: INITIAL
```

或者：

```text
CHOICE: REFINED
```

只有明确选择 refinement 时才采用新图。输出无法解析、评审失败或 refinement 没有生成有效图片时，程序默认保留初始图。

选择过程会保存为：

```text
selection_review.txt
selected_version.txt
```

这能避免“修改后的代码可以执行，但图片反而更差”的情况。

## 12. 第八步修改：图片完整性和尺寸保护

测试过程中还发现，模型可能将图外标注与：

```python
bbox_inches="tight"
```

组合，导致 Matplotlib 尝试生成极大的画布。其中一次异常图片被扩展到约54亿像素。

因此增加了以下保护：

- 使用 Pillow 验证 PNG 是否可以完整解码；
- 空文件、损坏文件和超大危险图片均判为失败；
- 提示模型将 `figsize` 控制在 `20×20` 英寸以内；
- DPI 控制在200以内；
- 标注尽量放在坐标轴内部；
- 避免图外元素与 `bbox_inches="tight"` 共同扩大画布；
- 如果进程超时但已经生成完整安全的 PNG，可以保留该图片。

## 13. 修改后的样例76结果

最终验证目录：

```text
workspace/demo_qwen_76_guarded_v3
```

最终结果：

```text
workspace/demo_qwen_76_guarded_v3/final.png
```

修复后的图采用统一的数据轴：

- 上方为水平箱线图，金额位于 x 轴；
- 下方为普通直方图，金额也位于 x 轴；
- 两张图均通过竖直红色虚线标记 Q1、中位数和 Q3；
- 直方图的频数轴保持正常范围；
- 柱形和柱顶频数可以正常显示；
- 最小值、Q1、中位数、Q3 和最大值均被标注。

测试过程中千问第一次生成的脚本曾出现字符串引号语法错误，但错误日志反馈和自动修复流程成功恢复，最终完成绘图。

## 14. 现在如何运行

### 14.1 安装依赖

在项目目录执行：

```powershell
python -m pip install -r requirements-local.txt
```

### 14.2 使用千问运行 MatPlotBench 样例

```powershell
$env:MATPLOT_PROVIDER="qwen"
$env:DASHSCOPE_API_KEY="your-dashscope-key"
$env:MATPLOT_MODEL="qwen-vl-max"

python local_run.py `
  --example 76 `
  --workspace workspace\example_76
```

程序会在工作目录中保存：

```text
request.txt                 原始需求
data.csv                    测试数据
generated_initial.py        第一次生成的代码
generated_initial.log       第一次执行日志
generated_repair.py         执行失败时的修复代码
initial.png                 初始有效图片
visual_feedback.txt         视觉检查意见
generated_refined.py        refinement 代码
refined_candidate.png       refinement 候选图片
selection_review.txt        新旧图评审结果
selected_version.txt        最终选择
final.png                   最终图片
```

### 14.3 绘制自己的数据

```powershell
python local_run.py `
  --prompt "读取 data.csv，绘制带标题、坐标轴标签和图例的折线图" `
  --data C:\path\to\data.csv `
  --workspace workspace\my_plot
```

多个数据文件可以重复使用 `--data`：

```powershell
python local_run.py `
  --prompt "比较 sales.csv 和 costs.csv" `
  --data C:\data\sales.csv `
  --data C:\data\costs.csv
```

### 14.4 跳过视觉 refinement

如果只想测试一次代码生成：

```powershell
python local_run.py --example 76 --no-visual-refine
```

## 15. 当前仍然存在的限制

本地化版本提高了可靠性，但仍然是会执行模型生成代码的研究型工具，存在以下限制：

- 静态检查目前只覆盖部分常见 Matplotlib 模式；
- 评审模型仍可能判断错误，只是现在有默认回退保护；
- 图表审美和复杂布局仍依赖所选模型能力；
- 每次完整运行通常需要多次模型调用；
- 运行时间和费用取决于模型、图片大小和 API 服务；
- 生成代码仍应放在隔离环境中执行，不应处理不可信提示词和数据。

## 16. 总结

原始 MatPlotAgent 的论文思路是有效的，但仓库主要服务于作者当时的实验环境，直接下载到本地会受到绝对路径、Linux 命令、旧模型、依赖和配置方式的影响。

本地修改没有替换论文核心思想，而是在其外部补充了一条更容易运行、更加安全和可验证的执行路径：

```text
跨平台加载数据
→ 模型生成代码
→ 执行与错误修复
→ 携带代码和数据的视觉检查
→ 基于上一版定点修改
→ 静态坐标语义检查
→ 新旧图片比较
→ 退化时自动回滚
→ 输出最终图片和完整过程记录
```

这使项目从“论文实验脚本”向“可以在本地重复运行和诊断的绘图 Agent”推进了一步。
