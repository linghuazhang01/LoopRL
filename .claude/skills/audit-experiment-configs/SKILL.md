---
name: audit-experiment-configs
description: 靠阅读来审计 LoopRL 的实验配置（configs/ 下 GRPO / RLTT / PLC 的 yaml）：先找出同一对照组里相似的配置，再逐段逐键对比、说清每份配置在训练什么，判断组内是否只差方法本身，以及一份配置或一次已完成的训练能否作为论文的正式结果。凡是用户新增或修改 configs/*.yaml、改共享超参、问"这几个 config 对齐了吗 / 这份配置在干什么 / 能不能当正式结果 / 这次实验算不算"、要跑正式实验、要把某次 run 写进实验记录或论文时，都用这个 skill，哪怕用户没有说"审计"二字。不写脚本、不跑代码，审计的产出是一份对比表和结论。
---

# 实验配置审计（LoopRL）：读、比、判

实验结论靠的是"同一套设置下只换方法"。审计就是把配置**读懂**，把同组配置**并排比对**，再**判断**
能不能当正式结果。全程靠读文件和推理，不写脚本、不执行代码：审计者要能向作者解释每一处差异
为什么存在，这是脚本给不了的。

规则来源：`CLAUDE.md`"配置对照组"一节（哪些键允许不同）；逐键含义、等价键对应关系、原作默认值
见 [references/parameter-map.md](references/parameter-map.md)，对比时打开它逐键查。

## 第一步：找相似配置

目标是确定被审配置属于哪个对照组、要和谁比。按这个顺序找：

1. **文件名**。约定是 `<方法>_<变体>_<规模>_<卡数>.yaml`：`grpo_baseline_1p4b_8gpu`、
   `rltt_baseline_1p4b_8gpu`、`plc_posterior_exit_gate_1p4b_8gpu` 是 1.4B 组；`grpo_baseline_8gpu.yaml`、
   `rltt_baseline.yaml`（没写规模的是 2.6B）和 `plc_posterior_exit_gate_2p6b_8gpu` 是 2.6B 组。
   以前的 `*_loop_audit.yaml` 诊断变体已删除：逐轮审计现在在所有配置里都打开。
2. **文件头注释**。每份配置开头都写了"属于哪个组、和同组文件差在哪、和另一规模差在哪"。
3. **对照组表**。`CLAUDE.md` 和 README"配置对照组"一节的表格，以及 `tests/test_rltt_baseline.py` 里的
   `MATCHED_GROUPS` 字典（登记在那里的才受测试保护）。
4. **用参数判断**。表里没有、名字也看不出来时，看四个键：`model.num_hidden_layers`（24 = 1.4B，48 = 2.6B）、
   `args.gradient_accumulation_steps`、`args.n_gpus`、`args.max_completion_length`。四个都相同的配置
   应当是一组；只差一个的，是另一组或另一条实验线（比如 1.4B 组对 2.6B 组只差模型）。

把找到的组写下来：组名、成员、各自的方法（`trainer` 加 `plc.enabled`）、谁是参照（有 GRPO 就以 GRPO 为参照，
它是论文的基线）。被审配置没有同组文件时，就和"最像的那份"比，并说明它为什么自成一组。

## 第二步：读懂每份配置在做什么

对比之前先各自读一遍，按段写一张"实验卡片"。读不出来的键去 parameter-map 查，再去原作脚本
`third_party/RLTT/{grpo,rltt}_experiments/*_train.py` 的 argparse 定义和 verl 配置构造处确认。

```
配置：<文件名>
方法：<GRPO | RLTT(loop_weighting=…) | PLC(gamma, lam, prior, gate loss)>；trainer=<grpo|rltt> → 走哪条 actor 前向
模型：Ouro-<规模>，<层数>层 × 4 轮，全参数 / LoRA；模型目录来自哪个环境变量
数据：<训练文件>，prompt 怎么来（parquet 自带 / RLTT prompt），prompt ≤ <n>，生成 ≤ <n>
批次：每步 <题数> × <采样数> = <rollout 数>，累积 <k> 步，mini-batch <m> 题 → 每 <k> 步一次更新，共 <steps> 步 = <updates> 次更新
优化：AdamW8bit lr <x> <调度> + warmup <r>（实际占更新数的 <r'>），betas <..>，wd <..>，grad clip <..>（GRPO 看 grad_clip，RLTT 看 max_grad_norm）
目标：clip 0.2，token-mean，KL（对照组统一关闭：beta 0、use_kl_loss false、kl_loss_coef 0），advantage 组内标准化
rollout：vLLM 占显存 <p>，kv_layout <..>，temperature <t>
输出：每 <n> 步存 checkpoint（含 hf_model？），验证开/关，wandb <mode> project/tags
诊断：audit.loop_distribution 开/关
隐含：模型目录的 rltt_detach_exit_pdf 决定 gate 是否被训练；RLTT 会改写 config.json 的 chunk size
```

写卡片时顺手核对自洽性：`verl_overrides.data.train_batch_size` 应等于 `args.num_prompts_per_batch`；
`actor.ppo_mini_batch_size` 应等于 题数 × 累积步数；`ppo_max_token_len` 应 ≥ prompt + 生成；
GRPO 的 `actor.grad_clip` 应等于 `args.max_grad_norm`；RLTT 的 `use_rltt_model` 为 true、
`lr_scheduler_type` 显式 `constant`。卡片里的"每步多少题、几次更新"要自己算一遍，不要抄注释。

## 第三步：并排对比

把同组文件并排打开，按段、按键顺序走一遍，每一处不同都归入三类之一并记进表：

| 类别 | 判定 | 例子 |
|---|---|---|
| 方法键 | 允许不同，记下即可 | `trainer`、`plc.*`、RLTT 的 `loop_weighting / progressive_alpha`、`wandb.run_name/tags/notes`（`project` 统一为 `looprl-ouro`、`entity` 为 null，属共享键） |
| 等价键 | 名字不同但要**取值等价**，对着 parameter-map 核对值 | GRPO 的 `use_ppo=false` 与 `verl_overrides` 里 `use_kl_in_reward=false / grad_clip / param_offload=true`；RLTT 的 `use_rltt_model=true / rltt_model_path / rltt_logprob_chunk_size / ref_offload=true / no_gradient_checkpointing=false / lr_scheduler_type=constant` |
| 共享键 | 必须逐字相同，不同即为问题 | 其余一切，包括 `env`、`rollout`、`model`、`audit`、注释里看不见的默认值 |

三件容易漏的事：

- **没写出来的键也在对比范围内**。`args` 里缺的键取原作默认值，而两份脚本的默认值有几处不同
  （`ppo_max_token_len`、`lr_scheduler_type`、ref offload、grad clip、`use_kl_in_reward`，见 parameter-map
  末尾）。一份写了、另一份没写，就要把默认值填上再比。
- **原作脚本自己写死的值**要知道：`betas (0.9, 0.999)`、`mini-batch 128`、`train_batch_size = 题数×采样数`、
  KL 进 reward（GRPO）。`verl_overrides` 存在的意义就是改正这些，所以组内每份文件都得有对应的覆盖，
  缺一个就是两种方法实际跑的不一样。
- **注释不算证据**。注释说"与 X 一致"，仍要看值。两份文件注释不同但值相同，不是问题；值不同注释相同，是问题。

对每个"问题"写清：键、各文件的值、哪个值是对照组想要的（以 GRPO 基线和论文 Table 6 为准）、
要改哪几份文件。改共享键必须整组一起改。

## 第四步：判定能否作为正式结果

正式结果 = 可以进论文主表 / 主曲线、与同组其他方法直接对比的数据。分三层，缺一层就降级。

**配置层**（读 yaml）
- 登记在 `MATCHED_GROUPS` 且同组对比表里没有共享键差异、等价键取值全部对上。
- `audit.loop_distribution` 组内一致（现在所有配置都是 `enabled: true`；审计只读、不改训练，组内一致即可）；
  `rollout.kv_layout=full`。
- `model.num_hidden_layers` 已写；wandb `enabled: true`、`mode: online`、`project: looprl-ouro`，
  tag 标出方法与模型规模（AGENTS.md 规则 4）；`max_steps=140`、`seed` 固定、
  checkpoint 含 `hf_model`。
- 所有与论文不同的默认值都已显式写出（题数、温度、长度、`ppo_max_token_len`）。

**运行层**（读 `<output_dir>/rltt_baseline_config.yaml` 和 `rltt_baseline_resolved.json`）
- `cli_overrides` 为空，或只有 `wandb.*` 命名键。改了 `args.*`、`verl_overrides.*` 等共享键的 run
  不是该配置的正式结果；只改 `plc.*` 的是**带标注的消融**，进消融表，不冒充主结果。
- 同组各 run 的已解析配置剥掉方法键后完全一致；`train_file` 和训练的模型目录相同；各 run 都有
  `audit/`（审计组内一致），且 `audit/step_*/` 里没有 `error_rank*.txt`。
- 都跑完同样的步数；中断续跑要注明。

**环境层**（配置里没有，必须查并写进实验记录）
- 代码 commit（工作区干净）；`third_party/RLTT` submodule commit 未动。
- 模型目录 `config.json` 的 `num_hidden_layers`、`rltt_detach_exit_pdf`、`rltt_logprob_chunk_size`，
  以及 `fetch_model_manifest_*.json` 的 SHA256；同组 run 用同一个目录且中途没被重建。
- 数据 manifest、`configs/rltt_env.lock.txt`、GPU 型号与卡数。

结论三档：**可作为正式结果** / **有条件可用**（列出每条保留意见和它对结论的影响，例如"审计设置与
同组其他 run 不同，只能比训练曲线、不能比审计指标"）/ **不可作为正式结果**（说明是 smoke、诊断还是消融，以及怎样重跑才算）。
环境层没核对的项写"未核对"，不要默认通过。

## 新增配置时

1. 从同组已有文件复制，只改方法相关的键；共享键一个都不碰，注释也同步改。
2. 按第二、三步把新文件与组内每一份并排比对，直到只剩方法键和取值对上的等价键。
3. 登记：`tests/test_rltt_baseline.py` 的 `MATCHED_GROUPS`，以及 `test_mini_batch_is_one_accumulated_update`、
   `tests/test_plc.py::test_other_configs_have_plc_off` 的参数列表（PLC 配置不进后者）；README 和
   `CLAUDE.md` 的对照组表格加一行。登记完由仓库自带的 `pytest` 做回归确认，那是改完文件后的检查，
   不是审计本身。
4. 文件头注释写清所属组、与同组文件的差异只有方法、与另一规模的差异是什么。

临时实验不要新建配置，用命令行 dot-list 覆盖；消融也走覆盖（`plc.gamma=0`、`plc.lam=0`、
`plc.use_gate_loss=false`）。

## 报告模板

```
# 配置审计：<组名或文件名>　<日期>
## 结论
<三档之一>，一句话原因。
## 对照组与实验卡片
<组成员、参照；每份配置一张卡片，或一张合并的卡片标出差异列>
## 逐键对比
| 键 | <文件A> | <文件B> | <文件C> | 类别 | 说明 |
<只列有差异的键；等价键写出"对应的值是否一致">
## 隐含差异
<随 trainer 变化的前向路径；模型目录对 gate 的影响；未写出的默认值>
## 正式就绪度
<配置层 / 运行层 / 环境层各自的问题；未核对的写"未核对">
## 下一步
<要改哪些文件、改成什么值、要补哪些登记、要重跑哪些 run>
```

把每个问题翻译成"哪个值应该是什么、要改哪几份文件"，把每条保留意见翻译成"对论文结论的影响"，
不要只说"不一致"。
