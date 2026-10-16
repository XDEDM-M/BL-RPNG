# BL-RPNG Allen–Cahn 正式实验：seed 42

## 结论摘要

本轮已完成 128×128 数据集 A 预训练、64×64 数据集 B 的 24 个主因果微调任务、4 个消融任务，以及冻结后的 128×128/20-step 细网格盲测。主指标是 50 个未见同族边界、每个边界 2 个新初值上的 per-rollout relative L2。

seed 42 的核心正结果是：**Newton 路径预训练明显改善了少样本跨分辨率泛化；固定强度路径微调的收益依赖数据预算；将路径损失退火后，在 8-boundary 预算上得到本轮最佳且统计显著的稳健方案。**

- 4-boundary：`BL-RPNG pretrain → final-only` 相对普通 `final-only pretrain → final-only` 改善 **10.49%**，单 seed 条件分层 bootstrap 95% CI 为 **[1.05%, 18.81%]**。
- 8-boundary：`BL-RPNG pretrain → annealed BL-RPNG finetune` 相对普通预训练基线改善 **10.86%**，95% CI 为 **[3.27%, 18.45%]**。
- 同一 8-boundary 路径预训练初始化下，退火微调相对 final-only 微调改善 **3.34%**，95% CI 为 **[0.51%, 6.23%]**。
- 所有 boundary-lifted 模型的边界误差均为 **0**；原始三通道 scratch 的最大边界误差为 **0.930**。

目前不能把单 seed 条件 CI 当作跨随机种子论文证据；它只用于筛选主假设与下一轮配置。

## 数据与门控

- 冻结 manifest：1,596 个 rollout case。
- A：1,000 train + 200 validation rollouts，128×128 输出网格，每条 10 步。
- B coarse：32 train、16 seen-validation、32 unseen-validation rollouts，64×64 输出网格。
- B fine/blind：B-train 配对 32、seen 16、unseen 100、frequency-OOD 100、amplitude-OOD 100 rollouts，128×128 输出网格，每条 20 步。
- nonlinear convergence = 1.0；valid Newton correction = 1.0；边界/lift/correction trace error = 0。
- A 的 BL-RPNG validation relL2 为 0.017195，对普通预训练的 0.017070 仅差 0.73%，满足 10% non-inferiority 门控。
- 数据和共享 manifest 的 SHA256 校验见 `allen_cahn_formal_data_verification.json`；盲测前冻结了 35 个 checkpoint、5 个数据文件和 60 个源码文件。

## 主因果矩阵

128×128、20-step、blind-unseen relative L2（越低越好）：

| B 边界预算 | scratch → final | final-pretrain → final | BL-pretrain → final | BL-pretrain → fixed BL |
|---:|---:|---:|---:|---:|
| 1 | 0.950209 | 0.054713 | 0.047402 | **0.044150** |
| 2 | 0.678976 | **0.051801** | 0.053270 | 0.056840 |
| 4 | 0.398001 | 0.050056 | **0.044807** | 0.046833 |
| 8 | 0.248160 | 0.048815 | **0.045017** | 0.045231 |

该矩阵说明两件事：

1. 少量 B 数据从零训练并不足以学习可迁移的动力学；A 预训练是主要性能跃迁来源。
2. Newton 路径预训练在预算 1、4、8 上优于普通预训练，但固定 `gradient ratio=0.15` 的路径微调只在预算 1 明显受益，在预算 2/4/8 出现目标冲突。

## 退火微调与消融

- 8-boundary 退火 BL-RPNG：**0.043514**，优于固定 BL-RPNG 的 0.045231，也优于 BL-pretrain → final 的 0.045017。
- 4-boundary 退火 BL-RPNG：0.046304，优于固定 BL-RPNG 的 0.046833，但仍不及 BL-pretrain → final 的 0.044807。
- 4-boundary raw scratch：0.224662，最大边界误差 0.930。
- 4-boundary raw legacy path：0.059613，最大边界误差 0.472。
- 4-boundary dual-stage：0.053001，低于单阶段方法且训练成本显著更高。

8-boundary 退火方案的 OOD 指标为：frequency-OOD 0.051303，amplitude-OOD 0.061825。普通 `final-pretrain → final` 对应为 0.054950 和 0.067195，因此两个 OOD 方向均未恶化。

真实 fine 初值与 coarse 初值插值到 fine 网格的误差差异通常只有约 0–2%，说明当前收益不是依赖某一种初值输入方式的偶然现象。B-train coarse-solver 插值误差下限约为 0.01877，当前最佳模型仍明显高于该下限，后续仍有提升空间。

## 理论解释与方法修订

boundary lifting 把非齐次 Dirichlet 问题转到齐次迹空间；Newton correction 监督则约束了神经算子的局部切空间，使预训练表示更接近离散非线性算子的逆 Jacobian 作用。这解释了路径预训练在跨边界、跨网格微调中的样本效率优势。

固定路径微调在部分 B 预算上失效，最合理的机制是：coarse-grid、固定边界的小数据 Newton correction 与最终 fine-grid rollout 目标之间存在 fidelity/covariate shift；持续保持 0.15 的辅助梯度比例会在后期阻止模型对最终 rollout 目标收敛。退火结果支持“路径损失是早期优化几何先验，而非全程等强度目标”的修订。

因此下一轮论文主模型建议冻结为：

`Boundary lift + Newton path pretraining + budget-aware path annealing during finetuning`

并保留 `BL-pretrain → final-only` 作为关键机制对照。下一轮先补 seeds 0/1/2/3，检验 4/5 seed 胜出和跨 seed 分层 bootstrap；在多 seed 结论成立前不启动 Gray–Scott 正式确认。

## 完整性

- 16/16 自动测试通过。
- PyTorch 2.11.0+cu130、2×CUDA GPU 可用，SfePy worker 可用，未修改 Conda 包。
- 所有 runner 的 workspace audit 均记录 `outside_workspace_unchanged=true`。
- 盲测后只放宽了一个与 cuFFT batch-shape 数值差异有关的集成测试容差（差异约 1.95e-5）；训练、checkpoint、数据与盲测评估代码均未更改。

机器可读结果见 `allen_cahn_formal_seed42_blind_summary.json`，全部单模型明细见同目录下的 `blind_b*_s42_*.json`。
