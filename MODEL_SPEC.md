# 02 Modeling SAP Freeze

状态：`FROZEN_BEFORE_MODEL_RESULTS`  
冻结时间：2026-09-18T12:42:00+08:00  
随机种子：`20260918`

本文件在读取任何正式模型性能结果前写入。其 SHA-256 另存于 `SAP_FREEZE.sha256`，后续不得因结果修改。

## 1. 主要与次要问题

主要问题：使用 ICU 0–12 h 的可运输共同信息在 MIMIC 开发并内部验证死亡风险模型，锁定后检验其在 SICdb 的辨别、校准和记录稀疏度敏感性。  
次要问题：在 SICdb dense-core 同患者中，比较 5、15、60 min 表征的预测性能，主要 contrast 为 5 min minus 60 min。

## 2. 特征定义

### Model 0：snapshot logistic baseline

固定原始特征顺序：age、sex_male、H0 HR、H0 MAP、H0 SpO2、三项 H0 缺失指示。连续变量训练折内 median imputation 和标准化；sex 缺失以训练折中位数填补。

### Model 1/2：预设摘要

age、sex_male，加每个信号的：12 h median、physiologic extreme（HR=max，MAP/SpO2=min）、SD、按小时中点的线性 slope、threshold burden（HR>100、MAP<65、SpO2<92）、有效小时比例、最长 missing streak。每个数值特征附固定 missing indicator。缺失填补仅在训练折拟合；Model 1 连续量标准化，Model 2 使用相同原始特征与指示。

### Model 3：compact GRU-D

输入固定为 12×3 values、mask、time-since-last-observation，age/sex 在最终层拼接。单层、hidden size 8 或 16、dropout 0 或 0.2、weight decay 0 或 1e-4；目标可训练参数 <20,000。观测值均值/SD 仅由训练折估计，missing 保持 mask，不预插值。主分析 BCE 的 positive class weight 仅由训练折事件率计算；另做一次未加权 5-fold 敏感性。若实现失败，唯一预设 fallback 为单层 GRU + mask + delta-time，并在 QC 中明示。

## 3. 内部验证与训练内调参

- MIMIC repeated stratified 5-fold CV，5 repeats；每个患者每 repeat 恰有一条 OOF prediction。
- repeat seeds 固定为 `20260918 + repeat_index`；每个 repeat 的五折复用于全部模型。
- Model 0：L2 logistic，C=`0.01,0.1,1,10`。
- Model 1：elastic-net logistic，C=`0.01,0.1,1,10`，l1_ratio=`0,0.5,1`。
- Model 2：XGBoost，max_depth=`2,3`，learning_rate=`0.03,0.10`，n_estimators=`100,300`，min_child_weight=`5,10`，subsample=0.8，colsample_bytree=0.8，reg_lambda=1。
- Model 0–2 的超参数仅在外层训练集内 3-fold 分层 CV 选择，排序规则为 mean log loss、再 Brier、再较小容量。
- Model 3 在每个外层训练集内固定分层 validation split（20%）调参/early stopping；最多 100 epochs、patience 10，按 validation log loss、再 Brier、再较小 hidden/dropout/weight decay选择。validation 不进入该外折拟合。
- 所有 imputation/scaling/tuning 均严格位于外折训练数据内；不设 MIMIC hold-out。
- 内部汇总以每 repeat OOF 指标的均值和 2.5%–97.5% repeat 分位数描述稳定性；另报告全体 repeat-patient OOF 行。

## 4. 全 MIMIC 拟合与锁模

- 各模型最终超参数取 25 个外层训练结果中的众数；平票依次按训练内平均 validation log loss、Brier、较小容量裁决。
- 用全 MIMIC 仅按上述锁定超参数重新拟合预处理与模型。
- 在任何 SICdb 预测之前保存 feature order、预处理对象、模型对象、参数量与文件 SHA-256，并生成 `04_locked_models/LOCKED_MODEL_MANIFEST.json` 及独立 hash sidecar。
- 外部阶段只允许 load/transform/predict；日志和代码扫描不得出现 `fit`, `partial_fit` 或 tuning 分支。

## 5. SICdb 锁定外部验证

- Primary representation：12×3 hourly median。
- Sensitivity representations：固定 sparse phases 0/15/30/45/59 min，每小时恰取对应 minute slot；全部报告范围和中位值。
- 四模型均报告 AUROC、AUPRC、Brier、log loss、calibration intercept、calibration slope；AUPRC 同时报事件率 no-skill baseline。
- 95% CI 采用 patient-level nonparametric bootstrap 2,000 次；同一 bootstrap index 同时用于全部模型/表示，模型差值为 paired bootstrap。
- 不在 SICdb 重拟合。Secondary transport diagnostic 可仅计算 intercept-only 与 intercept+slope recalibration，不替换 primary prediction。
- Decision curve 为 secondary，阈值固定 0.02–0.25、步长 0.01，并输出完整曲线。

## 6. SICdb 分辨率实验

- 先报告 dense-core death 数。若 <120，Model 3 不在该层训练；若 ≥120，可运行同一 compact GRU-D。
- dense-core repeated stratified 5-fold CV，5 repeats；5/15/60 min 使用完全相同 fold assignments。
- 对 Model 1/2，各分辨率根据对应 bins 计算同名预设摘要，特征数不变；对 Model 3，仅 sequence length 改变，hidden/层数/正则保持一致，参数量不因 sequence length 改变。
- 所有预处理仍仅在训练折拟合。主要差值方向统一为 5 min minus 60 min。
- ΔAUROC、ΔAUPRC、ΔBrier、Δlog loss 用同患者 paired bootstrap 2,000 次给 95% CI；15 min 只用于梯度证据。
- 信息层结果沿用已冻结的 event miss、burden error、extreme attenuation、SD retention，不重新选择阈值。

## 7. 性能、图表与裁决

- 主要指标：AUROC、AUPRC、Brier、log loss、calibration intercept/slope。
- 图 1–5 与表 1–3 均由 machine-readable 数据复算；绘图、预览和导出严格使用 R。
- Model 3 始终是 primary temporal model，不因内部 AUROC 排名改变。
- 全部预设步骤与独立 QC 完成后，综合 external discrimination、calibration、不确定性、sparse-phase 稳定性与 paired resolution gain，在四个冻结 verdict token 中选择一个；不以单一任意 AUROC 阈值机械裁决，不增加模型或改结局救题。

## 8. 隐私与可重复性

- 原始数据库 ID 与 person-level QC 仅保留在 internal 目录。
- 对外交付预测使用排序后匿名 analysis key，不含原始 ID、时间戳或可逆映射。
- 环境、脚本状态、哈希、锁模先后顺序和至少 30 条独立断言全部记录；任一关键计数或时间泄漏失败即 `STOP_AND_DEBUG`。
