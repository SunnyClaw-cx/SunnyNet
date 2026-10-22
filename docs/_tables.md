### 基线对照

| 结果文件 | 架构 | 预设 | 总参数 | 计算栈参数 | 步数 | train loss | val loss | ppl | ms/token | token/s | 峰值MB | 梯度范数 | 访存探针 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline_sunnynet | sunnynet | small | 1,413,908 | 365,332 | 1,200 | 4.1458 | 4.3367 | 76.4562 | 0.2831 | 3,531.9600 | 896.0000 | 0.8445 | page_faults_per_sec |
| baseline_transformer | transformer | small | 919,008 | 361,952 | 1,200 | 4.0950 | 4.2895 | 72.9265 | 0.3089 | 3,237.1500 | 878.9000 | 0.4838 | page_faults_per_sec |

### 消融

| 结果文件 | 架构 | 预设 | 总参数 | 计算栈参数 | 步数 | train loss | val loss | ppl | ms/token | token/s | 峰值MB | 梯度范数 | 访存探针 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ablation_full | sunnynet | small | 1,413,908 | 365,332 | 600 | 4.2143 | 4.5681 | 96.3595 | 0.4528 | 2,208.5900 | 876.8000 | 0.8752 | page_faults_per_sec |
| ablation_no_circulant | sunnynet | small | 1,413,896 | 365,320 | 600 | 4.2217 | 4.5663 | 96.1879 | 0.3259 | 3,068.0700 | 861.6000 | 0.8503 | page_faults_per_sec |
| ablation_no_current | sunnynet | small | 1,413,908 | 365,332 | 600 | 4.2108 | 4.5531 | 94.9228 | 0.4878 | 2,049.8800 | 858.3000 | 0.9127 | page_faults_per_sec |
| ablation_no_decay_gate | sunnynet | small | 1,413,908 | 365,332 | 600 | 4.2022 | 4.5539 | 95.0053 | 0.3231 | 3,095.2200 | 845.0000 | 0.8210 | page_faults_per_sec |
| ablation_no_local_taps | sunnynet | small | 1,411,860 | 363,284 | 600 | 4.2055 | 4.5722 | 96.7609 | 0.2605 | 3,838.4000 | 894.8000 | 0.7957 | page_faults_per_sec |
| ablation_none | sunnynet | small | 1,411,848 | 363,272 | 600 | 4.2523 | 4.6058 | 100.0649 | 0.2063 | 4,846.8200 | 804.8000 | 0.7870 | page_faults_per_sec |

### 种子稳定性

| 结果文件 | 架构 | 预设 | 总参数 | 计算栈参数 | 步数 | train loss | val loss | ppl | ms/token | token/s | 峰值MB | 梯度范数 | 访存探针 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| seed_1234 | sunnynet | small | 1,413,908 | 365,332 | 600 | 4.2143 | 4.5681 | 96.3595 | 0.2726 | 3,668.2100 | 900.7000 | 0.8752 | page_faults_per_sec |
| seed_2024 | sunnynet | small | 1,413,908 | 365,332 | 600 | 3.4746 | 4.5984 | 99.3242 | 0.2081 | 4,805.1100 | 890.7000 | 0.8684 | page_faults_per_sec |
| seed_3407 | sunnynet | small | 1,413,908 | 365,332 | 600 | 5.1415 | 4.6316 | 102.6768 | 0.1421 | 7,035.3600 | 892.2000 | 0.9188 | page_faults_per_sec |

### 缩放

| 结果文件 | 架构 | 预设 | 总参数 | 计算栈参数 | 步数 | train loss | val loss | ppl | ms/token | token/s | 峰值MB | 梯度范数 | 访存探针 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| scale_proto20m_sunnynet | sunnynet | proto20m | 20,955,698 | 16,761,394 | 300 | 4.7959 | 5.4883 | 241.8372 | 3.1642 | 316.0400 | 2,579.0000 | 1.8458 | page_faults_per_sec |
| scale_proto20m_transformer | transformer | proto20m | 21,288,240 | 16,762,160 | 300 | 4.2403 | 5.0955 | 163.2907 | 2.2805 | 438.4900 | 1,689.2000 | 1.3845 | page_faults_per_sec |

### 长序列训练

| 结果文件 | 架构 | 预设 | 总参数 | 计算栈参数 | 步数 | train loss | val loss | ppl | ms/token | token/s | 峰值MB | 梯度范数 | 访存探针 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| longseq_sunnynet_T1024 | sunnynet | small | 1,413,908 | 365,332 | 200 | 5.7437 | 5.1320 | 169.3472 | 0.2623 | 3,811.8200 | 1,269.4000 | 1.0209 | page_faults_per_sec |
| longseq_transformer_T1024 | transformer | small | 951,776 | 361,952 | 200 | 5.8290 | 5.3702 | 214.9164 | 0.5445 | 1,836.6800 | 2,008.3000 | 0.9143 | page_faults_per_sec |

### 学习率标定

| 结果文件 | 架构 | 预设 | 总参数 | 计算栈参数 | 步数 | train loss | val loss | ppl | ms/token | token/s | 峰值MB | 梯度范数 | 访存探针 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tune_sunnynet_0.001 | sunnynet | small | 1,413,908 | 365,332 | 260 | 5.6017 | 4.8858 | 132.3903 | 0.1859 | 5,380.0800 | 885.7000 | 1.3116 | page_faults_per_sec |
| tune_sunnynet_0.003 | sunnynet | small | 1,413,908 | 365,332 | 260 | 5.5429 | 4.7720 | 118.1564 | 0.2225 | 4,494.3900 | 890.9000 | 0.9899 | page_faults_per_sec |
| tune_sunnynet_0.006 | sunnynet | small | 1,413,908 | 365,332 | 260 | 5.5548 | 4.7822 | 119.3692 | 0.3196 | 3,129.3000 | 888.8000 | 1.0126 | page_faults_per_sec |
| tune_transformer_0.001 | transformer | small | 919,008 | 361,952 | 260 | 5.7266 | 5.0614 | 157.8070 | 0.3264 | 3,063.5700 | 884.6000 | 1.0341 | page_faults_per_sec |
| tune_transformer_0.003 | transformer | small | 919,008 | 361,952 | 260 | 5.5421 | 4.7989 | 121.3795 | 0.3134 | 3,190.7600 | 888.2000 | 0.7394 | page_faults_per_sec |
| tune_transformer_0.006 | transformer | small | 919,008 | 361,952 | 260 | 5.5219 | 4.7744 | 118.4380 | 0.2553 | 3,916.8600 | 889.0000 | 0.6246 | page_faults_per_sec |

---

### Baseline comparison

| Result file | Architecture | Preset | Total params | Compute-stack params | Steps | train loss | val loss | ppl | ms/token | token/s | Peak MB | Gradient norm | Memory probe |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline_sunnynet | sunnynet | small | 1,413,908 | 365,332 | 1,200 | 4.1458 | 4.3367 | 76.4562 | 0.2831 | 3,531.9600 | 896.0000 | 0.8445 | page_faults_per_sec |
| baseline_transformer | transformer | small | 919,008 | 361,952 | 1,200 | 4.0950 | 4.2895 | 72.9265 | 0.3089 | 3,237.1500 | 878.9000 | 0.4838 | page_faults_per_sec |

### Ablation

| Result file | Architecture | Preset | Total params | Compute-stack params | Steps | train loss | val loss | ppl | ms/token | token/s | Peak MB | Gradient norm | Memory probe |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ablation_full | sunnynet | small | 1,413,908 | 365,332 | 600 | 4.2143 | 4.5681 | 96.3595 | 0.4528 | 2,208.5900 | 876.8000 | 0.8752 | page_faults_per_sec |
| ablation_no_circulant | sunnynet | small | 1,413,896 | 365,320 | 600 | 4.2217 | 4.5663 | 96.1879 | 0.3259 | 3,068.0700 | 861.6000 | 0.8503 | page_faults_per_sec |
| ablation_no_current | sunnynet | small | 1,413,908 | 365,332 | 600 | 4.2108 | 4.5531 | 94.9228 | 0.4878 | 2,049.8800 | 858.3000 | 0.9127 | page_faults_per_sec |
| ablation_no_decay_gate | sunnynet | small | 1,413,908 | 365,332 | 600 | 4.2022 | 4.5539 | 95.0053 | 0.3231 | 3,095.2200 | 845.0000 | 0.8210 | page_faults_per_sec |
| ablation_no_local_taps | sunnynet | small | 1,411,860 | 363,284 | 600 | 4.2055 | 4.5722 | 96.7609 | 0.2605 | 3,838.4000 | 894.8000 | 0.7957 | page_faults_per_sec |
| ablation_none | sunnynet | small | 1,411,848 | 363,272 | 600 | 4.2523 | 4.6058 | 100.0649 | 0.2063 | 4,846.8200 | 804.8000 | 0.7870 | page_faults_per_sec |

### Seed stability

| Result file | Architecture | Preset | Total params | Compute-stack params | Steps | train loss | val loss | ppl | ms/token | token/s | Peak MB | Gradient norm | Memory probe |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| seed_1234 | sunnynet | small | 1,413,908 | 365,332 | 600 | 4.2143 | 4.5681 | 96.3595 | 0.2726 | 3,668.2100 | 900.7000 | 0.8752 | page_faults_per_sec |
| seed_2024 | sunnynet | small | 1,413,908 | 365,332 | 600 | 3.4746 | 4.5984 | 99.3242 | 0.2081 | 4,805.1100 | 890.7000 | 0.8684 | page_faults_per_sec |
| seed_3407 | sunnynet | small | 1,413,908 | 365,332 | 600 | 5.1415 | 4.6316 | 102.6768 | 0.1421 | 7,035.3600 | 892.2000 | 0.9188 | page_faults_per_sec |

### Scaling

| Result file | Architecture | Preset | Total params | Compute-stack params | Steps | train loss | val loss | ppl | ms/token | token/s | Peak MB | Gradient norm | Memory probe |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| scale_proto20m_sunnynet | sunnynet | proto20m | 20,955,698 | 16,761,394 | 300 | 4.7959 | 5.4883 | 241.8372 | 3.1642 | 316.0400 | 2,579.0000 | 1.8458 | page_faults_per_sec |
| scale_proto20m_transformer | transformer | proto20m | 21,288,240 | 16,762,160 | 300 | 4.2403 | 5.0955 | 163.2907 | 2.2805 | 438.4900 | 1,689.2000 | 1.3845 | page_faults_per_sec |

### Long-sequence training

| Result file | Architecture | Preset | Total params | Compute-stack params | Steps | train loss | val loss | ppl | ms/token | token/s | Peak MB | Gradient norm | Memory probe |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| longseq_sunnynet_T1024 | sunnynet | small | 1,413,908 | 365,332 | 200 | 5.7437 | 5.1320 | 169.3472 | 0.2623 | 3,811.8200 | 1,269.4000 | 1.0209 | page_faults_per_sec |
| longseq_transformer_T1024 | transformer | small | 951,776 | 361,952 | 200 | 5.8290 | 5.3702 | 214.9164 | 0.5445 | 1,836.6800 | 2,008.3000 | 0.9143 | page_faults_per_sec |

### Learning-rate calibration

| Result file | Architecture | Preset | Total params | Compute-stack params | Steps | train loss | val loss | ppl | ms/token | token/s | Peak MB | Gradient norm | Memory probe |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tune_sunnynet_0.001 | sunnynet | small | 1,413,908 | 365,332 | 260 | 5.6017 | 4.8858 | 132.3903 | 0.1859 | 5,380.0800 | 885.7000 | 1.3116 | page_faults_per_sec |
| tune_sunnynet_0.003 | sunnynet | small | 1,413,908 | 365,332 | 260 | 5.5429 | 4.7720 | 118.1564 | 0.2225 | 4,494.3900 | 890.9000 | 0.9899 | page_faults_per_sec |
| tune_sunnynet_0.006 | sunnynet | small | 1,413,908 | 365,332 | 260 | 5.5548 | 4.7822 | 119.3692 | 0.3196 | 3,129.3000 | 888.8000 | 1.0126 | page_faults_per_sec |
| tune_transformer_0.001 | transformer | small | 919,008 | 361,952 | 260 | 5.7266 | 5.0614 | 157.8070 | 0.3264 | 3,063.5700 | 884.6000 | 1.0341 | page_faults_per_sec |
| tune_transformer_0.003 | transformer | small | 919,008 | 361,952 | 260 | 5.5421 | 4.7989 | 121.3795 | 0.3134 | 3,190.7600 | 888.2000 | 0.7394 | page_faults_per_sec |
| tune_transformer_0.006 | transformer | small | 919,008 | 361,952 | 260 | 5.5219 | 4.7744 | 118.4380 | 0.2553 | 3,916.8600 | 889.0000 | 0.6246 | page_faults_per_sec |
