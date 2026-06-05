# Diffusion Policy项目重构

## 1.DP 项目结构规划

本项目采用较为清晰的工程化结构组织 Diffusion Policy 相关代码。整体设计目标是将模型结构、扩散过程、策略封装、数据处理、环境交互、训练流程和评估流程进行解耦，使各个模块职责明确，便于后续维护、调试和扩展。

DP项目整体框架规划如下：

```text
opendp/
├── configs/
│   ├── train.yaml
│   └── task_book_grasp.yaml
├── scripts/
│   ├── train.py
│   ├── eval.py
│   └── collect_data.py
├── src/
│   └── opendp/
│       ├── models/
│       │   ├── unet1d.py
│       │   ├── transformer.py
│       │   ├── vision_encoder.py
|       |   ├── time_embedding.py
|       |   └── condition_encoder.py
│       ├── diffusion/
│       │   ├── noise_scheduler.py
│       │   ├── ddpm.py
│       │   ├── ddim.py
│       │   └── sampler.py
│       ├── policies/
│       │   └── diffusion_policy.py
│       ├── datasets/
│       │   └── robot_dataset.py
│       ├── envs/
│       │   ├── base_env.py
│       │   ├── sim_env.py
│       │   ├── real_robot_env.py
│       │   ├── camera.py
│       │   └── rollout.py
│       └── utils/
├── data/
├── outputs/
├── checkpoints/
└── pyproject.toml
```

其中，src/opendp/ 是项目的核心代码目录，所有与 Diffusion Policy 算法直接相关的功能都放在该目录下。根目录下的 configs/ 用于管理实验配置，scripts/ 用于提供训练、评估和数据采集等运行入口，data/ 用于存放示教数据，outputs/ 用于保存实验输出结果，checkpoints/ 用于保存模型权重。

各个文件的作用如下:

```text
models/      定义可训练的网络本体
diffusion/   定义加噪、去噪、采样
policies/    把模型包装成机器人策略
datasets/    读取和处理示教数据
envs/        连接仿真或真实机器人
training/    组织训练流程
evaluation/  测试策略效果
utils/       放通用工具
```

## 2. DP 项目结构详细说明

本项目的核心代码放在 `src/opendp/` 目录下。该目录主要负责实现 Diffusion Policy 的模型结构、扩散过程、策略封装、数据读取和环境交互。

### 2.1 `models/`：模型结构

`models/` 用于定义 Diffusion Policy 中使用的神经网络结构。

* `unet1d.py`：定义 1D UNet，用于对动作序列进行去噪。
* `transformer.py`：定义 Transformer 结构，可作为另一种动作去噪网络。
* `vision_encoder.py`：定义视觉编码器，用于提取图像特征。
* `time_embedding.py`：定义扩散步数 `t` 的时间编码。
* `condition_encoder.py`：编码条件信息，例如图像特征、机器人状态或历史观测。

简单来说，`models/` 负责定义“模型长什么样”。

---

### 2.2 `diffusion/`：扩散过程

`diffusion/` 用于实现 Diffusion Policy 的加噪、去噪和采样过程。

* `noise_scheduler.py`：定义噪声调度器，管理 `beta_t`、`alpha_t` 等扩散参数。
* `ddpm.py`：实现 DDPM 的核心训练逻辑，包括加噪和 loss 计算。
* `sampler.py`：实现推理时的反向去噪采样过程，从随机噪声生成动作序列。

简单来说，`diffusion/` 负责“动作序列是如何通过扩散模型生成出来的”。

---

### 2.3 `policies/`：策略封装

`policies/` 用于将扩散模型封装成机器人可以直接调用的策略。

* `diffusion_policy.py`：接收当前观测，调用模型和采样器生成未来一段动作序列，并输出机器人当前需要执行的动作。

简单来说，`policies/` 负责把“模型生成的动作序列”变成“机器人实际执行的动作”。

---

### 2.4 `datasets/`：数据读取

`datasets/` 用于读取和处理机器人示教数据。

* `robot_dataset.py`：读取 demonstration 数据，并根据 observation horizon 和 action horizon 构造训练样本。

它主要负责：

* 读取图像、机器人状态和动作。
* 构造历史观测序列。
* 构造未来动作序列标签。
* 对图像和动作进行预处理与归一化。

简单来说，`datasets/` 负责把“原始示教数据”变成“模型训练需要的 batch”。

---

### 2.5 `envs/`：环境交互

`envs/` 用于封装仿真环境或真实机器人环境。

* `base_env.py`：定义统一的环境接口，例如 `reset()`、`step()`、`get_obs()`。
* `sim_env.py`：封装仿真环境，用于在模拟器中测试策略。
* `real_robot_env.py`：封装真实机器人接口，用于在真实机械臂上执行动作。

简单来说，`envs/` 负责连接算法和外部环境，让 policy 可以在仿真或真机中运行。

---

### 2.6 `utils/`：通用工具

`utils/` 用于存放项目中常用的辅助函数，例如：

* 随机种子设置。
* 配置读取。
* 日志记录。
* 模型保存与加载。
* tensor 和 numpy 转换。
* 图像可视化。
* 路径管理。

简单来说，`utils/` 负责存放多个模块都会用到的工具代码。

---
