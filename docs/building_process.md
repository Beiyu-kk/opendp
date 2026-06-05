# DP 项目搭建流程

## 1.框架搭建

1.安装uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2.使用uv创建src layout项目框架

```bash
uv init opendp --package
cd opendp
```

3.补充项目框架

```bash
mkdir -p src/opendp/{models,diffusion,policies,datasets,envs,utils}
mkdir -p configs scripts data outputs checkpoints
```

## git保存该项目

创建新的github仓库对项目进行保存

```bash
git init
git add .
git commit -m "create a new dp project"
git branch -M main
git remote add origin git@github.com:Beiyu-kk/opendp.git
git push -u origin main
```

## 2.codex编写具体代码

接下来，使用codex进行每个脚本的创建和具体的代码编写，提示词参考[agent_prompt.md](./agent_prompt/1.create_project.md)，初始目标如下：

```
先实现一个状态输入版 Diffusion Policy，暂时不实现图像输入和真实机器人控制。

项目应当完成以下流程：

读取状态-动作数据
        ↓
构造 observation history 和 future action sequence
        ↓
对 action sequence 加噪
        ↓
模型预测噪声
        ↓
计算 DDPM loss 并训练
        ↓
保存 checkpoint
        ↓
加载 checkpoint
        ↓
通过 DDPM sampler 生成动作序列
        ↓
在 toy sim 环境中 rollout

当前阶段的目标是**跑通训练和评估闭环**，不是实现完整论文级 Diffusion Policy。
```
