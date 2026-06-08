# opendp

基于 `uv` 的极简状态扩散策略（Diffusion Policy）项目。

## 运行

```bash
uv run python scripts/collect_data.py
uv run python scripts/train.py
uv run python scripts/eval.py
```

默认配置在一个玩具级 2D 到达任务上进行训练：

- 数据集：`data/book_grasp_state.npz`
- 模型检查点：`checkpoints/state_diffusion_policy.pt`
- 推理结果：`outputs/eval_rollout.json`

生成的数据集、检查点和推理结果均被 git 忽略。
