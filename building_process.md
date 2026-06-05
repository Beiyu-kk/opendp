# DP 项目搭建流程

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
mkdir -p docs
```
