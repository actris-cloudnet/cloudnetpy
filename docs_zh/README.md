# CloudnetPy 中文文档

这是 CloudnetPy 的中文翻译文档。

## 构建文档

### 安装依赖

首先安装文档构建所需的依赖：

```bash
pip install -r requirements.txt
```

### 构建 HTML 文档

在 `docs_zh` 目录中运行：

```bash
# Linux/macOS
make html

# Windows
sphinx-build -M html source .
```

文档将生成在 `docs_zh/html` 目录中。

### 查看文档

构建完成后，在浏览器中打开 `docs_zh/html/index.html` 查看中文文档。

## 文档结构

```
docs_zh/
├── source/              # 文档源文件
│   ├── index.rst       # 主页
│   ├── overview.rst    # 概述
│   ├── installation.rst # 安装说明
│   ├── quickstart.rst  # 快速入门
│   ├── api.rst         # API 参考
│   ├── fileformat.rst  # 文件格式
│   ├── conf.py         # Sphinx 配置文件
│   ├── _static/        # 静态文件（CSS、图片等）
│   ├── _templates/     # 模板文件
│   └── model-evaluation/  # 模式评估文档
│       └── overview.rst
├── Makefile            # 构建脚本
└── requirements.txt    # Python 依赖
```

## 注意事项

- 本文档翻译自英文版本，保持了与原文档相同的结构
- 代码示例保持英文，确保代码可以正常运行
- 图片和图表引用保持不变，与原文档共享资源
- API 文档自动从源代码生成，保持与英文版本同步
