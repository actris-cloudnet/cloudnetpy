# noqa: INP001
import os
import sys

sys.path.insert(0, os.path.abspath("../.."))
sys.path.insert(0, os.path.abspath("../../cloudnetpy/"))

# Sphinx 文档构建器的配置文件。
#
# 有关内置配置值的完整列表，请参阅文档：
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- 项目信息 -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

# -- 项目信息 -----------------------------------------------------

project = "CloudnetPy"
copyright = "2022, 芬兰气象研究所"  # noqa: A001
author = "芬兰气象研究所"

# -- 通用配置 ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.imgmath",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.autosummary",
    "sphinx.ext.autosectionlabel",
]

templates_path = ["_templates"]
exclude_patterns = ["_temp/*"]
autodoc_member_order = "bysource"

# 设置语言为简体中文
language = "zh_CN"

# -- 国际化配置 -----------------------------------------------------
locale_dirs = ['locale/']   # gettext 翻译文件目录
gettext_compact = False     # 为每个文档创建单独的 pot 文件
gettext_uuid = False        # 不使用 UUID
gettext_location = False    # 不在 pot 文件中包含位置信息
gettext_auto_build = True   # 自动构建 mo 文件

# -- HTML 输出选项 -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
# 包含本地 _static 和英文文档的 _static（用于共享图片）
html_static_path = ["_static", "../../docs/source/_static"]

napoleon_google_docstring = True
