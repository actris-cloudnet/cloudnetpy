# 静态资源说明

本目录包含文档使用的静态资源文件（CSS、图片等）。

## 图片资源

大部分图片资源位于英文文档目录 `../../../docs/source/_static/` 中。

为了避免重复，你可以：

### 方案一：复制图片（推荐用于独立部署）

将所需图片从英文文档复制到此目录：

**Windows:**
```cmd
copy ..\..\docs\source\_static\*.png .
copy ..\..\docs\source\_static\*.jpg .
copy ..\..\docs\source\_static\*.jpeg .
```

**Linux/macOS:**
```bash
cp ../../docs/source/_static/*.png .
cp ../../docs/source/_static/*.jpg .
cp ../../docs/source/_static/*.jpeg .
```

### 方案二：使用符号链接（推荐用于开发环境）

创建符号链接指向英文文档的静态资源：

**Windows (需要管理员权限):**
```cmd
mklink /D images ..\..\docs\source\_static
```

**Linux/macOS:**
```bash
ln -s ../../../docs/source/_static images
```

然后在 `.rst` 文件中引用图片时使用：`_static/images/example_data.png`

### 方案三：修改配置文件

在 `conf.py` 中添加额外的静态路径：

```python
html_static_path = ['_static', '../../../docs/source/_static']
```

## 当前文件

- `custom.css` - 自定义样式表，用于调整文档布局

## 需要的图片

文档中引用的图片：
- example_data.png - 概述页面
- CLU_workflow.png - 概述页面
- quickstart_radar.png - 快速入门页面
- quickstart_lidar.png - 快速入门页面
- quickstart_mwr.png - 快速入门页面
- quickstart_model.png - 快速入门页面
- quickstart_classification.png - 快速入门页面
- 20190517_mace-head_classification.png - 模式评估页面
- 20190517_mace-head_iwc-Z-T-method.png - 模式评估页面
- 20190517_mace-head_lwc-scaled-adiabatic.png - 模式评估页面
- 20190517_mace-head_cf_ecmwf_group.png - 模式评估页面
- 20190517_mace-head_iwc_ecmwf_group.png - 模式评估页面
- 20190517_mace-head_lwc_ecmwf_group.png - 模式评估页面
- L3_process.png - 模式评估页面
