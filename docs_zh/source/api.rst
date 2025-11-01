API 参考
=============

.. note::

   本文档中的 API 参考部分由 Sphinx 从源代码自动生成，部分技术描述保留英文以确保准确性。
   对于主要函数，我们在下方提供了详细的中文说明和使用示例。


高级函数
--------------------

CloudnetPy 的高级函数提供了一种简单的机制来处理云遥感测量并生成 Cloudnet 产品。
完整的处理分步进行。每一步都会生成一个文件，该文件用作下一步的输入。

原始数据转换
...................

不同的 Cloudnet 仪器以各种格式（netCDF、二进制、文本）提供原始数据，
首先需要将其转换为包含统一单位和其他元数据的同质 Cloudnet netCDF 文件。
这一初始处理步骤对于确保后续处理步骤适用于所有支持的仪器组合是必要的。

雷达数据处理
~~~~~~~~~~~~

**METEK MIRA 云雷达数据转换**

.. autofunction:: instruments.mira2nc

**中文说明：**

将 METEK MIRA 系列云雷达的原始数据文件转换为标准的 Cloudnet netCDF 格式。

- **输入格式**：MIRA 雷达原始数据文件（通常为 .mmclx 或 .mmclx.gz 格式）
- **输出格式**：Cloudnet Level 1b netCDF 文件
- **主要功能**：提取雷达反射率、多普勒速度等参数，添加必要的元数据

**使用示例：**

.. code-block:: python

   from cloudnetpy.instruments import mira2nc
   
   # 基本用法
   uuid = mira2nc('20230729_0000.mmclx', 'radar.nc', {'name': 'Munich'})
   
   # 包含更多站点信息
   site_meta = {
       'name': 'Munich',
       'latitude': 48.148,
       'longitude': 11.573,
       'altitude': 538
   }
   uuid = mira2nc('radar_raw.mmclx', 'radar_processed.nc', site_meta)

----

**RPG 云雷达数据转换**

.. autofunction:: instruments.rpg2nc

**中文说明：**

处理 RPG (Radiometer Physics GmbH) 云雷达数据，支持 94 GHz FMCW 雷达。

- **输入格式**：RPG 雷达 netCDF 文件
- **特点**：RPG 雷达通常集成了微波辐射计通道，可同时提供液态水路径数据

----

**BASTA 雷达数据转换**

.. autofunction:: instruments.basta2nc

**中文说明：**

转换法国 BASTA (Bistatic rAdar SysTem for Atmospheric studies) 雷达数据。

----

.. autofunction:: instruments.galileo2nc

.. autofunction:: instruments.copernicus2nc

激光雷达/云高仪数据处理
~~~~~~~~~~~~~~~~~~~~~~~

**云高仪数据转换**

.. autofunction:: instruments.ceilo2nc

**中文说明：**

处理各种品牌的云高仪（激光雷达）数据，包括：

- Lufft CHM 15k
- Vaisala CL31, CL51, CL61
- Jenoptik CHM 15k

云高仪是激光雷达的一种，专门用于测量云底高度和大气后向散射特性。

**使用示例：**

.. code-block:: python

   from cloudnetpy.instruments import ceilo2nc
   
   # 转换云高仪数据
   site_meta = {
       'name': 'Munich',
       'altitude': 538  # 海拔高度（米）
   }
   uuid = ceilo2nc('CHM15kxLMU_20230729.nc', 'lidar.nc', site_meta)

----

**PollyXT 激光雷达数据转换**

.. autofunction:: instruments.pollyxt2nc

**中文说明：**

处理 PollyXT 多波长拉曼偏振激光雷达数据，这是一种先进的大气探测激光雷达系统。

----

微波辐射计数据处理
~~~~~~~~~~~~~~~~~~~

**HATPRO 微波辐射计数据转换**

.. autofunction:: instruments.hatpro2nc

**中文说明：**

处理 RPG-HATPRO (Humidity And Temperature PROfiler) 微波辐射计数据。

- **输入格式**：HATPRO 二进制文件（.LWP, .IWV 等）
- **输出变量**：液态水路径（LWP）、积分水汽（IWV）等
- **重要参数**：需要指定日期，因为多个文件可能对应一天

**使用示例：**

.. code-block:: python

   from cloudnetpy.instruments import hatpro2nc
   
   # 处理某一天的所有 HATPRO 文件
   site_meta = {'name': 'Munich', 'altitude': 538}
   uuid, valid_files = hatpro2nc(
       '.',  # 文件所在目录
       'mwr.nc',  # 输出文件
       site_meta,
       date='2023-07-29'  # 指定日期
   )
   print(f"使用了 {len(valid_files)} 个文件")

----

**Radiometrics 微波辐射计数据转换**

.. autofunction:: instruments.radiometrics2nc

**中文说明：**

处理 Radiometrics 公司的微波辐射计数据。

----

雨滴谱仪和气象站数据处理
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: instruments.parsivel2nc

.. autofunction:: instruments.thies2nc

.. autofunction:: instruments.ws2nc

**中文说明：**

- ``parsivel2nc``：处理 OTT Parsivel² 雨滴谱仪数据
- ``thies2nc``：处理 Thies 激光雨滴谱仪数据  
- ``ws2nc``：处理气象站数据（温度、湿度、气压等）


分类文件生成
...................

分类文件是 Cloudnet 处理的核心步骤，它将所有输入数据（雷达、激光雷达、模式、微波辐射计）
整合到统一的时间-高度网格上，并对每个网格点进行初步分类。

**生成分类文件**

.. autofunction:: categorize.generate_categorize

**中文说明：**

这是 Cloudnet 处理的关键步骤，将多源观测数据融合并分类。

**必需输入：**
- 雷达数据（radar）
- 大气模式数据（model）

**可选输入：**
- 激光雷达数据（lidar）- 强烈推荐，用于改进低层云和气溶胶识别
- 微波辐射计数据（mwr）- 用于液态水路径信息

**使用示例：**

.. code-block:: python

   from cloudnetpy.categorize import generate_categorize
   
   # 基本用法（仅雷达和模式）
   input_files = {
       'radar': 'radar.nc',
       'model': 'ecmwf_model.nc'
   }
   uuid = generate_categorize(input_files, 'categorize.nc')
   
   # 完整用法（包含所有仪器）
   input_files = {
       'radar': 'radar.nc',
       'lidar': 'lidar.nc',
       'model': 'ecmwf_model.nc',
       'mwr': 'mwr.nc'
   }
   uuid = generate_categorize(input_files, 'categorize.nc')

**输出内容：**

分类文件包含：
- 统一网格上的所有观测数据
- 分类比特位（云相态、降水、昆虫等）
- 质量控制标记
- 大气参数（温度、压力、湿度）


产品生成
..................

从分类文件可以生成多种地球物理产品，每个产品聚焦于特定的云属性。

**分类产品**

.. autofunction:: products.generate_classification

**中文说明：**

生成目标分类产品，这是最简单的 Cloudnet 产品。

**产品内容：**
- ``target_classification``：目标分类（云滴、冰、融化层、昆虫、气溶胶等）
- ``detection_status``：探测状态
- ``quality_bits``：质量标记

**使用示例：**

.. code-block:: python

   from cloudnetpy.products import generate_classification
   
   uuid = generate_classification('categorize.nc', 'classification.nc')

----

**冰水含量产品**

.. autofunction:: products.generate_iwc

**中文说明：**

检索大气冰水含量（Ice Water Content, IWC）。

**方法：**
- 使用雷达反射率和温度的经验关系
- 适用于冰云区域

**产品变量：**
- ``iwc``：冰水含量 (kg m⁻³)
- ``iwc_error``：估计误差
- ``iwc_retrieval_status``：检索状态

**使用示例：**

.. code-block:: python

   from cloudnetpy.products import generate_iwc
   
   uuid = generate_iwc('categorize.nc', 'iwc.nc')

----

**液态水含量产品**

.. autofunction:: products.generate_lwc

**中文说明：**

检索液态水含量（Liquid Water Content, LWC）。

**方法：**
- 绝热法：基于云绝热假设
- 缩放法：使用微波辐射计液态水路径进行缩放

**产品变量：**
- ``lwc``：液态水含量 (kg m⁻³)
- ``lwc_error``：估计误差

**使用示例：**

.. code-block:: python

   from cloudnetpy.products import generate_lwc
   
   uuid = generate_lwc('categorize.nc', 'lwc.nc')

----

**毛毛雨产品**

.. autofunction:: products.generate_drizzle

**中文说明：**

检索毛毛雨（drizzle）参数，包括粒子大小和下落速度。

----

**液滴有效半径产品**

.. autofunction:: products.generate_der

**中文说明：**

检索云滴有效半径（Droplet Effective Radius, DER）。

----

**冰粒有效半径产品**

.. autofunction:: products.generate_ier

**中文说明：**

检索冰粒有效半径（Ice Effective Radius, IER）。


可视化结果
...................

CloudnetPy 提供易于使用的绘图接口，可以快速可视化各种产品。

**生成图表**

.. autofunction:: plotting.generate_figure

**中文说明：**

快速生成 Cloudnet 数据的可视化图表。

**功能特点：**
- 自动识别变量类型并选择合适的色标
- 支持同时绘制多个变量
- 可自定义图表参数
- 支持保存为图片文件

**使用示例：**

.. code-block:: python

   from cloudnetpy.plotting import generate_figure
   
   # 绘制单个变量
   generate_figure('radar.nc', ['Zh'])
   
   # 绘制多个变量
   generate_figure('categorize.nc', ['Z', 'v', 'width', 'ldr'])
   
   # 保存为文件
   generate_figure('classification.nc', 
                   ['target_classification'],
                   output_filename='classification.png',
                   show=False)
   
   # 自定义参数
   from cloudnetpy.plotting import PlotParameters
   params = PlotParameters(
       image_name='classification.png',
       show=False,
       dpi=150
   )
   generate_figure('classification.nc', 
                   ['target_classification'],
                   plot_params=params)

----

**绘图参数类**

.. autoclass:: plotting.PlotParameters

**中文说明：**

控制图表生成的参数类，包括：
- 图片尺寸、DPI
- 是否显示/保存
- 自定义标题等

----

.. autoclass:: plotting.PlotMeta

.. autoclass:: plotting.Dimensions


详细模块参考
======================

以下部分包含各个模块的详细 API 参考，主要面向开发者。
这些模块的文档从源代码自动生成，部分内容为英文。

分类模块
------------------

分类是 CloudnetPy 的子包。它包含创建 Cloudnet 分类文件时使用的多个模块。
这些是底层模块，通常不需要直接调用。

radar
.....

.. automodule:: categorize.radar
   :members:

lidar
.....

.. automodule:: categorize.lidar
   :members:

mwr
...

.. automodule:: categorize.mwr
   :members:

model
.....

.. automodule:: categorize.model
   :members:

classify
........

.. automodule:: categorize.classify
   :members:

melting
.......

.. automodule:: categorize.melting
   :members:

freezing
........

.. automodule:: categorize.freezing
   :members:


falling
.......

.. automodule:: categorize.falling
   :members:


insects
.......

.. automodule:: categorize.insects
   :members:


atmos
.....

.. automodule:: categorize.atmos
   :members:


droplet
.......

.. automodule:: categorize.droplet
   :members:


产品模块
----------------

产品是 CloudnetPy 的子包。它包含对应于不同 Cloudnet 产品的多个模块。

classification
..............

.. automodule:: products.classification
   :members:


iwc
...

.. automodule:: products.iwc
   :members:


lwc
...

.. automodule:: products.lwc
   :members:


drizzle
.......

.. automodule:: products.drizzle
   :members:


der
...

.. automodule:: products.der
   :members:


ier
...

.. automodule:: products.ier
   :members:


product_tools
.............

.. automodule:: products.product_tools
   :members:


工具模块
--------

具有低级功能的各种工具模块。这些模块提供文件操作、数据处理等辅助功能。

文件连接工具
............

concat_lib
~~~~~~~~~~

**中文说明：**

``concat_lib`` 模块提供连接（合并）netCDF 文件的功能，用于将多个时间段的数据文件合并成单个文件。

**主要功能：**

1. **truncate_netcdf_file** - 截断 netCDF 文件
2. **update_nc** - 追加数据到现有文件
3. **concatenate_files** - 连接多个 netCDF 文件
4. **concatenate_text_files** - 连接文本文件
5. **bundle_netcdf_files** - 将多个文件打包成日文件

**使用场景：**
- 将小时数据合并成日数据
- 合并多个站点或时间段的观测数据
- 创建测试用的小文件

**详细 API 参考：**

.. automodule:: concat_lib
   :members:

----

实用工具
~~~~~~~~

utils
"""""

**中文说明：**

``utils`` 模块包含各种实用工具函数，如数据转换、单位换算、数组操作等。

.. automodule:: utils
   :members:

----

cloudnetarray
"""""""""""""

**中文说明：**

``cloudnetarray`` 模块定义了 CloudnetPy 使用的数组类，包含额外的元数据和操作方法。

.. automodule:: cloudnetarray
   :members:

----

datasource
""""""""""

**中文说明：**

``datasource`` 模块处理数据源的读取和管理，支持多种输入格式。

.. automodule:: datasource
   :members:

----

output
""""""

**中文说明：**

``output`` 模块负责生成符合 Cloudnet 标准的 netCDF 输出文件，包括元数据、变量属性等。

.. automodule:: output
   :members:
