========
概述
========

Cloudnet 处理方案
--------------------------

CloudnetPy (`Tukiainen 2020`_) 是一个实现了所谓 Cloudnet 处理方案的 Python 包
(`Illingworth 2007`_)。Cloudnet 处理从地基遥感测量生成云属性的垂直剖面。
将云雷达、光学激光雷达、微波辐射计和热力学（模式或探空）数据结合起来，
精确表征高达 15 公里的云层，具有高时间和垂直分辨率。

.. figure:: _static/example_data.png
	   :width: 500 px
	   :align: center

           Cloudnet 处理中使用的输入数据示例：雷达反射率因子（顶部）、平均
           多普勒速度（第二）、激光雷达后向散射系数（第三）、
           以及来自微波辐射计的液态水路径（底部）。

测量和模式数据被整合到共同的网格中，并分类为冰、液态、气溶胶、昆虫等。
然后，在进一步的处理步骤中可以检索冰水含量等地球物理产品。更详细的描述
可以在 `Illingworth 2007`_ 及其参考文献中找到。

.. note::

    近实时 Cloudnet 数据可以在 https://cloudnet.fmi.fi 访问。

需求说明
-----------------

在未来几年，Cloudnet 将成为 `ACTRIS`_（气溶胶、云和痕量气体研究基础设施）
的关键组件之一，其中 Cloudnet 框架将用于近实时处理每天数千兆字节的云遥感数据。
ACTRIS 研究基础设施目前处于实施阶段，计划在 2025 年全面运行。

为了满足 ACTRIS 的要求，需要一个强大的、开源的软件，能够可靠地处理大量数据。
CloudnetPy 软件包旨在执行 ACTRIS 云遥感的业务处理，从欧洲约 15 个测量站点
提供质量控制的数据产品。与原始专有 Cloudnet 软件不同，CloudnetPy 是开源的，
包括测试、文档和用户友好的 API，研究社区可以使用这些来进一步开发现有方法
并创建新产品。

.. _Tukiainen 2020: https://doi.org/10.21105/joss.02123
.. _Illingworth 2007: https://journals.ametsoc.org/doi/abs/10.1175/BAMS-88-6-883
.. _ACTRIS: http://actris.eu/

.. important::

   CloudnetPy 是原始 Matlab 处理原型代码的 Python 重写实现。
   CloudnetPy 具有多项改进的方法和错误修复、开源代码库、
   netCDF4 文件格式和详尽的文档。

业务 Cloudnet 处理的核心是 CloudnetPy，但还包括校准数据库
和全面的质量控制/保证程序：

.. figure:: _static/CLU_workflow.png
	   :width: 650 px
	   :align: center

           ACTRIS 中业务 Cloudnet 处理的工作流程。


另见：

- Cloudnet 数据门户：https://cloudnet.fmi.fi/
- CloudnetPy 源代码：https://github.com/actris-cloudnet/cloudnetpy
- ACTRIS 主页：http://actris.eu/
- ACTRIS 数据门户：http://actris.nilu.no/
