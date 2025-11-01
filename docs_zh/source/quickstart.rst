==========
快速入门
==========

在本教程中，使用 CloudnetPy 的高级 API 从原始数据创建产品。

原始数据转换步骤
-------------------

在生成 Cloudnet 产品之前，我们需要将原始仪器数据转换为 Cloudnet Level 1b netCDF 文件。
让我们从 `慕尼黑 <https://cloudnet.fmi.fi/site/munich>`_ 站点的以下原始文件开始：

- METEK MIRA 云雷达：`20230729_0000.mmclx.gz <https://cloudnet.fmi.fi/api/download/raw/62905a7c-4e18-474f-8532-bb59f39ca4ff/20230729_0000.mmclx.gz>`_（使用前请解压）
- Lufft CHM 15k 云高仪：`CHM15kxLMU_20230729.nc <https://cloudnet.fmi.fi/api/download/raw/7d1909f3-c73f-4de9-a771-e6795751e495/CHM15kxLMU_20230729.nc>`_
- RPG HATPRO 微波辐射计：`230729.LWP <https://cloudnet.fmi.fi/api/download/raw/490704b2-7533-4137-979b-f197a6c72e17/230729.LWP>`_
- ECMWF 模式：`20230729_munich_ecmwf.nc <https://cloudnet.fmi.fi/api/download/product/856b5a84-155b-427d-b914-09238c206c02/20230729_munich_ecmwf.nc>`_

在 Linux 和 macOS 上可以使用以下命令下载这些文件：

.. code-block:: console

    curl -O https://cloudnet.fmi.fi/api/download/raw/62905a7c-4e18-474f-8532-bb59f39ca4ff/20230729_0000.mmclx.gz
    gunzip 20230729_0000.mmclx.gz
    curl -O https://cloudnet.fmi.fi/api/download/raw/7d1909f3-c73f-4de9-a771-e6795751e495/CHM15kxLMU_20230729.nc
    curl -O https://cloudnet.fmi.fi/api/download/raw/490704b2-7533-4137-979b-f197a6c72e17/230729.LWP
    curl -O https://cloudnet.fmi.fi/api/download/product/856b5a84-155b-427d-b914-09238c206c02/20230729_munich_ecmwf.nc

可以使用 `Cloudnet 数据门户 API <https://docs.cloudnet.fmi.fi/api/data-portal.html#get-apiraw-files--upload>`_ 找到更多原始文件。

雷达处理
~~~~~~~~~~~~~~~~

在第一个示例中，我们将原始 METEK MIRA-36 云雷达文件转换为
可以在进一步处理步骤中使用的 Cloudnet netCDF 文件。

.. code-block:: python

    from cloudnetpy.instruments import mira2nc
    uuid = mira2nc('20230729_0000.mmclx', 'radar.nc', {'name': 'Munich'})

变量 ``uuid`` 包含生成的 ``radar.nc`` 文件的唯一标识符。
有关更多信息，请参阅此函数的 `API 参考 <api.html#instruments.mira2nc>`__。

您可以从新生成的文件中绘制雷达反射率因子等变量。

.. code-block:: python

    from cloudnetpy.plotting import generate_figure
    generate_figure('radar.nc', ['Zh'])

.. figure:: _static/quickstart_radar.png

激光雷达处理
~~~~~~~~~~~~~~~~

接下来，我们将原始 Lufft CHM 15k 云高仪（激光雷达）文件转换为 Cloudnet netCDF 文件，
并处理信噪比筛选的后向散射系数。稍后也需要这个转换后的激光雷达文件。

.. code-block:: python

    from cloudnetpy.instruments import ceilo2nc
    uuid = ceilo2nc('CHM15kxLMU_20230729.nc', 'lidar.nc', {'name': 'Munich', 'altitude': 538})

变量 ``uuid`` 包含生成的 ``lidar.nc`` 文件的唯一标识符。
有关更多信息，请参阅此函数的 `API 参考 <api.html#instruments.ceilo2nc>`__。

您可以从新生成的文件中绘制衰减后向散射系数等变量。

.. code-block:: python

    generate_figure('lidar.nc', ['beta'])

.. figure:: _static/quickstart_lidar.png


微波辐射计处理
~~~~~~~~~~~~~~

接下来，我们将 RPG-HATPRO 微波辐射计（MWR）二进制文件（例如 \*.LWP）转换为 Cloudnet
netCDF 文件以检索积分液态水路径（LWP）。

.. code-block:: python

    from cloudnetpy.instruments import hatpro2nc
    uuid, valid_files = hatpro2nc('.', 'mwr.nc', {'name': 'Munich', 'altitude': 538}, date='2023-07-29')

变量 ``uuid`` 包含生成的 ``mwr.nc`` 文件的唯一标识符，``valid_files`` 包含用于处理的文件。
有关更多信息，请参阅此函数的 `API 参考 <api.html#instruments.hatpro2nc>`__。

您可以从新生成的文件中绘制液态水路径等变量。

.. code-block:: python

    generate_figure('mwr.nc', ['lwp'])

.. figure:: _static/quickstart_mwr.png

.. note::

    如果您有 94 GHz RPG 云雷达，则不需要单独的 MWR 仪器。
    RPG 雷达包含一个提供 LWP 测量的单一 MWR 通道，可以在
    CloudnetPy 中使用。尽管如此，如果可能的话，始终建议为测量站点
    配备专用的多通道辐射计。


模式数据
~~~~~~~~~~

下一处理步骤所需的模式文件可以从
`Cloudnet 数据门户 API <https://docs.cloudnet.fmi.fi/api/data-portal.html#get-apimodel-files--modelfile>`_ 下载。
根据站点和日期，可能有多个可用的模式。
不同模式的列表可以在 `这里 <https://cloudnet.fmi.fi/api/models/>`_ 找到。

您可以从模式文件中绘制云分数等变量。

.. code-block:: python

    generate_figure('20230729_munich_ecmwf.nc', ['cloud_fraction'])

.. figure:: _static/quickstart_model.png


Cloudnet 产品生成
------------------

在处理完原始雷达、激光雷达和 MWR 文件，并获取模式文件后，
可以创建 Cloudnet 产品。

分类处理
~~~~~~~~~~~~~~~~~~~~~

在下一个示例中，我们从上面生成的 ``radar.nc``、``mwr.nc`` 和 ``lidar.nc``
文件开始创建一个分类文件。所需的 ``20230729_munich_ecmwf.nc`` 文件可以
从本页顶部下载。

.. code-block:: python

    from cloudnetpy.categorize import generate_categorize
    input_files = {
        'radar': 'radar.nc',
        'lidar': 'lidar.nc',
        'model': '20230729_munich_ecmwf.nc',
        'mwr': 'mwr.nc'
    }
    uuid = generate_categorize(input_files, 'categorize.nc')

变量 ``uuid`` 包含生成的 ``categorize.nc`` 文件的唯一标识符。
有关更多信息，请参阅此函数的 `API 参考 <api.html#categorize.generate_categorize>`__。
请注意，对于 94 GHz RPG 云雷达，``radar.nc`` 文件可以用作两个输入的输入：
``'radar'`` 和 ``'mwr'``。


生成分类产品
~~~~~~~~~~~~~~~~~~~~~

在最后一个示例中，我们创建最小且最简单的 Cloudnet
产品，即分类产品。生成产品的函数总是使用分类文件作为输入。

.. code-block:: python

    from cloudnetpy.products import generate_classification
    uuid = generate_classification('categorize.nc', 'classification.nc')

变量 ``uuid`` 包含生成的 ``classification.nc`` 文件的唯一标识符。
其他产品也有相应的函数（参见 :ref:`产品生成`）。

您可以从新生成的文件中绘制目标分类等变量。

.. code-block:: python

    generate_figure('classification.nc', ['target_classification'])

.. figure:: _static/quickstart_classification.png
