=========================
安装说明
=========================

CloudnetPy 可以安装在任何拥有 Python 3.10 或更高版本的计算机上。
实际的安装过程取决于操作系统。下面的说明适用于 Ubuntu 22.04。

Python 安装
-------------------

.. code-block:: console

   $ sudo apt update && sudo apt upgrade
   $ sudo apt install python3-venv python3-pip python3-tk

虚拟环境
-------------------

创建一个新的虚拟环境并激活它：

.. code-block:: console

   $ python3 -m venv venv
   $ source venv/bin/activate


基于 Pip 的安装
----------------------

CloudnetPy 可以从 Python 包索引 `PyPI
<https://pypi.org/project/cloudnetpy/>`_ 获取。
使用 Python 的包管理器 `pip <https://pypi.org/project/pip/>`_，
将 CloudnetPy 包安装到虚拟环境中：

.. code-block:: console

   (venv)$ pip3 install cloudnetpy

CloudnetPy 现在已准备好从该虚拟环境中使用。
