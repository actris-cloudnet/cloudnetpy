# Sphinx 文档的最小 makefile
#

# 你可以从命令行设置这些变量。
SPHINXOPTS    = -E -a
SPHINXBUILD   = sphinx-build
SOURCEDIR     = source
BUILDDIR      = .

# 首先放置，以便没有参数的 "make" 就像 "make help"。
help:
	@$(SPHINXBUILD) -E help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# 捕获所有目标：使用新的 "make mode" 选项将所有未知目标路由到 Sphinx。
# $(O) 用作 $(SPHINXOPTS) 的快捷方式。
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
