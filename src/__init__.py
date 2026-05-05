# src/__init__.py
# Makes `src` a proper Python package so that submodules such as
# `src.data`, `src.models.cnn_pretrained`, and `src.models.cnn_scratch`
# can be imported with standard absolute-import syntax across all notebooks
# and scripts regardless of the execution environment (local or server).
