# src/models/__init__.py
# Makes `src.models` a proper Python package so that model-family subpackages
# such as `src.models.cnn_pretrained`, `src.models.cnn_scratch`,
# `src.models.vit`, and `src.models.vit_scratch` are
# discoverable as first-class Python packages.
# Without this file the relative imports inside each model subpackage
# (`from .models import ...`, `from .utils import ...`) raise a NameError
# because Python cannot resolve the parent package context.
