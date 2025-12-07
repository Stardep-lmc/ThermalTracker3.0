# models/__init__.py
from .motr import build

def build_model(args):
    return build(args)