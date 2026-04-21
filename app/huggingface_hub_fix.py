"""
Nuclear fix for huggingface_hub cached_download ImportError
This module completely replaces the broken import
"""

# Import the real huggingface_hub
import huggingface_hub
from huggingface_hub import hf_hub_download

# Force the cached_download attribute to exist
huggingface_hub.cached_download = hf_hub_download

# Export everything from the real module
__all__ = huggingface_hub.__all__

# Make this module behave exactly like the real one
for attr in dir(huggingface_hub):
    if not attr.startswith('_'):
        globals()[attr] = getattr(huggingface_hub, attr)

print("NUCLEAR FIX: huggingface_hub.cached_download successfully patched")
