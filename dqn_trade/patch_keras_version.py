# patch_keras_version.py
import tensorflow.keras
import tensorflow as tf

# Apply the patch
setattr(tensorflow.keras, '__version__', tf.__version__)


