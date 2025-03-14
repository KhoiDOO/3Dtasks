import numpy as np
import io
import base64
from PIL import Image

# Helper function to convert numpy array to base64
def array_to_base64(array):
    image = Image.fromarray((array * 255).astype(np.uint8))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")