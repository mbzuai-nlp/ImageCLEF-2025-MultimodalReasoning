import base64
import io
import math
from dataclasses import dataclass

from PIL import Image


@dataclass(frozen=True)
class ImagePreprocessingConfig:
    resize_images: bool = True
    max_image_long_side: int = 2048
    max_image_pixels: int = 2_000_000
    jpeg_quality: int = 90


def image_has_alpha(image) -> bool:
    if "transparency" in image.info:
        return True
    return "A" in image.getbands()


def resize_image_if_needed(image, config: ImagePreprocessingConfig):
    if not config.resize_images:
        return image

    width, height = image.size
    longest_side = max(width, height)
    total_pixels = width * height

    if (
        longest_side <= config.max_image_long_side
        and total_pixels <= config.max_image_pixels
    ):
        return image

    scale = 1.0
    if longest_side > config.max_image_long_side:
        scale = min(scale, config.max_image_long_side / longest_side)
    if total_pixels > config.max_image_pixels:
        scale = min(scale, math.sqrt(config.max_image_pixels / total_pixels))

    new_width = max(1, int(math.floor(width * scale)))
    new_height = max(1, int(math.floor(height * scale)))

    while (
        max(new_width, new_height) > config.max_image_long_side
        or (new_width * new_height) > config.max_image_pixels
    ):
        if new_width >= new_height and new_width > 1:
            new_width -= 1
        elif new_height > 1:
            new_height -= 1
        else:
            break

    if (new_width, new_height) == image.size:
        return image

    return image.resize((new_width, new_height), resample=Image.Resampling.LANCZOS)


def encode_image_data_url(image, config: ImagePreprocessingConfig) -> str:
    buffer = io.BytesIO()
    if image_has_alpha(image):
        image.save(buffer, format="PNG")
        mime_type = "image/png"
    else:
        jpeg_image = image.convert("RGB") if image.mode != "RGB" else image
        jpeg_image.save(
            buffer,
            format="JPEG",
            quality=config.jpeg_quality,
            optimize=True,
        )
        mime_type = "image/jpeg"
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"
