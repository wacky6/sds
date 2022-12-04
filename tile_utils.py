import numpy as np
from typing import Generator, List, Union

"""
Split image into tiles.
- image: Tensor with shape (rows, cols, channels)
- tile: int, pixel count in rows / cols
- padding: int, number of pixels along width or height that are shared across two tiles in one tile movement

If image is smaller than (tile, size), the remaining area is filled with 0.
If image is larger than (tile, tile), the returned tiles will be within the original image.

Returns a list of Tensor with shape (tile, tile, channels)
"""
def clever_tiles(image: np.ndarray, tile, padding):
    assert len(image.shape) == 3, "image should be (rows, cols, channel)"
    assert tile > 0, "tile size should be positive"
    assert tile % 8 == 0, "tile size should be multiples of 8"
    assert padding >= 0, "padding size should be non-neative"
    assert padding % 2 == 0, "padding size be multiples of 2 to allow clever_merge_tiles"
    assert padding < tile, "padding size should be less than tile size"

    h, w, c = image.shape

    # Ensure image produces at least 1 tile.
    pad_h, pad_w = max(tile, h), max(tile, w)
    image = np.pad(image, [(0, pad_h), (0, pad_w), (0,0)])

    top, left = 0, 0
    result = []

    while True:
        while True:
            result.append(image[top:top+tile, left:left+tile, :])
            if left >= w - tile:
                break
            left = min(left + tile - padding, w - tile)
        left = 0
        if top >= h - tile:
            break
        top = min(top + tile - padding, h - tile)

    return result

"""
Merge scaled tiles generated by clever_tile(image, tile, padding) into a single image.

Returns a Tensor with shape (scale*image.height, scale*image_width, image.channels)

- `image`, `tile`, `padding` are the values passed to clever_tile().
- `scale` is the scaling factor that applies to processed tiles, must be integer.

Constraint: `scale * tile` must be divisible by 2.
"""
# NOTE: Tested correct
def clever_merge(tiles: List[np.ndarray], image: np.ndarray, tile: int, padding: int, scale: int):
    assert len(image.shape) == 3, "image should be (rows, cols, channel)"
    assert tile > 0, "tile size should be positive"
    assert padding >= 0, "padding size should be non-neative"
    assert padding % 2 == 0, "padding size be multiples of 2 to allow clever_merge_tiles"
    assert padding < tile, "padding size should be less than tile size"
    assert scale > 0 and isinstance(scale, int), "scale should be a positive integer"
    assert scale * tile % 2 == 0, "scale * tile must be divisible by 2 for merging"

    h, w, c = image.shape
    # Ensure image produces at least 1 tile.
    pad_h, pad_w = max(tile, h), max(tile, w)
    image = np.pad(image, [(0, pad_h), (0, pad_w), (0,0)])

    out = np.zeros([image.shape[0]*scale, image.shape[1]*scale, c])

    top, left = 0, 0
    nth_tile = 0

    while True:
        while True:
            assert nth_tile < len(tiles), """number of tiles is less than the expected amount,
                 please check clever_merge and clever_tiles's `tile` and `padding` arguments match"""
            scaled_top, scaled_left = top * scale, left * scale
            scaled_tile = scale * tile
            scaled_half_padding = scale * padding // 2
            if scaled_left == 0 and scaled_top == 0:
                out[
                    0:scaled_tile,
                    0:scaled_tile,
                    :,
                ] = tiles[nth_tile][
                    :,
                    :,
                    :,
                ]
            elif scaled_left == 0 and scaled_top != 0:
                out[
                    scaled_top+scaled_half_padding:scaled_top + scaled_tile,
                    0:scaled_tile,
                    :
                ] = tiles[nth_tile][
                    scaled_half_padding:,
                    :,
                    :,
                ]
            elif scaled_left != 0 and scaled_top == 0:
                out[
                    0:scaled_tile,
                    scaled_left+scaled_half_padding:scaled_left + scaled_tile,
                    :,
                ] = tiles[nth_tile][
                    :,
                    scaled_half_padding:,
                    :,
                ]
            elif scaled_left != 0 and scaled_top != 0:
                out[
                    scaled_top+scaled_half_padding:scaled_top + scaled_tile,
                    scaled_left+scaled_half_padding:scaled_left + scaled_tile,
                    :,
                ] = tiles[nth_tile][
                    scaled_half_padding:,
                    scaled_half_padding:,
                    :,
                ]
            else:
                assert false, "Not-reached"

            nth_tile += 1
            if left >= w - tile:
                break
            left = min(left + tile - padding, w - tile)
        left = 0
        if top >= h - tile:
            break
        top = min(top + tile - padding, h - tile)

    # Do a final crop
    out = out[0:h*scale, 0:w*scale, :]
    return out