def calculate_bbox(x1, y1, x2, y2, image_width, image_height):
    # ensure ordering
    x_left = min(x1, x2)
    y_top = min(y1, y2)
    x_right = max(x1, x2)
    y_bottom = max(y1, y2)

    # convert to normalized top-left format
    x = x_left / image_width
    y = y_top / image_height

    w = (x_right - x_left) / image_width
    h = (y_bottom - y_top) / image_height

    return [x, y, w, h]


bbox1 = calculate_bbox(
    x1=285,
    y1=2766,
    x2=860,
    y2=1891,
    image_width=2316,
    image_height=3088,
)

print("bbox1:")
print([round(v, 8) for v in bbox1])

bbox2 = calculate_bbox(
    x1=160,
    y1=2354,
    x2=860,
    y2=1475,
    image_width=2316,
    image_height=3088,
)

print("bbox2:")
print([round(v, 8) for v in bbox2])

bbox3 = calculate_bbox(
    x1=89,
    y1=2380,
    x2=802,
    y2=1341,
    image_width=2316,
    image_height=3088,
)

print("bbox3:")
print([round(v, 8) for v in bbox3])
