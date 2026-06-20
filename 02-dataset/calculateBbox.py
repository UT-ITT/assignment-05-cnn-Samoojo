def calculate_bbox(x1, y1, x2, y2, image_width, image_height):
    x_left = min(x1, x2)
    x_right = max(x1, x2)
    y_top = min(y1, y2)
    y_bottom = max(y1, y2)

    bbox_width = x_right - x_left
    bbox_height = y_bottom - y_top

    x_center = x_left + bbox_width / 2
    y_center = y_top + bbox_height / 2

    return [
        x_center / image_width,
        y_center / image_height,
        bbox_width / image_width,
        bbox_height / image_height,
    ]


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
