def hex_to_rgb(hex_value):
    # Remove the '#' symbol if present
    hex_value = hex_value.lstrip('#')

    # Convert hexadecimal to decimal
    red = int(hex_value[0:2], 16)
    green = int(hex_value[2:4], 16)
    blue = int(hex_value[4:6], 16)

    return [red, green, blue]



def rgb_to_bgr(rgb_value):
    r, g, b = rgb_value
    bgr_value = [b, g, r]
    return bgr_value