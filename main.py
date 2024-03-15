import numpy as np
from scipy.stats import truncnorm
import matplotlib.pyplot as plt
import noise


# Generate simple landscape
def generate_landscape(width, height, scale):
    color_world = np.zeros((height, width, 3), dtype=np.uint8)

    # Get base sky color and sky color blend factor function
    sky_base_color = get_base_sky_color()
    sky_color_blend_factor_fnc = get_sky_color_blend_factor_fnc(height)

    # Loop through each pixel
    for x in range(width):
        hill_height = int(noise.pnoise1(x * scale, octaves=2) * height / 4 + height / 2)
        for y in range(height):
            if y >= hill_height:  # Create hill
                color_world[y][x] = get_hill_color(x, y, scale, hill_height)
            else:  # Create sky
                color_world[y][x] = blend_colors(sky_base_color, np.array([255, 255, 255]),
                                                 sky_color_blend_factor_fnc(y))
    return color_world


def get_base_sky_color():
    """
    Maps a value in terms of standard deviations from the mean to a color gradient.
    Returns:
    - A tuple representing the RGB values of the corresponding color.
    """

    mean, std_dev = 0, 1
    sample = np.random.normal(mean, std_dev)
    z_score = (sample - mean) / std_dev

    # Define the colors
    mean_color = np.array([0, 0, 139])  # Dark blue
    left_color = np.array([135, 206, 235])  # Light sky blue
    right_color = np.array([255, 160, 122])  # Pinkish reddish orange

    # Interpolate between mean_color and left_color/right_color
    if z_score < 0:
        return blend_colors(mean_color, left_color, abs(z_score))
    else:
        return blend_colors(mean_color, right_color, abs(z_score))


def get_sky_color_blend_factor_fnc(height):

    # Generate y-value of horizon using a Normal distribution centered at 256 (halfway
    # through the height) and truncated at 0 and 512
    horizon_mean = height / 2
    horizon_std_dev = height / 8
    lower_bound = 0
    upper_bound = height

    # Calculate the boundaries for the standard normal distribution
    a, b = (lower_bound - horizon_mean) / horizon_std_dev, (upper_bound - horizon_mean) / horizon_std_dev
    horizon_y_val = truncnorm.rvs(a, b, loc=horizon_mean, scale=horizon_std_dev, size=1)[0]
    return lambda y: ((height - horizon_y_val) - y) / (height - horizon_std_dev)


def get_hill_color(x, y, scale, hill_height):
    # Set base grass colors
    grass_color_1 = np.array([65, 152, 10])
    grass_color_2 = np.array([19, 133, 16])

    # Generate a gradient level based on noise and distance up the hill
    noise_value = (noise.pnoise2(x * scale * 8, y * scale * 8) + 1) / 2
    blend_factor = noise_value * (y - hill_height) / hill_height

    # Set hill color to noisy mix of two grass colors
    return blend_colors(grass_color_1, grass_color_2, blend_factor)


def blend_colors(color1, color2, x):
    return np.round(color1 + (color2 - color1) * x).astype(int)


def main():
    width, height = 1024 * 2, 1024
    scale = 0.001

    color_world = generate_landscape(width, height, scale)

    plt.imshow(color_world)
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    main()
