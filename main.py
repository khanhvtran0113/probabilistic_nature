import numpy as np
from scipy.stats import truncnorm
from scipy.stats import bernoulli
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
                sky_white = blend_colors(sky_base_color, np.array([255, 255, 255]), 0.9)
                color_world[y][x] = blend_colors(sky_base_color, sky_white,
                                                 sky_color_blend_factor_fnc(y))

    add_clouds(color_world, width, height, scale)
    add_rocks(color_world, width, height, scale)
    add_trees(color_world, width, height, scale)
    return color_world


def add_clouds(color_world, width, height, scale):
    num_of_clouds = int(np.random.geometric(1 / 5))

    cloud_locations = list()
    for i in range(num_of_clouds):
        x = int(np.random.uniform(0, width))
        hill_height = int(noise.pnoise1(x * scale, octaves=2) * height / 4 + height / 2)
        y = int(np.random.uniform(0, hill_height-300))
        cloud_locations.append([x, y])

    for x, y in cloud_locations:
        draw_clouds(color_world, x, y, height, width, scale)


def draw_clouds(color_world, x, y, height, width, scale):
    aspect_ratio = np.random.uniform(1.5, 2.5)
    imperfection = scale * 500

    cloud_size = int(np.random.normal(200, scale * 1000))
    cloud_color = np.array([250, 250, 250])

    radius_x = cloud_size // 2
    radius_y = int(radius_x / aspect_ratio)
    for i in range(-radius_y, radius_y + 1):
        for j in range(-radius_x, radius_x + 1):
            distance = (j / radius_x) ** 2 + (i / radius_y) ** 2
            if distance <= 1 and 0 < y - i < height and 0 < x + j < width:
                fuzziness = np.random.uniform(-imperfection, imperfection)
                if distance + fuzziness <= 1:
                    color_world[y - i, x + j] = cloud_color


def add_rocks(color_world, width, height, scale):
    num_of_rocks = int(np.random.geometric(1 / 5))

    rock_locations = list()
    for i in range(num_of_rocks):
        x = int(np.random.uniform(0, width))
        hill_height = int(noise.pnoise1(x * scale, octaves=2) * height / 4 + height / 2)
        y = int(np.random.uniform(hill_height, height))
        rock_locations.append([x, y])

    for x, y in rock_locations:
        draw_rocks(color_world, x, y, height, width, scale)


def draw_rocks(color_world, x, y, height, width, scale):
    aspect_ratio = np.random.uniform(1.2, 2)
    imperfection = scale * 100

    rock_size = int(180 * (y - 0.5 * height) / height)
    rock_color = np.array([128, 128, 128])

    radius_x = rock_size // 2
    radius_y = int(radius_x / aspect_ratio)
    for i in range(-radius_y, radius_y + 1):
        for j in range(-radius_x, radius_x + 1):
            if ((j / (radius_x * (1 + np.random.uniform(-imperfection, imperfection)))) ** 2 +
                (i / (radius_y * (1 + np.random.uniform(-imperfection, imperfection)))) ** 2) <= 1:
                if 0 < y - i < height and 0 < x + j < width:
                    color_world[y - i, x + j] = rock_color


def add_trees(color_world, width, height, scale):
    # Generate list of centers
    num_of_centers = np.random.binomial(3, 0.5)
    centers = list()
    for i in range(num_of_centers):
        cx = int(np.random.uniform(0, width))
        hill_height = int(noise.pnoise1(cx * scale, octaves=2) * height / 4 + height / 2)
        cy = int(np.random.uniform(hill_height, height))
        centers.append([cx, cy])

    # Randomly pick season
    season = np.random.randint(0, 3)

    # Loop through x and y
    for x in range(0, width, 50):
        hill_height = int(noise.pnoise1(x * scale, octaves=2) * height / 4 + height / 2)
        for y in range(hill_height, height, 50):
            if rbf_distribution(x, y, centers, 300):
                draw_tree(color_world, x, y, height, width, season, scale)


def draw_tree(color_world, x, y, height, width, season, scale):
    if season == 0:  # Spring
        bark_color = np.array([140, 70, 20])  # Lighter brown for spring
        leaf_color = np.array([60, 179, 113])  # Light green for spring leaves
    elif season == 1:  # Summer
        bark_color = np.array([101, 67, 33])  # Dark brown for summer
        leaf_color = np.array([0, 128, 0])  # Dark green for summer leaves
    elif season == 2:  # Fall
        bark_color = np.array([150, 75, 25])  # Medium brown for fall
        leaf_color = np.array([255, 165, 0])  # Orange for fall leaves
    else:  # Winter
        bark_color = np.array([120, 60, 30])  # Grey-brown for winter
        leaf_color = np.array([255, 255, 255])  # White for snow-covered leaves

    # Get initial tree size
    tree_height = int(240 * (y - 0.5 * height) / height)
    tree_width = int(120 * (y - 0.5 * height) / height)

    # Add variance to tree size
    tree_height = int(tree_height * np.random.normal(1, scale * 200))
    tree_width = int(tree_width * np.random.normal(1, scale * 200))

    # Draw the trunk
    trunk_height = tree_height // 3
    trunk_width = tree_width // 3
    trunk_x = x + (tree_width - trunk_width) // 2
    for i in range(trunk_height):
        for j in range(trunk_width):
            if 0 < trunk_x + j < width and 0 < y - i < height:
                color_world[y - i][trunk_x + j] = bark_color

    # Draw the leaves
    leaves_height = tree_height - trunk_height
    leaves_width = tree_width
    for i in range(leaves_height):
        for j in range(leaves_width):
            if abs(j - leaves_width // 2) <= leaves_height - i:
                if 0 < x + j < width and 0 < y - trunk_height - i < height:
                    color_world[y - trunk_height - i][x + j] = leaf_color


def rbf_distribution(x, y, centers, sigma):
    probability = 0
    for cx, cy in centers:
        distance = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        probability += np.exp(-distance ** 2 / (2 * sigma ** 2))
    return 1 == bernoulli.rvs(probability / 10)


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
    horizon_mean = height / 4
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
