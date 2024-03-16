# Landscape Generator

This Python script generates a procedural landscape image using Perlin noise and other random distributions. The landscape consists of hills, sky, clouds, rocks, and trees, each with varying colors and sizes.

## Features

- Generates a landscape with a sky, hills, clouds, rocks, and trees.
- Utilizes Perlin noise to create natural-looking terrain.
- Randomly places clouds, rocks, and trees based on statistical distributions.
- Supports customization of image dimensions and scale.

## Requirements

- Python 3
- NumPy
- SciPy
- Matplotlib
- `noise` module (for Perlin noise)

## Usage

1. Ensure all required libraries are installed.
2. Run the script:
   ```bash
   python landscape_generator.py
   ```
## Customization
You can customize the landscape by modifying the following parameters in the main() function:

- width: Width of the generated image.
- height: Height of the generated image.
- scale: Scale factor for the randomness, affecting the terrain variance.
