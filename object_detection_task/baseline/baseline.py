import numpy as np


def calculate_brightness(frame: np.ndarray) -> np.ndarray:
    """Calculate the brightness of each pixel in an image frame.

    Args:
        frame (np.ndarray): A numpy array representing an image frame.
            Array shape should be (height, width, 3).

    Returns:
        np.ndarray: A 2D numpy array of the same height and width as the input frame,
            containing the brightness values of each pixel.
    """
    # Normalizing the color values to be between 0 and 1
    normalized_frame = frame / 255.0

    # Calculating the brightness for each pixel
    brightness = np.sum(normalized_frame**2, axis=2) / 3
    return brightness
