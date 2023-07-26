from utils import *

def lip_mask(src: np.ndarray, points: np.ndarray, color: list):
    """
    Given a src image, points of lips and a desired color
    Returns a colored mask that can be added to the src
    """
    mask = np.zeros_like(src)  # Create a mask
    # Mask for the required facial feature
    mask = cv2.fillPoly(mask, [points], color)
    # Blurring the region, so it looks natural
    # TODO: Get glossy finishes for lip colors, instead of blending in replace the region
    #mask = cv2.cvtColor(mask, cv2.COLOR_RGB2BGR)
    mask = cv2.GaussianBlur(mask, (7, 7), 10)
    return mask


def brow_mask(src: np.ndarray, points: np.ndarray, color: list):

    mask = np.zeros_like(src)  # Create a mask
    # Mask for the required facial feature
    mask = cv2.fillPoly(mask, [points], color)
    # Blurring the region, so it looks natural
    # mask = cv2.cvtColor(mask, cv2.COLOR_RGB2BGR)
    mask = cv2.GaussianBlur(mask, (7, 7), 10)
    return mask


def face2_mask(src: np.ndarray, points: np.ndarray, color: list):

    mask = np.zeros_like(src)  # Create a mask
    # Mask for the required facial feature
    mask = cv2.fillPoly(mask, [points], color)
    # Blurring the region, so it looks natural
    # mask = cv2.cvtColor(mask, cv2.COLOR_RGB2BGR)
    mask = cv2.GaussianBlur(mask, (7, 7), 5)
    return mask


def blush_mask(src: np.ndarray, points: np.ndarray, color: list, radius: int):
    """
    Given a src image, points of the cheeks, desired color and radius
    Returns a colored mask that can be added to the src
    """
    # TODO: Make the effect more subtle
    mask = np.zeros_like(src)  # Mask that will be used for the cheeks
    for point in points:
        # Blush => Color filled circle
        mask = cv2.circle(mask, point, radius, color, cv2.FILLED)
        x, y = point[0] - radius, point[1] - radius  # Get the top-left of the mask
        mask[y:y + 2 * radius, x:x + 2 * radius] = vignette(mask[y:y + 2 * radius, x:x + 2 * radius],
                                                            20)  # Vignette on the mask

        # mask = cv2.cvtColor(mask, cv2.COLOR_RGB2BGR)
    return mask

def mask_skin(src: np.ndarray):
    """
    Given a source image of a person (face image)
    returns a mask that can be identified as the skin
    """
    lower = np.array(
        [0, 133, 77], dtype='uint8')  # The lower bound of skin color
    # Upper bound of skin color
    upper = np.array([255, 173, 127], dtype='uint8')
    dst = cv2.cvtColor(src, cv2.COLOR_BGR2YCR_CB)  # Convert to YCR_CB
    skin_mask = cv2.inRange(dst, lower, upper)  # Get the skin
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    # Dilate to fill in blobs
    skin_mask = cv2.dilate(skin_mask, kernel, iterations=2)[..., np.newaxis]

    if skin_mask.ndim != 3:
        skin_mask = np.expand_dims(skin_mask, axis=-1)
    # A binary mask containing only 1s and 0s
    return (skin_mask / 255).astype("uint8")


def face_mask(src: np.ndarray, points: np.ndarray):
    """
    Given a list of face landmarks, return a closed polygon mask for the same
    """
    mask = np.zeros_like(src)
    mask = cv2.fillPoly(mask, [points], (255, 255, 255))
    #mask = cv2.cvtColor(mask, cv2.COLOR_RGB2BGR)
    return mask


def vignette(src: np.ndarray, sigma: int):
    """
    Given a src image and a sigma, returns a vignette of the src
    """
    height, width, _ = src.shape
    kernel_x = cv2.getGaussianKernel(width, sigma)
    kernel_y = cv2.getGaussianKernel(height, sigma)

    kernel = kernel_y * kernel_x.T
    mask = kernel / kernel.max()
    blurred = cv2.convertScaleAbs(src.copy() * np.expand_dims(mask, axis=-1))
    return blurred
