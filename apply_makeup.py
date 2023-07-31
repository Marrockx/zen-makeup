from utils import *
from landmarks import detect_landmarks, normalize_landmarks, plot_landmarks
from segments import lip_mask, brow_mask, face2_mask, blush_mask


def apply_all_makeup(src: np.ndarray, is_stream: bool, features: list, show_landmarks: bool = False):
    """
    Takes in a source image and applies effects onto it.
    """
    ret_landmarks = detect_landmarks(src, is_stream)
    height, width, _ = src.shape
    if ret_landmarks is not None:
        feature_landmarks = normalize_landmarks(ret_landmarks, height, width)
        output = src.copy()

        for feature in features:
            if feature['name'] == 'lips':
                lip_landmarks = normalize_landmarks(
                    ret_landmarks, height, width, upper_lip + lower_lip)
                mask = lip_mask(output, lip_landmarks, feature['color'])
                output = cv2.addWeighted(output, 1.0, mask, 0.4, 0.0)

            elif feature['name'] == 'lbrow':
                lbrow_landmarks = normalize_landmarks(
                    ret_landmarks, height, width, left_brow)
                mask = brow_mask(output, lbrow_landmarks, feature['color'])
                output = cv2.addWeighted(output, 1.0, mask, 0.4, 0.0)

            elif feature['name'] == 'rbrow':
                rbrow_landmarks = normalize_landmarks(
                    ret_landmarks, height, width, right_brow)
                mask = brow_mask(output, rbrow_landmarks, feature['color'])
                output = cv2.addWeighted(output, 1.0, mask, 0.4, 0.0)

            elif feature['name'] == 'blush':
                blush_landmarks = normalize_landmarks(
                    ret_landmarks, height, width, cheeks)
                mask = blush_mask(output, blush_landmarks,
                                  feature['color'], 50)
                output = cv2.addWeighted(output, 1.0, mask, 0.3, 0.0)

            elif feature['name'] == 'foundation':
                face_landmarks = normalize_landmarks(ret_landmarks, height, width, face_conn)
                mask = face2_mask(output, face_landmarks, feature['color'])
                output = cv2.addWeighted(output, 1.0, mask, 0.3, 0.0)

            if show_landmarks:
                output = plot_landmarks(output, feature_landmarks, True) # type: ignore

        return output

    else:
        print("No landmarks detected.")
