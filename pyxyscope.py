import sys
import yaml
import cv2
import cvlib
import numpy as np
import os.path
import sounddevice as sd


def __init__(args):
    config_files = [fn for fn in args if fn[-5:] == '.yaml']

    if len(config_files) == 0:
        config_file_name = 'default_config.yaml'
    elif len(config_files) == 1:
        config_file_name = config_files[0]
    else:
        ERR_MSG = 'Too many config files provided.'\
                  ' Only one or none is allowed.'
        raise ValueError(ERR_MSG)

    load_config(config_file_name)


def load_config(config_file_name):
    global config
    with open(config_file_name, 'r') as f:
        config = yaml.safe_load(f)


def read_face_from_file(im_path, down_scale):
    image_in = cv2.imread(im_path)
    faces, _ = cvlib.detect_face(image_in)

    if faces is not None:
        (x1, y1, x2, y2) = faces[0]
        pad = 60
        w = x2-x1 + pad*2
        h = y2-y1 + pad*2
        x = x1-pad
        y = y1-pad
        image_out = image_in[y:y+h, x:x+w]

    return image_out


def invert(image_in):
    try:
        max_val = np.iinfo(image_in.dtype).max
    except ValueError:
        max_val = 1
    return max_val - image_in.copy()


def threshold(image_in, level):
    image_out = np.zeros_like(image_in)
    image_out[image_in > level] = 1
    return image_out


def rms(x):
    return np.sqrt(np.mean(x*np.conj(x)))


def to_float_and_normalize_rms(x):
    y = np.array(x).astype(np.float64)
    return y/rms(y)


def to_float_and_normalize_max(x):
    y = np.array(x).astype(np.float64)
    return y/np.max(y)


def dither(image_in_any_type, threshold_levels=[0.6, 0.8]):
    image_in = to_float_and_normalize_rms(image_in_any_type.copy())
    image_out = threshold(image_in.copy(), level=threshold_levels[0])
    for i, row in enumerate(image_in):
        for j, col in enumerate(row):
            image_out[i, j] = threshold(image_out[i, j],
                                        level=threshold_levels[1])

            err = image_in[i, j] - image_out[i][j]
            if j < (image_in.shape[1]-1):
                image_out[i][j+1] += err*7/16
            if i < (image_in.shape[0]-1):
                image_out[i+1][j] += err*5/16
                if j < (image_in.shape[1]-1):
                    image_out[i+1][j+1] += err*1/16
                if j > 0:
                    image_out[i+1][j-1] += err*3/16

    if np.any((image_out % 1) != 0):
        raise AssertionError('Dither unsucsessful. Generated non binary data.')

    return image_out


def process_image(file_name):
    max_dim_out = max(list(config['desired_resolution'].values()))
    image_path = os.path.join(config['source_dir'], file_name)
    image_in = cv2.imread(image_path)

    max_dim_in = max(image_in.shape)
    down_scale = max_dim_in//max_dim_out
    if down_scale < 1:
        down_scale = 1

    new_size = (image_in.shape[1]//down_scale,
                image_in.shape[0]//down_scale)

    small_image = cv2.resize(image_in, new_size, interpolation=cv2.INTER_CUBIC)
    gray_image = cv2.cvtColor(small_image, cv2.COLOR_BGR2GRAY)
    return dither(gray_image)


def process_face(file_name):
    max_dim_out = max(list(config['desired_resolution'].values()))
    image_path = os.path.join(config['source_dir'], file_name)
    face = read_face_from_file(image_path, down_scale=2)

    max_dim_in = max(face.shape)
    down_scale = max_dim_in//max_dim_out
    if down_scale < 1:
        down_scale = 1

    new_size = (face.shape[1]//down_scale,
                face.shape[0]//down_scale)

    small_image = cv2.resize(face, new_size, interpolation=cv2.INTER_CUBIC)
    gray_image = cv2.cvtColor(small_image, cv2.COLOR_BGR2GRAY)
    return dither(gray_image)


# opencv cannot handle multi line
def process_text(text):
    im_w = config['desired_resolution']['w']
    im_h = config['desired_resolution']['h']
    image = np.zeros((im_h, im_w))
    font = cv2.FONT_HERSHEY_SIMPLEX
    THICKNESS = 1

    step_size = 0.99
    scale = 10/step_size
    text_too_big = True
    while text_too_big:
        scale *= step_size
        ((text_w, text_h), _) = cv2.getTextSize(text=text,
                                                fontFace=font,
                                                fontScale=scale,
                                                thickness=THICKNESS)
        text_too_big = ((text_w > im_w) or (text_h > im_h))

    cv2.putText(image,
                text,
                (0, im_h//2),
                font,
                scale,
                (255, 255, 255),
                THICKNESS)
    return to_float_and_normalize_max(image)


def create_xy_pattern(image):
    fs = config['sample_rate']
    fps = config['desired_frame_rate']

    n_vert = image.shape[1]
    n_horz = image.shape[0]

    n_white = np.sum(image)
    n_hold = int(fs/fps/n_white)
    if n_hold < 1:
        n_hold = 1

    actual_fps = fs/(n_hold*n_white)
    x_dc = []
    y_dc = []

    for i in range(n_horz):
        for j in range(n_vert):
            for k in range(n_hold):
                if (image[-i, j] == 1):
                    x_dc += [j/n_vert]
                    y_dc += [i/n_horz]

    x = x_dc - np.mean(x_dc)
    y = y_dc - np.mean(y_dc)
    return x, y, actual_fps, n_hold


def create_frame(frame_spec):
    functions = {'image': process_image,
                 'face': process_face,
                 'text': process_text}
    bitmap = functions[frame_spec['kind']](frame_spec['source'])
    return create_xy_pattern(bitmap)


def main():
    base_frames = {}
    for frame_number, frame_spec in config['base_frames'].items():
        base_frames[frame_number] = create_frame(frame_spec)

    fs = config['sample_rate']

    x = []
    y = []
    for scene in config['frame_order']:
        bf = base_frames[scene['base_frame']]
        block_size = len(bf[0])
        n_reps = int((scene['duration']*fs)//block_size)
        if n_reps < 1:
            n_reps = 1
        x += np.tile(bf[0], (n_reps,)).tolist()
        y += np.tile(bf[1], (n_reps,)).tolist()

    data = np.array([x/rms(x), y/rms(y)]).T
    dB_vol = -6
    vol = 10**(dB_vol/20)
    sd.play(data*vol, fs, blocking=True)


if __name__ == "__main__":
    __init__(sys.argv[1:])
    main()
else:
    __init__([])
