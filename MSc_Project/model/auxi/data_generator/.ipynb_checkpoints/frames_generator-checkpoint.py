import h5py
import random
import numpy as np

from .shapes import triangle, square, circle


shape_func = {}
shape_func['square'] = square
shape_func['circle'] = circle
shape_func['triangle'] = triangle


def latent_generator(M_min, M_max, width, restrictions):
    lm = []
    num_elements = np.random.randint(M_min, M_max + 1)
    for _ in range(num_elements):
        lv = {}
        lv["x"], lv["y"] = restrictions["position"] if "position" in restrictions else (np.random.randint(width), np.random.randint(width))
        lv["color"] = restrictions["color"] if "color" in restrictions else random.choice([(1, 0, 0), (0, 1, 0), (0, 0, 1)])
        lv["shape"] = restrictions["shape"] if "shape" in restrictions else random.choice(['square', 'circle', 'triangle'])
        lm.append(lv)
    return lm




def latent_to_frame(lm, width):
    def paint(frame, painted, element, color):
        for i in range(width):
            for j in range(width):
                if bool(painted[i, j]):
                    pass
                else:
                    if element[i, j] == 1:
                        frame[0, i, j] = color[0]
                        frame[1, i, j] = color[1]
                        frame[2, i, j] = color[2]
                        painted[i, j] = 1
        return frame, painted
    frame = np.zeros((3, width, width))
    painted = np.zeros((width, width))
    for lv in lm:
        center = (lv["x"], lv["y"])
        element = shape_func[lv["shape"]](center, width)
        color = lv["color"]
        frame, painted = paint(frame, painted, element, color)
    return frame


def label_from_matrix(l_matrix, M, width):
    def get_shape_vector(shape):
        return [shape=='triangle', shape=='square', shape=='circle']
    latent_vector = []
    for obj in l_matrix:
        obj_vector = [obj['x']/width, obj['y']/width, obj['color'][0], obj['color'][1], obj['color'][2]]
        shape_vector = get_shape_vector(obj['shape'])
        obj_vector.extend(shape_vector)
        latent_vector.extend(obj_vector)
    return np.pad(np.array(latent_vector, dtype=float), (0,    8*M-len(latent_vector)), 'constant')