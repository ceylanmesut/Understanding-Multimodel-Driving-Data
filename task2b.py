
import numpy as np
import matplotlib.pyplot as plt
from utils.load_data import load_data

def get_data():
    data = load_data('data/data.p')
    image = data['image_2']
    p_rect_20 = data['P_rect_20']
    objects = data['objects'][:-1]
    return image, p_rect_20, objects


def get_rotation_matrix_around_y(angle):
    # https://en.wikipedia.org/wiki/Rotation_matrix
    angle *= -1
    return np.array([
        [np.cos(angle), 0, -np.sin(angle)],
        [0, 1, 0],
        [np.sin(angle), 0, np.cos(angle)]
    ])


def get_full_transformation_matrix(translation, angle):
    rotation = get_rotation_matrix_around_y(angle)
    full_matrix = np.identity(4)
    full_matrix[:3, :3] = rotation
    full_matrix[:3, 3] = translation
    return full_matrix


def get_axis_aligned_box_on_origin(dimensions):
    height, width, length = dimensions
    output = np.zeros((4, 4))
    output[:, -1] = 1  # homogeneous coordinates
    # do the bottom part from the positive quadrant clockwise
    output[0, :3] = np.array([length/2, 0, width/2])
    output[1, :3] = np.array([-length/2, 0, width/2])
    output[2, :3] = np.array([-length/2, 0, -width/2])
    output[3, :3] = np.array([length/2, 0, -width/2])
    # do the top part of the bounding box
    output = np.concatenate([output, output], axis=0)
    output[4:, 1] = -height
    return output


def plot_square(corners, color):
    for i in range(len(corners)):
        start_x, start_y = corners[i]
        end_x, end_y = corners[(i + 1) % 4]
        plt.plot([start_x, end_x], [start_y, end_y], color=color)


def plot_boxes(boxes):
    for box in boxes:
        color = 'orange' if box[-1, -1] != boxes[-1, -1, -1] else 'red'
        plot_square(box[:4], color)
        plot_square(box[4:], color)

        for i in range(4):
            start_x, start_y = box[i]
            end_x, end_y = box[i + 4]
            plt.plot([start_x, end_x], [start_y, end_y], color=color)


def main():
    image, p_rect_20, objects = get_data()

    axis_aligned_boxes = [get_axis_aligned_box_on_origin(obj[1:4]) for obj in objects]

    cam2_3d_boxes = []
    for axis_aligned_box, obj in zip(axis_aligned_boxes, objects):
        full_matrix = get_full_transformation_matrix(obj[4: 7], obj[-1])
        cam2_3d_boxes.append(np.matmul(axis_aligned_box, full_matrix.T))

    not_scaled_rectified_boxes = [np.matmul(box, p_rect_20.T) for box in cam2_3d_boxes]
    boxes = np.apply_along_axis(lambda p: p[:2] / p[2], 1, np.concatenate(not_scaled_rectified_boxes, axis=0))
    boxes = boxes.reshape((-1, 8, 2))

    plt.imshow(image)
    plot_boxes(boxes)
    plt.savefig("plots/t2b.png", dpi=300)

if __name__ == '__main__':
    main()
