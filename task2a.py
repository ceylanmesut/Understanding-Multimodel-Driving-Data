
import numpy as np
import matplotlib.pyplot as plt

from utils.load_data import load_data


def get_info_from_data():
    data = load_data('data/data.p')
    velodyne_points = data['velodyne']
    velodyne_points[:, 3] = np.ones(len(velodyne_points))
    image = data['image_2']
    t_cam0_velo = data['T_cam0_velo']
    p_rect_20 = data['P_rect_20']
    sem_label = data['sem_label']
    color_map = data['color_map']
    return velodyne_points, image, t_cam0_velo, p_rect_20, sem_label, color_map


def get_points_on_cam2_plane(front_velodyne_points, p_rect_20, t_cam0_velo):
    proj_matrix = np.matmul(p_rect_20, t_cam0_velo)
    cam2_point_plane = np.matmul(front_velodyne_points, proj_matrix.T)
    for i in range(len(cam2_point_plane)):
        cam2_point_plane[i] = cam2_point_plane[i] / cam2_point_plane[i, -1]
    return cam2_point_plane


def main():
    velodyne_points, image, t_cam0_velo, p_rect_20, sem_label, color_map = get_info_from_data()

    indices_of_points_with_positive_x = np.where(velodyne_points[:, 0] > 0)[0]
    front_velodyne_points = velodyne_points[indices_of_points_with_positive_x]
    sem_label = sem_label[indices_of_points_with_positive_x].reshape(-1)

    cam2_point_plane = get_points_on_cam2_plane(front_velodyne_points, p_rect_20, t_cam0_velo)

    for i in range(len(cam2_point_plane)):
        u, v, _ = cam2_point_plane[i]
        if 0 <= u <= 1239 and 0 <= v <= 375:
            image[int(v), int(u)] = color_map[sem_label[i]]
            image[int(v)+1, int(u)] = color_map[sem_label[i]]
            image[int(v), int(u)+1] = color_map[sem_label[i]]
            image[int(v)+1, int(u)+1] = color_map[sem_label[i]]

    plt.imshow(image)
    plt.savefig("plots/t2a.png", dpi=300)


if __name__ == '__main__':
    main()

