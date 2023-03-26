import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
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


def angle_to_x_y_plane(vector_1: np.ndarray) -> float:
    vector_2 = np.copy(vector_1)
    vector_2[2] = 0
    unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
    unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    return np.degrees(np.arccos(dot_product)) * np.sign(vector_1[-1])


def angle_to_x_z_plane(vector_1: np.ndarray) -> float:
    vector_2 = np.copy(vector_1)
    vector_2[1] = 0
    unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
    unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    return np.degrees(np.arccos(dot_product)) * np.sign(vector_1[1])


def main():
    velodyne_points, image, t_cam0_velo, p_rect_20, sem_label, color_map = get_info_from_data()
    points_as_angles = np.array([[angle_to_x_z_plane(p), angle_to_x_y_plane(p)] for p in velodyne_points[:, :3]])

    flags = np.logical_and(velodyne_points[:, 0] > 0, np.abs(points_as_angles[:, 0]) < 40)
    flags = np.logical_and(flags, np.logical_not(np.isnan(points_as_angles[:, 0])))
    flags = np.logical_and(flags, np.logical_not(np.isnan(points_as_angles[:, 1])))

    filtered_velodyne_points = velodyne_points[flags]
    filtered_points_as_angles = points_as_angles[flags]

    centers = KMeans(n_clusters=64, random_state=42).fit(filtered_points_as_angles[:, 1].reshape((-1, 1))).cluster_centers_
    centers = np.sort(centers.reshape(-1))

    cam2_points = get_points_on_cam2_plane(filtered_velodyne_points, p_rect_20, t_cam0_velo)
    colors = np.array([
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
    ])
    for i in range(len(cam2_points)):
        u, v, _ = cam2_points[i]
        if 1 <= u <= 1239 and 1 <= v <= 375:
            angle_to_x_y = filtered_points_as_angles[i, 1]
            index_of_closest_center = np.abs(centers - angle_to_x_y).argmin()
            c = colors[index_of_closest_center % 3]
            for box_i in range(-1, 2):
                for box_j in range(-1, 2):
                    image[int(v) + box_i, int(u) + box_j] = c

    plt.imshow(image)
    plt.savefig("plots/t3.png", dpi=300)


if __name__ == '__main__':
    main()
