
import numpy as np
import matplotlib.pyplot as plt
from utils.load_data import load_data

# Task1: Visalizing Bird's Eye View


def get_data_in_homogeneous_coordinates():
    data = load_data('data/data.p')
    velodyne_data = data['velodyne']
    velodyne_points = velodyne_data[:, :3]
    velodyne_points_homogeneous = np.concatenate([velodyne_points, np.ones((len(velodyne_points), 1))], axis=1)
    velodyne_intensities = velodyne_data[:, 3]
    return velodyne_points_homogeneous, velodyne_intensities


def transform_to_image_coordinates(points):
    forward_offset = max(points[:, 0])
    left_offset = min(points[:, 1])
    transformation_matrix = np.array([[0, -1, 0, -left_offset], [-1, 0, 0, forward_offset],])
    
    return np.matmul(points, transformation_matrix.T)


def two_dimensions_discretization(points, intensities, resolution_m):
    x_span = max(points[:, 0])
    y_span = max(points[:, 1])

    grid = np.zeros((int(x_span / resolution_m + 1), int(y_span / resolution_m + 1)))

    for point, intensity in zip(points, intensities):
        x = int(point[0] // resolution_m)
        y = int(point[1] // resolution_m)
        old = grid[x, y]
        grid[x, y] = max(old, intensity)

    return grid


def main():
    points_in_velodyne_coordinates, velodyne_intensities = get_data_in_homogeneous_coordinates()
    camera_points = transform_to_image_coordinates(points_in_velodyne_coordinates)
    
    two_d_grid = two_dimensions_discretization(camera_points, velodyne_intensities, resolution_m=0.2)
    figure = plt.figure(figsize=(10,8))
    plt.imshow(two_d_grid.T)
    plt.savefig('plots/t1.png', dpi=300)


if __name__ == '__main__':
    main()

