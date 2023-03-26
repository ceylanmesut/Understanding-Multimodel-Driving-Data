

from utils.load_data import load_data
import os
from  utils import data_utils
import numpy as np
import matplotlib.pyplot as plt
import math 
import cv2
import pandas as pd


def get_paths(frame_number):
    """Accepts frame number as three digits i.e 037, 345, 002"""

    frame_number=str(frame_number)
    velo_points_path=("data\\problem_4\\velodyne_points\\data\\0000000"+frame_number+".bin")
    img_path=("data\\problem_4\\image_02\\data\\0000000"+frame_number+".png")
    angular_velocity_path=("data\\problem_4\\oxts\\data\\0000000"+frame_number+".txt")
    velocity_path=("data\\problem_4\\oxts\data\\0000000"+frame_number+".txt")

    return velo_points_path, img_path, angular_velocity_path, velocity_path

def get_time_files_path():

    start_time_path=("data\\problem_4\\velodyne_points\\timestamps_start.txt")
    end_time_path=("data\problem_4\\velodyne_points\\timestamps_end.txt")
    cam_trigger_time_path=("data\\problem_4\\velodyne_points\\timestamps.txt")

    return start_time_path, end_time_path, cam_trigger_time_path

def project_points(points, img_path):

    # Loading Velodyne points and filtering them.
    velo_points=points
    shape_x=np.shape(velo_points)[0]   
    velo_points=np.hstack((velo_points, np.ones(shape=(shape_x, 1)))) 
    pos_point_ind = np.where(velo_points[:, 0] > 0)[0]
    points_front = velo_points[pos_point_ind]


    # Building Rotation and Translation Matrix
    rotation, translation =data_utils.calib_velo2cam("data\\problem_4\\calib_velo_to_cam.txt")
    rotation=np.vstack((rotation, np.zeros(shape=(1,3)))) #extending rotation matrix
    translation=np.vstack((translation, np.ones(shape=(1,1))))
    r_t_matrix = np.hstack((rotation, translation)) # 4x4

    # Obtaining Camera 2 Intrinsic Parameters
    cam2_intrinsics=data_utils.calib_cam2cam("data\\problem_4\\calib_cam_to_cam.txt", mode="02")
    cam2_intrinsics=np.hstack((cam2_intrinsics, np.zeros(shape=(3,1)))) # 3x4 matrix

    # Creating Projection Points
    proj_matrix=np.matmul(cam2_intrinsics, r_t_matrix)
    projection=np.matmul(points_front, proj_matrix.T)
    projection=projection.T
    projection_image = projection / projection[2] # Normalizing 

    color = data_utils.depth_color(projection[2], min_d=projection[2].min(), max_d=projection[2].max())
    img=cv2.imread(img_path)
    valid_points = (projection_image[0]>=0) & (projection_image[0]<img.shape[1]) & (projection_image[1]>=0) & (projection_image[1]<img.shape[0])
    projection_image = projection_image[:,valid_points]

    img_to_show = data_utils.print_projection_plt(projection_image, color[valid_points], img)

    return img_to_show

def project_undist_points(points, img_path):

    velo_points=points
    shape_x=np.shape(velo_points)[0]   
    velo_points=np.hstack((velo_points, np.ones(shape=(shape_x, 1))))
    pos_point_ind = np.where(velo_points[:, 0] > 0)[0]
    points_front = velo_points[pos_point_ind]


def calculate_timestamps(start_time_path, end_time_path, cam_trigger_time_path, num_of_frame):

    # Building lists for dataframe skeleton
    start_time_list=[]
    end_time_list=[]
    camera_trigger_list=[]
    index_list=[]

    # Building lists for dataframe skeleton
    start_time_list=[]
    end_time_list=[]
    camera_trigger_list=[]
    index_list=[]

    for i in range(num_of_frame):
        file_index="{0:010}".format(i)    

        # Obtaining timestamps by calling provided from data_utils.
        start_time=data_utils.compute_timestamps(timestamps_f=start_time_path, ind=file_index)
        end_time=data_utils.compute_timestamps(timestamps_f=end_time_path, ind=file_index)
        cam_trigger_time=data_utils.compute_timestamps(timestamps_f=cam_trigger_time_path, ind=file_index)
        
        # Adding values to respected lists.
        index_list.append(file_index)
        start_time_list.append(start_time)
        end_time_list.append(end_time)
        camera_trigger_list.append(cam_trigger_time)
    
    return index_list, start_time_list, end_time_list, camera_trigger_list


def create_df_for_angle_calculation(frame_index_data, start_time_data, end_time_data, camera_trigger_data):
    
    
    # Creating dataframe for computing radian and angle
    dataframe=pd.DataFrame(columns=["Frame_Index","Start_Time","End_Time","Frame_Trigger_Time","LIDAR_Turn_Time"])
    dataframe["Frame_Index"]=frame_index_data
    dataframe["Start_Time"]=start_time_data
    dataframe["End_Time"]=end_time_data
    dataframe["Frame_Trigger_Time"]=camera_trigger_data
    dataframe["LIDAR_Turn_Time"]=(dataframe["End_Time"]-dataframe["Start_Time"])
    dataframe["LIDAR_Turn_Radian"]=6.28319 #Radian equivalent of 360 degree
    dataframe["LIDAR_Start_Time-Frame_Trigger_Time"]=(dataframe["Start_Time"]-dataframe["Frame_Trigger_Time"])
    dataframe["Diff_LIDAR_Cam_Radian"]=((dataframe["LIDAR_Start_Time-Frame_Trigger_Time"])*(dataframe["LIDAR_Turn_Radian"]))/(dataframe["LIDAR_Turn_Time"])
    dataframe["LIDAR_Angle"]=dataframe["Diff_LIDAR_Cam_Radian"]*180/math.pi

    return print("Angle Relation Between LIDAR and Camera\n", dataframe["LIDAR_Angle"].head(6))

def undistort_velo_points(velo_points_path, angular_velocity_path, velocity_path, img_path):   
    
    # Loading velodyne points and preparing the array.
    velo_points_original=data_utils.load_from_bin(velo_points_path)
    shape_x=np.shape(velo_points_original)[0]   

    velo_points=np.delete(velo_points_original, obj=2,axis=1) 
    velo_points_x_y=np.hstack((velo_points, np.ones(shape=(shape_x, 1))))


    # Obtaining x and y values.
    velo_points_x=velo_points_x_y[:,0]
    velo_points_y=velo_points_x_y[:,1]

    # Computing Inverse Tangent of x, y to compute angle at x-y plane.
    ratio_y_x=velo_points_y/velo_points_x
    angle_of_points=np.arctan(ratio_y_x) # in radians

    # Computing Rotation Matrix along x and y axis

    # Loading angular velocity values from GPS/IMU recordings
    angular_velocity = data_utils.load_oxts_angular_rate(angular_velocity_path)
    angular_velocity_z= angular_velocity[-1] # Obtaining angular velocity around z axis for rotation.

    # for each point, theta value of translation matrix is computed. Since GPS/IMU system records at 100 Hz and LIDAR 10 Hz
    # 0.1 is multiplied to balance out the timings of recordings.
    theta = (((angular_velocity_z * angle_of_points) / (2*math.pi)) * 0.1) * -1


    # Loading velocity valus from GPS/IMU and computing translation. 
    velocity=data_utils.load_oxts_velocity(velocity_path)
    translation_x = ([velocity[0] * ((angle_of_points / (2*math.pi)*0.1)*-1)])  
    translation_y = np.array([velocity[1] * ((angle_of_points / (2*math.pi)*0.1)*-1)]) 


    # Creating velo_points array like array to fill it with new points.
    undistorted_points=np.ones_like(velo_points_x_y)

    # Now for each point, Rotation and Translation Matrix is calculated.
    for i in range(len(velo_points_x_y)):
        
        # Building Rotation and Translation Matrices
        rotation_matrix = np.array([[np.cos(theta[i]), -np.sin(theta[i])], [np.sin(theta[i]), np.cos(theta[i])]]) # 2x2 
        translation_matrix =np.vstack([translation_x[0][i], translation_y[0][i]]) # 2x1
        r_t_matrix=np.concatenate((rotation_matrix, translation_matrix),axis=1) # 2x3 

        # Stored as undistorted points
        velo_points_original[i][:2]= np.dot(r_t_matrix, velo_points_x_y[i])  

    undistorted_points=velo_points_original  

    return undistorted_points


def show_images(velo_points_path, angular_velocity_path, velocity_path, img_path):

    raw_image=cv2.imread(img_path) 
    distorted_points=data_utils.load_from_bin(velo_points_path)
    distorted_points_image =project_points(distorted_points, img_path)
    undistorted_velo_points=undistort_velo_points(velo_points_path, angular_velocity_path, velocity_path, img_path)
    undistoreted_points_to_project=project_points(undistorted_velo_points, img_path)
    
    f, axarr = plt.subplots(3,figsize=(10, 6), dpi=500)
    plt.xticks(fontsize=10) 
    axarr[0].imshow(raw_image)
    axarr[1].imshow(distorted_points_image)
    axarr[2].imshow(undistoreted_points_to_project)

    
    return plt.savefig("output"+img_path[-7:-4]+".png", dpi=300), plt.show()



def main():

    # Demonstrating angle relation.
    start_time_path, end_time_path, cam_trigger_time_path = get_time_files_path()
    frame_index_data, start_time_data, end_time_data, camera_trigger_data= calculate_timestamps(start_time_path, end_time_path, cam_trigger_time_path, num_of_frame=5)
    create_df_for_angle_calculation(frame_index_data, start_time_data, end_time_data, camera_trigger_data)

    # Printing out the related frames.
    frame_numbers=["037", "310", "320"]
    for i in frame_numbers:
        velo_points_path, img_path, angular_velocity_path, velocity_path = get_paths(frame_number=i)
        show_images(velo_points_path, angular_velocity_path, velocity_path, img_path)


if __name__=='__main__':
    main()

