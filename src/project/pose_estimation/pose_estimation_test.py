"""
With this script is possible to estimate the the picking pose of target point.

"""
import cv2
import zivid
from pathlib import Path
from typing import Tuple
import numpy as np
from nptyping import Floating, NDArray, Shape
import open3d as o3d


def _create_open3d_point_cloud(
    xyz: np.ndarray[Shape["*, *, 3"], Floating], rgba: np.ndarray[Shape["*, *, 3"], Floating]
) -> o3d.geometry.PointCloud:
    """Create a point cloud in Open3D format from NumPy array.

    Args:"
        rgba: A numpy array with RGBA values of a 2D image
        xyz: A numpy array of X, Y and Z point cloud coordinates

    Returns:
        refined_point_cloud_open3d: Point cloud in Open3D format without Nans or non finite values

    """
    xyz_reshaped = np.nan_to_num(xyz).reshape(-1, 3)
    rgb = rgba[:, :, 0:3].reshape(-1, 3)

    point_cloud_open3d = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz_reshaped))
    point_cloud_open3d.colors = o3d.utility.Vector3dVector(rgb / 255)

    refined_point_cloud_open3d = o3d.geometry.PointCloud.remove_non_finite_points(
        point_cloud_open3d, remove_nan=True, remove_infinite=True
    )
    return refined_point_cloud_open3d


def _get_transformation_matrix(
    u_matrix: np.ndarray[Shape["3, 3"], Floating], point: np.ndarray[Shape["3"], Floating]
) -> np.ndarray[Shape["4, 4"], Floating]:
    """create a 4x4 transformation matrix to transform a corrdinate frame,
    where Z is aligned with a unit vector and the translation is given by a point

    Args:
        u_matrix: [3,3] numpy array with U matrix
        point: A numpy array with a point with the translate information

    Returns:
        transformation_matrix: A numpy array with a 4x4 transformation matrix

    """
    unit_vector = u_matrix[:3,2]
    transform = np.identity(4)
    unit_x = np.array((1,0,0))
    y_axis = np.cross(unit_x,-unit_vector)
    x_axis = np.cross(y_axis,unit_vector)
    transform[:3,0] = x_axis
    transform[:3,1] = y_axis
    transform[:3,2] = unit_vector
    transform[:3,3] = point

    return transform


def _get_z_rotation_matrix(theta):
    """create a 4x4 transformation matrix that is a rotation over z-axis for a given angle

    Args:
        theta: An angle that defines a rotation on Z-axis in radians

    Returns:
        rotation: A numpy array with a 4x4 transformation matrix

    """
    rotation = np.array(
        [[np.cos(theta), -np.sin(theta), 0, 0], [np.sin(theta), np.cos(theta), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    )
    return rotation


def _display_point_cloud(
    xyz: np.ndarray[Shape["*, *, 3"], Floating], rgba: np.ndarray[Shape["*, *, 3"], Floating]
) -> None:
    """Display point cloud provided from 'xyz' with colors from 'rgb'.

    Args:
        rgba: A numpy array with RGBA values of a 2D image
        xyz: A numpy array of X, Y and Z point cloud coordinates

    """
    xyz_reshaped = np.nan_to_num(xyz).reshape(-1, 3)
    rgb = rgba[:, :, 0:3].reshape(-1, 3)

    point_cloud_open3d = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz_reshaped))
    point_cloud_open3d.colors = o3d.utility.Vector3dVector(rgb / 255)

    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window()
    visualizer.add_geometry(point_cloud_open3d)

    visualizer.get_render_option().point_size = 1
    visualizer.get_render_option().show_coordinate_frame = True
    visualizer.get_view_control().set_front([0, 0, -1])
    visualizer.get_view_control().set_up([0, -1, 0])

    visualizer.run()
    visualizer.destroy_window()


def _display_point_cloud_with_coordinate_frame_on_a_point(
    point_cloud_open3d: o3d.geometry.PointCloud,
    u_matrix: np.ndarray[Shape["3, 3"], Floating],
    point: np.ndarray[Shape["3"], Floating],
    z_rotation: float,
) -> None:
    """Create a mesh of a coordinate frame and visualize it with the point cloud
    of a checkerboard and the checkerboard coordinate frame.

    Args:
        point_cloud_open3d: An Open3d point cloud of a checkerboard
        u_matrix: A numpy array with a U Matrix that is a normal of a plane
        point: A numpy array with a coordinates of a point in the plane
        z_rotation: An angle that defines a rotation on Z-axis in radians

    Returns:
        pose: A numpy array with the pose of the coordinate frame

    """
    transform = _get_transformation_matrix(u_matrix=u_matrix, point=point)
    rotation_matrix = _get_z_rotation_matrix(z_rotation)
    pose = transform @ rotation_matrix

    coord_frame_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=30)
    coord_frame_mesh.transform(pose)

    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window()
    visualizer.add_geometry(point_cloud_open3d)
    visualizer.add_geometry(coord_frame_mesh)
    visualizer.run()
    visualizer.destroy_window()

    return pose

def _display_point_cloud_with_coordinate_frame_on_two_points(
    point_cloud_open3d: o3d.geometry.PointCloud,
    u_matrix: np.ndarray[Shape["3, 3"], Floating],
    u_matrix_modified: np.ndarray[Shape["3, 3"], Floating],
    point: np.ndarray[Shape["3"], Floating],
    z_rotation: float,
) -> None:
    """Create a mesh of a coordinate frame and visualize it with the point cloud
    of a checkerboard and the checkerboard coordinate frame.

    Args:
        point_cloud_open3d: An Open3d point cloud of a checkerboard
        u_matrix: A numpy array with a U Matrix that is a normal of a plane
        u_matrix_modified: A numpy array with a modified Matrix
        point: A numpy array with a coordinates of a point in the plane
        z_rotation: An angle that defines a rotation on Z-axis in radians

    Returns:
        pose: A numpy array with the pose of the coordinate frame

    """
    transform = _get_transformation_matrix(u_matrix=u_matrix, point=point)
    rotation_matrix = _get_z_rotation_matrix(z_rotation)
    pose = transform @ rotation_matrix

    coord_frame_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=20)
    coord_frame_mesh.transform(pose)

    transform_modified = _get_transformation_matrix(u_matrix=u_matrix_modified, point=point)
    rotation_matrix_modified = _get_z_rotation_matrix(z_rotation)
    pose_modified = transform_modified @ rotation_matrix_modified

    coord_frame_mesh_modified = o3d.geometry.TriangleMesh.create_coordinate_frame(size=30)
    coord_frame_mesh_modified.transform(pose_modified)

    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window()
    visualizer.add_geometry(point_cloud_open3d)
    visualizer.add_geometry(coord_frame_mesh)
    visualizer.add_geometry(coord_frame_mesh_modified)
    visualizer.run()
    visualizer.destroy_window()

    return pose

def _roi_between_two_spheres(
    xyz: np.ndarray[Shape["*, *, 3"], Floating],
    centroid: np.ndarray[Shape["3"], Floating],
    inner_radius_threshold: int,
    outer_radius_threshold: int,
) -> np.ndarray[Shape["*, *, 3"], Floating]:
    """Filters out the data outside the region of interest defined by the checkerboard centroid.

    Args:
        xyz: A numpy array of X, Y and Z point cloud coordinates
        centroid: A numpy array of X, Y and Z central point
        inner_radius_threshold: An interger that defines the radius of the inner sphere
        outer_radius_threshold: An interger that defines the radius of the outer sphere

    Returns:
        xyz: A numpy array of X, Y and Z point cloud coordinates within the region of interest

    """
    xyz_filtered = xyz.copy()
    radius = np.linalg.norm(centroid - xyz_filtered, axis=2)
    xyz_filtered[radius < inner_radius_threshold] = np.nan
    xyz_filtered[radius > outer_radius_threshold] = np.nan

    return xyz_filtered


def _plane_fit(
    points: NDArray[Shape["*, *, 3"], Floating]
) -> Tuple[NDArray[Shape["3, 3"], Floating], float, NDArray[Shape["3"], Floating]]:
    """plane fitting with Singular Value Decomposition.

    Args:
        points: points that lie in the plane

    Returns:
        u_matrix: NDArray[3,3] U matrix of SVD, singular value decomposition
        point_in_plane: NDArray[3] point_in_plane

    """
    points_no_nan = points[~np.isnan(points).any(axis=2)]
    point_in_plane = points_no_nan.mean(axis=0)
    distance_to_mean_point = points_no_nan - point_in_plane[np.newaxis, :]
    M = np.dot(distance_to_mean_point.T, distance_to_mean_point)
    u = np.linalg.svd(M)[0]
    return (u, point_in_plane)


def main():

    with zivid.Application():
        data_file = "C:/ProgramData/Zivid/BinWithArucoMarker.zdf"
        print(f"Reading ZDF frame from file: {data_file}")
        frame = zivid.Frame(data_file)
        point_cloud = frame.point_cloud()
        xyz = point_cloud.copy_data("xyz")
        rgba = point_cloud.copy_data("rgba")

        """
        Getting target point as Index and find U matrix of the plane near target point 
        """
        # Get target point using 2D image index[height, width]
        # Target height and width
        target_height = 600
        target_width = 960
        target_point = xyz[target_height, target_width]

        # masked region nearby target point
        mask_region = _roi_between_two_spheres(
            xyz,
            target_point,
            inner_radius_threshold=0,
            outer_radius_threshold=10, # you can change the size of plane
        )
        # Display masked region of point cloud
        _display_point_cloud(mask_region, rgba)

        # Fiting a plane on all given points
        u_matrix, point_in_plane = _plane_fit(mask_region)
        print(f"\nPlane U matrix = \n{u_matrix}\nZ vector = \n{u_matrix[:,2]}\nA point in plane = {point_in_plane}")
        
        # drawing a coordinate frame on the point cloud and inspecting it
        open3d_point_cloud = _create_open3d_point_cloud(xyz, rgba) # you can change "xyz" or "mask_region"
        picking_pose = _display_point_cloud_with_coordinate_frame_on_a_point(
            point_cloud_open3d=open3d_point_cloud,
            u_matrix=u_matrix,
            point=point_in_plane,
            z_rotation=0, # 0'=0 , 90' = 1.5707963268
        )
        
        print(f"The picking pose of Target point is :\n{picking_pose}")

        """
        Changing X axis & Y axis direction! it mean target coordinate is rotating by Z-Axis. 
        """
        # Making x-direction, vx = vetor x, 
        vx = xyz[target_height, target_width+10] - xyz[target_height, target_width-10]
        # copy Z(Normal)-direction!
        vz = u_matrix[:,2]
        # find vx[2] values using vx and vz is perpendicular!
        vx[2] = -(vx[0]*vz[0]+vx[1]*vz[1])/vz[2]
        vx = vx / np.linalg.norm(vx) 
        # find vy using vector cross
        vy = np.cross(vx, vz)
        vy = vy / np.linalg.norm(vy)
        
        u_matrix_modified = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]], Floating)
        u_matrix_modified[:, 0] = vx
        u_matrix_modified[:, 1] = vy
        u_matrix_modified[:, 2] = vz
        
        picking_pose = _display_point_cloud_with_coordinate_frame_on_two_points(
            point_cloud_open3d=open3d_point_cloud,
            u_matrix=u_matrix,
            u_matrix_modified=u_matrix_modified,
            point=point_in_plane,
            z_rotation=0, # 0'=0 , 90' = 1.5707963268
        )
        print(f"The picking pose of Target point is :\n{picking_pose}")

if __name__ == "__main__":
    main()
