import os, cv2, argparse
import os.path as osp
import numpy as np
from natsort import natsorted
from tqdm import tqdm
import scipy.optimize


def read_images(data_dir):
    """
    Reads the images from the data directory

    :param data_dir: Path to the data directory
    :type data_dir: str
    :return: List of images
    :return_format: [img_name, img, img_grayscale]
    :rtype: list
    """
    images = []
    try:
        for img_name in natsorted(os.listdir(data_dir)):
            img = cv2.imread(osp.join(data_dir, img_name))
            if img is not None:
                img_grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                images.append([img_name, img, img_grayscale])
    except Exception as e:
        print(e)

    return images

def find_world_coords(length, width, square_size):
    """
    Finds the world coordinates of the checkerboard corners

    :param length: length of the checkerboard
    :type length: int
    :param width: width of the checkerboard
    :type width: int
    :param square_size: size of the square
    :type square_size: float
    :return: World coordinates of the checkerboard corners 
    :return_format: numpy.ndarray([[x, y], [x, y], ...]) | Shape: (length * width, 2)
    :rtype: numpy.ndarray
    """
    world_coords_x, world_coords_y = np.meshgrid(range(length), range(width))
    world_coords = np.hstack((world_coords_x.reshape(-1, 1), world_coords_y.reshape(-1, 1)))
    world_coords = world_coords * square_size
    
    return world_coords # Shape: (length * width, 2)

def find_checkerboard_coords(images, pattern_size, data_dir):
    """
    Finds the checkerboard corners in the images

    :param images: List of images | Format: [img_name, img, img_grayscale]
    :type images: list
    :param pattern_size: Size of the checkerboard | Format: (width, height)
    :type pattern_size: tuple
    :param data_dir: Path to the data directory
    :type data_dir: str
    :return: Checkerboard corners | 
    :return_format: [[img_name, corners], [img_name, corners], ...]
    :rtype: list
    """
    corners = []
    results_dir = osp.join("Results", "Detected_Corners")
    if not osp.exists(results_dir): os.makedirs(results_dir)

    pbar = tqdm(enumerate(images), total=len(images), desc="Detecting Corners")
    for ind, image in pbar:
        image_copy = image[1].copy()
        ret, corner = cv2.findChessboardCorners(image[2], pattern_size, None)
        if ret:
            corner = np.squeeze(corner)
            # Save the image with the corners drawn
            cv2.drawChessboardCorners(image_copy, pattern_size, corner, ret)
            cv2.imwrite(osp.join(results_dir, image[0]), image_copy)
        else:
            continue

        corners.append([image[0], corner])
    
    # Format: [[img_name, corners], [img_name, corners], ...]
    # Corners: shape ((length * width), 1, 2) 
    return corners 

def find_homography(world_coords, img_coords):
    """
    Finds the homography matrix for the images

    :param world_coords: World coordinates of the checkerboard corners
    :type world_coords: numpy.ndarray
    :param img_coords: Image coordinates of the checkerboard corners
    :type img_coords: numpy.ndarray
    :raises ValueError: If the shape of world coordinates and image coordinates do not match
    :return: Homography matrix
    :rtype: numpy.ndarray
    """
    if world_coords.shape != img_coords.shape:
        raise ValueError("The shape of world coordinates and image coordinates must match.")
    
    A = []
    for i in range(world_coords.shape[0]):
        X1, Y1 = world_coords[i, :]
        x2, y2 = img_coords[i, :]
        
        A.append([-X1, -Y1, -1, 0, 0, 0, x2*X1, x2*Y1, x2])
        A.append([0, 0, 0, -X1, -Y1, -1, y2*X1, y2*Y1, y2])
    
    A = np.array(A)
    U, S, V = np.linalg.svd(A)
    H = np.reshape(V[-1], (3, 3))
    H = (1 / H.item(8)) * H  # Normalize so that h33 becomes 1
    return H


def find_homography_matrices(img_corners, world_coords):
    """
    Finds the homography matrices for the images

    :param img_corners: Checkerboard corners
    :type img_corners: list
    :param world_coords: World coordinates of the checkerboard corners
    :type world_coords: numpy.ndarray
    :return: Homography matrices
    :return_format: [[img_name, homography_matrix], [img_name, homography_matrix], ...]
    :rtype: list
    """
    homography_matrices = []
    for corner_info in img_corners:
        img_name = corner_info[0]
        corners = corner_info[1]
        H = find_homography(world_coords, corners)

        homography_matrices.append([img_name, H])
    
    # Format: [[img_name, homography_matrix], [img_name, homography_matrix], ...]
    return homography_matrices


def compute_b_vector(homography_matrices):
    """_summary_

    :param homography_matrices: Homography matrices 
    :type homography_matrices: list Format: [[img_name, homography_matrix], [img_name, homography_matrix], ...]
    :return: b vector
    :rtype: numpy.ndarray
    """
    # V matrix for each homography matrix
    V = [] # Shape is (2 * len(homography_matrices), 6)

    def compute_vij(H, i, j):
        """
        Computes the vij vector

        :param H: Transpose of Homography matrix
        :type H: numpy.ndarray
        :param i: Index
        :type i: int
        :param j: Index
        :type j: int
        :return: vij vector
        :return_format: numpy.ndarray
        :rtype: numpy.ndarray
        """
        vij = np.array([
            H[i][0] * H[j][0], 
            H[i][0] * H[j][1] + H[i][1] * H[j][0], 
            H[i][1] * H[j][1], 
            H[i][2] * H[j][0] + H[i][0] * H[j][2], 
            H[i][2] * H[j][1] + H[i][1] * H[j][2], 
            H[i][2] * H[j][2]
        ])
        return vij.T
    
    for homography_info in homography_matrices:
        H = homography_info[1]
        v11 = compute_vij(H.T, 0, 0).T
        v12 = compute_vij(H.T, 0, 1).T
        v22 = compute_vij(H.T, 1, 1).T
        v = np.vstack((v12, (v11 - v22)))
        V.append(v)

    # Convert V to numpy array of shape (2 * len(homography_matrices), 6)
    V = np.array(V).reshape(-1, 6)

    # Compute the b vector
    U, S, Vt = np.linalg.svd(V, full_matrices=True)
    b = Vt[-1, :]

    return b

    
def extract_intrinsic_matrix(b):
    """
    Computes the intrinsic matrix K from the b vector

    :param b: b vector
    :type b: numpy.ndarray
    :return: Intrinsic matrix
    :rtype: numpy.ndarray
    """
    v0 = (b[1] * b[3] - b[0] * b[4]) / (b[0] * b[2] - b[1] ** 2) # v0 Prindipal point in y direction
    arb_scale = b[5] - (b[3] ** 2 + v0 * (b[1] * b[3] - b[0] * b[4])) / b[0] # scale_factor
    u_scale_factor = np.sqrt(arb_scale / b[0]) # alpha
    v_scale_factor = np.sqrt((arb_scale * b[0]) / (b[0] * b[2] - b[1] ** 2)) # beta
    skew = -1 * b[1] * (u_scale_factor ** 2) * v_scale_factor / arb_scale # gamma
    u0 = (skew * v0 / v_scale_factor) - (b[3] * (u_scale_factor ** 2)) / arb_scale # u0 Principal point in x direction

    # Intrinsic matrix
    K = np.array([
        [u_scale_factor, skew, u0],
        [0, v_scale_factor, v0],
        [0, 0, 1]
    ])

    return K

def extract_extrinsic_matrix(K, homography_matrices):
    """
    Computes the extrinsic matrix R from the intrinsic matrix K and homography matrices

    :param K: Intrinsic matrix
    :type K: numpy.ndarray
    :param homography_matrices: Homography matrices
    :type homography_matrices: list
    :return: Extrinsic matrices
    :rtype: list Format: [[img_name, extrinsic_matrix], [img_name, extrinsic_matrix], ...]
     """
    R = []
    for homography_info in homography_matrices:
        H = homography_info[1]
        h1 = H[:, 0]
        h2 = H[:, 1]
        h3 = H[:, 2]
        scale_factor = 1 / np.linalg.norm(np.linalg.inv(K) @ h1, ord=2)
        r1 = (scale_factor * np.dot(np.linalg.inv(K), h1)).reshape(-1, 1)
        r2 = (scale_factor * np.dot(np.linalg.inv(K), h2)).reshape(-1, 1)
        t = (scale_factor * np.dot(np.linalg.inv(K), h3)).reshape(-1, 1)

        # Transformation matrix
        coord_transform_matrix = np.hstack((r1, r2, t))
        R.append([homography_info[0], coord_transform_matrix])
    return R


def get_optimization_parameters(A, distortion_vec):
    """
    Gets the optimization parameters
    """
    return np.array([A[0][0], A[0][1], A[1][1], A[0][2], A[1][2], distortion_vec.flatten()[0], distortion_vec.flatten()[1]])


def objective_function(x0, R, world_coords, img_corners):
    """
    Objective function for optimization
    """
    total_error, _, _ = compute_projection_error(x0, R, world_coords, img_corners)
    return total_error

def compute_projection_error(x0, R, world_coords, img_corners):
    """
    Computes the reprojection error

    :param x0: Parameters for optimization
    :type x0: numpy.ndarray
    :param R: Extrinsic matrices
    :type R: list
    :param world_coords: World coordinates of the checkerboard corners
    :type world_coords: numpy.ndarray
    :param img_corners: Checkerboard corners
    :type img_corners: list
    :return: Reprojection error
    :rtype: numpy.ndarray
    """
    u0 = x0[3]
    v0 = x0[4]
    k1 = x0[5]
    k2 = x0[6]

    # Compute the intrinsic matrix
    A = np.array([
        [x0[0], x0[1], u0],
        [0, x0[2], v0],
        [0, 0, 1]
    ])

    total_error = 0
    all_reprojected_corners = []
    individual_img_errors = []

    for i, corner_info in enumerate(img_corners):
        image_name = corner_info[0]
        extrinsic_matrix = R[i][1]
        image_error= 0
        reprojected_corners = []

        total_transformation_matrix = A @ extrinsic_matrix

        for j, corner in enumerate(corner_info[1]):

            # Fetch the ground truth image coordinates
            image_gt_corner = corner.reshape(-1, 1)
            image_gt_corner = np.vstack((image_gt_corner, 1))

            # Convert the world coordinates to homogeneous coordinates
            world_coord = np.hstack((world_coords[j], 1)).reshape(-1, 1)

            # Camera coordinate
            camera_coord = extrinsic_matrix @ world_coord
            x = camera_coord[0] / camera_coord[2]
            y = camera_coord[1] / camera_coord[2]

            # Pixel coordinate
            pixel_coord = total_transformation_matrix @ world_coord
            u = pixel_coord[0] / pixel_coord[2]
            v = pixel_coord[1] / pixel_coord[2]

            u_hat = u + (u - u0) * (k1 * (x**2 + y**2) + k2 * (x**2 + y**2)**2)
            v_hat = v + (v - v0) * (k1 * (x**2 + y**2) + k2 * (x**2 + y**2)**2)

            image_projected_corner = np.array([u_hat, v_hat]).reshape(-1, 1)
            image_projected_corner = np.vstack((image_projected_corner, 1))

            reprojected_corners.append(image_projected_corner)

            image_error += np.linalg.norm(image_projected_corner - image_gt_corner, ord=2)
        
        image_error = image_error / len(corner_info[1])
        individual_img_errors.append([image_name, image_error])

        total_error += image_error / len(img_corners)
        all_reprojected_corners.append(reprojected_corners)

    # return np.array(total_error)
    return np.array([total_error, 0, 0, 0, 0, 0, 0]), all_reprojected_corners, individual_img_errors


def undistort_image(image, A, distortion):
    """
    Undistorts the image
    """
    dist = distortion
    h, w = image.shape[:2]
    dst = cv2.undistort(image, A, dist)
    return dst

def log_error(error):
    """
    Logs the error
    """
    with open("error_logs.txt", "w") as f:
        for err in error:
            f.write(f"{err[0]}: Before Error: {err[1]}, After Error: {err[2]}\n") 


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", default="Data", help="Path to calibration images")

    args = parser.parse_args()
    data = args.data

    # Define pattern size
    PATTERN_SIZE = (9, 6)

    # Read the calibration images
    calibration_images = read_images(osp.join("Data", "Calibration_Imgs"))

    # Find the checkerboard world coordinates
    world_coordinates = find_world_coords(PATTERN_SIZE[0], PATTERN_SIZE[1], 21.5) # 9x6 checkerboard with 21.5mm square size

    # Find the checkerboard corners for images in dataset
    img_corners = find_checkerboard_coords(calibration_images, PATTERN_SIZE, data)

    # Find homography matrices for the images
    homography_matrices = find_homography_matrices(img_corners, world_coordinates)

    # Compute the B matrix
    vec_b = compute_b_vector(homography_matrices)

    # Compute intrinsic matrix
    A = extract_intrinsic_matrix(vec_b)
    print(f"Initial Intrinsic Matrix:\n{A}\n")

    # Compute the extrinsic matrix
    R = extract_extrinsic_matrix(A, homography_matrices) # Format: [[img_name, extrinsic_matrix], [img_name, extrinsic_matrix], ...]

    # Initial distortion estimates
    distortion = np.array([0, 0]).reshape(-1, 1)

    # Get optimization params:
    x0 = get_optimization_parameters(A, distortion)
    print(f"Initial Optimization Parameters: {x0}\n")

    # Optimize the intrinsic matrix
    # x = scipy.optimize.minimize(fun=objective_function, x0=x0, method="Powell", args=(R, world_coordinates, img_corners))
    x = scipy.optimize.least_squares(fun=objective_function, x0=x0, method="lm", args=(R, world_coordinates, img_corners), verbose=2)
    _u_scale_factor, _arb_scale, _v_scale_factor, _u0, _v0, _k1, _k2 = x.x

    # Compute the optimized intrinsic matrix
    A_optimized = np.array([
        [_u_scale_factor, _arb_scale, _u0],
        [0, _v_scale_factor, _v0],
        [0, 0, 1]
    ])

    distortion_optimized = np.array([_k1, _k2, 0, 0, 0])
    print(f"Optimized Intrinsic Matrix:\n{A_optimized}")

    # Compute before and after reprojection error for individual images
    before_reprojection_error, _, before_individual_image_error = compute_projection_error(x0, R, world_coordinates, img_corners)
    after_reprojection_error, reprojected_points, after_individual_image_error = compute_projection_error(x.x, R, world_coordinates, img_corners)

    # Visualize the reprojected corners using cv2.undistor
    results_dir = osp.join("Results", "Reprojected_Corners")
    if not osp.exists(results_dir): os.makedirs(results_dir)

    for i, img_info in tqdm(enumerate(calibration_images)):
        img_name = img_info[0]
        img_copy = img_info[1].copy()
        reprojected_img = undistort_image(img_copy, A_optimized, distortion_optimized)
        
        for corner in reprojected_points[i]:
            cv2.circle(reprojected_img, (int(corner[0]), int(corner[1])), 11, (0, 0, 255), 4)
            cv2.circle(reprojected_img, (int(corner[0]), int(corner[1])), 3, (0, 255, 0), -1)

        cv2.imwrite(osp.join(results_dir, img_name), reprojected_img)
    
    # Log the reprojection error for each image
    error_logs = []
    for i, error in enumerate(before_individual_image_error):
        error_logs.append([error[0], error[1], after_individual_image_error[i][1]])
    
    log_error(error_logs)

if __name__ == "__main__":
    main()