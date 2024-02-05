import os, cv2, argparse
import os.path as osp
import numpy as np
from natsort import natsorted
from tqdm import tqdm


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

    ########## Sanity check ##########
    # print("World Coordinates X:")
    # print(world_coords_x)
    # print(f"World Coordinates X Shape: {world_coords_x.shape}\n")
    # print("World Coordinates Y:") 
    # print(world_coords_y)
    ########## Sanity check ##########

    world_coords = np.hstack((world_coords_x.reshape(-1, 1), world_coords_y.reshape(-1, 1)))
    # print(f"World Coordinates Shape: {world_coords.shape}")
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
    results_dir = osp.join(data_dir, "Results")
    if not osp.exists(results_dir): os.makedirs(results_dir)

    pbar = tqdm(enumerate(images), total=len(images), desc="Detecting Corners")
    for ind, image in pbar:
        ret, corner = cv2.findChessboardCorners(image[2], pattern_size, None)
        if ret:
            corner = np.squeeze(corner)
            # Save the image with the corners drawn
            cv2.drawChessboardCorners(image[1], pattern_size, corner, ret)
            cv2.imwrite(osp.join(results_dir, image[0]), image[1])
        else:
            continue

        corners.append([image[0], corner])
    
    # Format: [[img_name, corners], [img_name, corners], ...]
    # Corners: shape ((length * width), 1, 2) 
    return corners 


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
        H, _ = cv2.findHomography(world_coords, corners)
        homography_matrices.append([img_name, H])
    
    # Format: [[img_name, homography_matrix], [img_name, homography_matrix], ...]
    return homography_matrices


def compute_b_vector(homography_matrices):
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
        # print(f"Shape of v for image {homography_info[0]} --> {v.shape}")
        V.append(v)

    # Convert V to numpy array of shape (2 * len(homography_matrices), 6)
    V = np.array(V).reshape(-1, 6)
    print(f"Shape of V: {V.shape}")

    # Compute the b vector
    U, S, Vt = np.linalg.svd(V, full_matrices=True)
    b = Vt[-1, :]
    # b = Vt.T[:, -1]
    print(f"Shape of b: {b.shape}")
    print(f"b: {b}")

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
    arb_scale = b[5] - (b[3] ** 2 + v0 * (b[1] * b[3] - b[0] * b[4])) / b[0] # scale_factorbda
    u_scale_factor = np.sqrt(arb_scale / b[0]) # alpha
    v_scale_factor = np.sqrt((arb_scale * b[0]) / (b[0] * b[2] - b[1] ** 2)) # beta
    skew = -1 * b[1] * (u_scale_factor ** 2) * v_scale_factor / arb_scale # gamma
    u0 = (skew * v0 / v_scale_factor) - (b[3] * (u_scale_factor ** 2)) / arb_scale # u0 Prindipal point in x direction

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", default="Data", help="Path to calibration images")

    args = parser.parse_args()
    data = args.data

    # Define pattern size
    PATTERN_SIZE = (9, 6)

    # Read the calibration images
    calibration_images = read_images(osp.join("Data", "Calibration_Imgs"))

    ########## Sanity check ##########
    # cv2.imwrite("Grayscale_Image.png", calibration_images[0][2])

    # Find the checkerboard world coordinates
    world_coordinates = find_world_coords(PATTERN_SIZE[0], PATTERN_SIZE[1], 21.5) # 9x6 checkerboard with 21.5mm square size
    # print(world_coordinates)


    # Find the checkerboard corners for images in dataset
    img_corners = find_checkerboard_coords(calibration_images, PATTERN_SIZE, data)

    # ######### Sanity check ##########
    # for ind, corner_info in enumerate(img_corners):
    #     image_name = corner_info[0]
    #     if image_name == calibration_images[ind][0]:
    #         print(f"Image number = {ind + 1}")
    #         print(f"Mapping is correct for image {image_name}")
    #         print(f"Corners detected for image {image_name}: {corner_info[1].shape}\n")
    #         if ind == len(img_corners) - 1: print(f"Corner matrix: {corner_info[1]}")
    # ######### Sanity check ##########

    # Find homography matrices for the images
    homography_matrices = find_homography_matrices(img_corners, world_coordinates)


    # ########## Sanity check ##########
    # print(f"Number of homography matrices: {len(homography_matrices)}")
    # for ind, homography_info in enumerate(homography_matrices):
    #     image_name = homography_info[0]
    #     if image_name == calibration_images[ind][0]:
    #         print(f"Image number = {ind + 1}")
    #         print(f"Mapping is correct for image {image_name}")
    #         print(f"Homography matrix for image {image_name}: {homography_info[1]} with shape {homography_info[1].shape}\n")
    # ########## Sanity check ##########

    # Compute the B matrix
    vec_b = compute_b_vector(homography_matrices)

    # Compute intrinsic matrix
    K = extract_intrinsic_matrix(vec_b)
    print(K)

    # Compute the extrinsic matrix
    R = extract_extrinsic_matrix(K, homography_matrices)





if __name__ == "__main__":
    main()