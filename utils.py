import os, cv2, math
import numpy as np
from scipy import ndimage
from scipy.ndimage import convolve
from scipy.signal import medfilt2d
from skimage import color, io, segmentation
from skimage import color
import matplotlib.pyplot as plt
import networkx as nx
from scipy.spatial import distance
from skimage import img_as_float


def resizeImage(image):
    image = image[0:690, 180:1100]
    image = cv2.resize(image, (640, 480), interpolation=cv2.INTER_AREA)
    return image


def steerGaussFilterOrder2viz(image, theta, sigma):
    ## convert to radian angle
    rad_theta = -theta * (np.pi / 180)
    """[summary]
    This function implements the steerable filter
    of the second deriative of Gaussian function
    (X-Y separable version)
    Args:
        image ([ndarray]): [the input image MxNxP]
        theta ([int]): [the orientation]
        sigma ([int]): [the standard deviation of the Gaussian template]

    Return:
        output ([type]): [the response of derivative in the theta direction]
    """
    ###################### determine necessary filter ######################
    Wx = np.floor((8/2)*sigma)
    if Wx < 1:
        Wx = 1
    x = np.arange(-Wx, Wx+1)
    xx, yy = np.meshgrid(x, x)
    g0 = np.exp(-(xx**2 + yy**2) / (2*sigma**2)) / (sigma*np.sqrt(2*np.pi))
    G2a = -g0 / sigma**2 + g0 * xx**2 / pow(sigma, 4)
    G2b = g0 * xx * yy / pow(sigma, 4)
    G2c = -g0 / sigma**2 + g0 * yy**2 / pow(sigma, 4)
    # cv2.imwrite(f"G2a_{theta}.png", cv2.normalize(G2a, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1))
    # cv2.imwrite(f"G2b_{theta}.png", cv2.normalize(G2b, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1))
    # cv2.imwrite(f"G2c_{theta}.png", cv2.normalize(G2c, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1))

    G = pow(np.cos(rad_theta), 2)*G2a \
      + pow(np.sin(rad_theta), 2)*G2c \
      - 2*np.cos(rad_theta)*np.sin(rad_theta)*G2b
    # cv2.imwrite(f"G_{theta}.png", cv2.normalize(G, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1))

    I2a = ndimage.filters.convolve(\
        image, G2a, mode='nearest')
    I2b = ndimage.filters.convolve(\
        image, G2b, mode='nearest')
    I2c = ndimage.filters.convolve(\
        image, G2c, mode='nearest')

    # cv2.imwrite(f"I2a_{theta}.png", cv2.normalize(I2a, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1))
    # cv2.imwrite(f"I2b_{theta}.png", cv2.normalize(I2b, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1))
    # cv2.imwrite(f"I2c_{theta}.png", cv2.normalize(I2c, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1))

    J =   pow(np.cos(rad_theta), 2)*I2a \
        + pow(np.sin(rad_theta), 2)*I2c \
        - 2*np.cos(rad_theta)*np.sin(rad_theta)*I2b

    # cv2.imwrite(f"J_{theta}.png", cv2.normalize(J, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1))
    return J

def steerGaussFilterOrder2(image, theta, sigma):
    ## convert to radian angle
    rad_theta = -theta * (np.pi / 180)
    """[summary]
    This function implements the steerable filter
    of the second deriative of Gaussian function
    (X-Y separable version)
    Args:
        image ([ndarray]): [the input image MxNxP]
        theta ([int]): [the orientation]
        sigma ([int]): [the standard deviation of the Gaussian template]

    Return:
        output ([type]): [the response of derivative in the theta direction]
    """
    ###################### determine necessary filter ######################
    Wx = np.floor((8/2)*sigma)
    if Wx < 1:
        Wx = 1
    x = np.arange(-Wx, Wx+1)
    xx, yy = np.meshgrid(x, x)
    g0 = np.exp(-(xx**2 + yy**2) / (2*sigma**2)) / (sigma*np.sqrt(2*np.pi))
    G2a = -g0 / sigma**2 + g0 * xx**2 / pow(sigma, 4)
    G2b = g0 * xx * yy / pow(sigma, 4)
    G2c = -g0 / sigma**2 + g0 * yy**2 / pow(sigma, 4)

    G = pow(np.cos(rad_theta), 2)*G2a \
      + pow(np.sin(rad_theta), 2)*G2c \
      - 2*np.cos(rad_theta)*np.sin(rad_theta)*G2b

    I2a = ndimage.filters.convolve(\
        image, G2a, mode='nearest')
    I2b = ndimage.filters.convolve(\
        image, G2b, mode='nearest')
    I2c = ndimage.filters.convolve(\
        image, G2c, mode='nearest')

    J =   pow(np.cos(rad_theta), 2)*I2a \
        + pow(np.sin(rad_theta), 2)*I2c \
        - 2*np.cos(rad_theta)*np.sin(rad_theta)*I2b

    return J


def getBaseValue(line):
    """
    We already have formula y = ax + b
    or ax - y + b = 0
    extracting a, -1, b
    """
    slope = (line[3] - line[1]) / (line[2] - line[0])
    bias = line[1] - slope * line[0]
    base_value = np.array([slope, -1, bias])
    return base_value


def houghTransform(image):
    ## Normalize image to range 0-255
    normalize_image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1).astype(np.uint8)
    ## Getting the edege with Canny transform
    canny_image = cv2.Canny(normalize_image, threshold1=40, threshold2=80)
    ## Setting parameters for Hough Transform
    rho, theta, threshold, min_line_length, max_line_gap = 3, np.pi/180, 30, 20, 30
    color, thickness = [255, 0, 0], 1
    ## Extract lines from Hough Transform
    lines = cv2.HoughLinesP(canny_image, 
                            rho,
                            theta,
                            threshold,
                            np.array([]),
                            minLineLength=min_line_length,
                            maxLineGap=max_line_gap)
    if lines is None:
        output_line = [0, 0, 0, 0]
        status = 0
        return output_line, status
    ## Select the longest line
    elif len(lines) == 1:
        output_line = lines[0][0]
    else:
        len_array = np.zeros((len(lines), 1))
        for i in range(len(lines)):
            point_1 = lines[i][0][:2]
            point_2 = lines[i][0][2:]
            len_array[i] = np.sqrt(np.sum((point_1 - point_2)**2))
        index_max = np.argmax(len_array)
        output_line = lines[index_max][0]
    # Extend the line to the left and the right
    if len(output_line) == 0:
        status = 0
        output_line = 0
        return output_line, status
    else:
        # Get the base value of line and 
        base_value = getBaseValue(output_line)
        x_min, x_max = 0, image.shape[1]
        y_min, y_max = 0, image.shape[0]
        x2, y2 = int((y_max - base_value[2]) / base_value[0]), y_max
        x1, y1 = x_min, int(base_value[0] * x_min + base_value[2])
        output_line = x1, y1, x2, y2
        status = 1

        return output_line, status


def getGaussianFilter(shape=(3, 3), sigma=0.5):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    H = np.exp(-(pow(x, 2) + pow(y, 2)) / (2.*pow(sigma, 2)))
    H[H < np.finfo(H.dtype).eps*H.max()] = 0
    sumh = H.sum()
    if sumh != 0:
        H /= sumh
    return H


def detectAnomalies(image, sigma_s=12, normalize=False, status='original'):
    # convert to LAB space color
    if status == 'original':
        lab = image
    elif status == 'hsv':
        # rgb = color.hsv2rgb(image)
        lab = color.rgb2lab(image.astype('float'))
        # cv2.imwrite("hsv-lab.png", lab)
    elif status == 'rgb':
        lab = color.rgb2lab(image.astype('float'))
        # cv2.imwrite("rgb-lab.png", lab)
    # lab = image
    # get size of image
    height, width = lab.shape[:2]
    # get sigma (the standard deviation of Gaussian function)
    sigma = int(np.ceil(min(height, width) / sigma_s))
    # get kernel size
    kernel_size = 3*sigma + 1
    # get Gaussian filter
    gaussian_filter = getGaussianFilter((1, kernel_size), sigma)
    # get specific color channel in LAB space color
    L_channel, A_channel, B_channel = None, None, None
    L_channel = lab[..., 0]
    A_channel = lab[..., 1]
    B_channel = lab[..., 2]
    # apply gaussian filter to each color channel
    gaussian_L, gaussian_A, gaussian_B = None, None, None
    gaussian_L = convolve(L_channel, gaussian_filter, mode='nearest')
    # plt.title("Gaussian")
    # plt.imshow(gaussian_L, cmap='gray')
    # plt.show()
    gaussian_A = convolve(A_channel, gaussian_filter, mode='nearest')
    gaussian_B = convolve(B_channel, gaussian_filter, mode='nearest')
    
    # get result from Gaussian filter
    filtered_result = None
    filtered_result = pow(np.subtract(L_channel, gaussian_L), 2) \
                    + pow(np.subtract(A_channel, gaussian_A), 2) \
                    + pow(np.subtract(B_channel, gaussian_B), 2)

    # convert range result to 0-255
    filtered_result = cv2.normalize(filtered_result, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    # plt.title("Filter")
    # plt.imshow(filtered_result, cmap='gray')
    # plt.show()

    if normalize==True:
        normalized_result = cv2.normalize(filtered_result, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_8UC1)
        return normalized_result
    else:
        return filtered_result


def anomaliesRGBGenertor(image):
    ## convert image to HSV color
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    # cv2.imwrite("hsv.png", hsv_image)

    ## define blue color mask
    blue_lower = np.array([100, 150, 0], dtype=np.uint8)
    blue_upper = np.array([140, 255, 255], dtype=np.uint8)
    ## get mask
    mask = cv2.inRange(hsv_image, blue_lower, blue_upper)
    # change image to red where blue is found
    # hsv_image[mask > 0] = (0, 0, 0)
    # cv2.imwrite("hsv-filter.png", hsv_image)

    # detect anomalies in HSV image
    anomalies_HSV = detectAnomalies(hsv_image, sigma_s=4, normalize=True, status='hsv')
    # cv2.imwrite("hsv-anomalies.png", anomalies_HSV*255)
    ## convert input image to RGB color
    # rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    ## detect anomalies in input image
    anomalies_RGB = detectAnomalies(image, sigma_s=4, normalize=True, status='rgb')
    # cv2.imwrite("rgb-anomalies.png", anomalies_RGB*255)
    ## bitwise and two results: anomalies_HSV & anomalies_RGB
    anomalies_image = cv2.bitwise_or(anomalies_HSV, anomalies_RGB, mask=None)
    ## normalize anomly image to range(0, 1)
    anomalies_image = cv2.normalize(anomalies_image, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_8UC1)

    return anomalies_image



def prepare_images(image_path):
    input_image = io.imread(image_path)
    input_image = input_image[0:690, 180:1100]
    input_image = cv2.resize(input_image, (640, 480), interpolation=cv2.INTER_AREA)
    # Converting rgb image float
    rgb_image = img_as_float(input_image)
    # converting rgb image to CIE-Lab image
    lab_image = img_as_float(color.rgb2lab(input_image))
    # Converting rgb image to gray
    gray_image = img_as_float(color.rgb2gray(input_image))

    return rgb_image, lab_image, gray_image



def retrieve_node_distance(min_path, main_graph):
    distance = 0
    for i in range(1, len(min_path)):
        distance += main_graph[min_path[i-1]][min_path[i]]['weight']
    return distance


def compute_saliency_cost(w_bg, wCtr, smoothness):
    n = len(w_bg)
    A = np.zeros((n, n))
    b = np.zeros(n)
    for i in range(n):
        A[i, i] += 2*w_bg[i] + 2*wCtr[i]
        b[i] = 2*wCtr[i]
        for j in range(n):
            A[i, i] += 5*smoothness[i, j]
            A[i, j] -= 5*smoothness[i, j]
    x = np.linalg.solve(A, b)
    return x


def extract_saliency(gray_image, SEGMENTS_IMAGE, unique_clusters, wCtr):
    # rgb_image, lab_image, gray_image = prepare_images(image_path)
    # SEGMENTS_IMAGE, unique_clusters, wCtr = extract_SOD(rgb_image, lab_image)
    saliency = gray_image.copy()
    # optimal_saliency = compute_saliency_cost(w_bg, wCtr, smoothness)
    for cluster in unique_clusters:
        saliency[SEGMENTS_IMAGE == cluster] = wCtr[cluster]
    
    saliency_norm = cv2.normalize(saliency, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_8UC1)
    
    saliency_norm = np.uint8(saliency_norm)
    _, saliency_norm_binary = cv2.threshold(saliency_norm, 0, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(saliency_norm_binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    contours_image = np.zeros((saliency_norm_binary.shape[0], saliency_norm_binary.shape[1], 3))
    # filtering contours by areas
    filter_contours = [contours[i] for i in range(len(contours)) if 200 <= cv2.contourArea(contours[i]) <= 50000]
    cv2.drawContours(contours_image, filter_contours, -1, (0, 255, 0), 1)
    saliency_anomalies = cv2.cvtColor(np.float32(contours_image), cv2.COLOR_BGR2GRAY)
    saliency_anomalies[saliency_anomalies == 0.0] = 0
    saliency_anomalies[saliency_anomalies != 0.0] = 1
    saliency_anomalies = ndimage.binary_fill_holes(saliency_anomalies).astype(np.float32)
    saliency_anomalies = cv2.normalize(saliency_anomalies, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_8UC1)

    return saliency

def extract_SOD(rgb_image, lab_image, gray_image, disparity_map):
    h, w = rgb_image.shape[:2]
    # Applying the SLIC segmentation to the rgb image
    SEGMENTS_IMAGE = segmentation.slic(rgb_image, n_segments=200, compactness=10, sigma=1, enforce_connectivity=False)

    # Unique clusters in SLIC segmentation
    unique_clusters = np.unique(SEGMENTS_IMAGE)
    # Number of unique clusters in SLIC segmentation
    num_clusters = len(unique_clusters)
    # Horizontal coordinates ignore the boundary
    horizontal = np.c_[SEGMENTS_IMAGE[:-1,:].ravel(), SEGMENTS_IMAGE[1:,:].ravel()]
    # Vertical coordinates ignore boundary
    vertical = np.c_[SEGMENTS_IMAGE[:, :-1].ravel(), SEGMENTS_IMAGE[:, 1:].ravel()]
    # Stacking horizontal and vertical coordinates
    all_edges = np.vstack([vertical, horizontal])
    # Ignoring diagonal line
    all_edges = np.sort(all_edges[all_edges[:, 0] != all_edges[:, 1], :], axis=1)
    # Hashing all edges
    edges_hash = all_edges[:, 0] + num_clusters*all_edges[:, 1]
    # Getting edges
    edges = [[unique_clusters[x % num_clusters], unique_clusters[x // num_clusters]] for x in np.unique(edges_hash)]
    # Setting grid of input image
    grid_h, grid_w = np.mgrid[:h, :w]
    # Declaring some variable
    cluster_centers = dict()    # coordinates dictionary of the center of each cluster
    cluster_colors = dict()     # average color dictionary of each cluster
    bitwise_boundary = dict()   # δ(·) is 1 for superpixel on the image boundary and 0 otherwise
    cluster_disparity = dict()
    # Getting coordinates, average color and checking bitwise boundary of each cluster
    for cluster in unique_clusters:
        # Accessing the coordinates of cluster in original image
        h_pixels, w_pixels = grid_h[SEGMENTS_IMAGE == cluster], grid_w[SEGMENTS_IMAGE == cluster]
        # Getting the coordinate of center of cluster
        cluster_centers[cluster] = [h_pixels.mean(), w_pixels.mean()]
        # Getting the average color of each cluster
        cluster_colors[cluster] = np.mean(lab_image[SEGMENTS_IMAGE == cluster], axis=0)

        # Checking bitwise boundary
        if np.any(h_pixels == 0) or np.any(w_pixels == 0) or np.any(h_pixels == h-1) or np.any(w_pixels == w-1):
            bitwise_boundary[cluster] = 1
        else:
            bitwise_boundary[cluster] = 0
    # Creating a graph to assign distance of each pair of clusters
    main_graph = nx.Graph()
    # Computing color distance between each pair of pixels ignore boundary pixels
    for edge in edges:
        pixel_i, pixel_j = edge[0], edge[1]
        color_distance  = distance.euclidean(cluster_colors[pixel_i], cluster_colors[pixel_j])
        # disparity_distance = distance.euclidean(cluster_disparity[pixel_i], cluster_disparity[pixel_j])
        main_graph.add_edge(pixel_i, pixel_j, weight=color_distance)
    # del color_distance
    # Computing color distance between each pair of clusters on boundary
    for cluster_i in unique_clusters:
        for cluster_j in unique_clusters:
            if (bitwise_boundary[cluster_i] == 1) and (bitwise_boundary[cluster_j] == 1):
                color_distance = distance.euclidean(cluster_colors[cluster_i] , cluster_colors[cluster_j])
                # disparity_distance = distance.euclidean(cluster_disparity[cluster_i], cluster_disparity[cluster_j])
                main_graph.add_edge(cluster_i, cluster_j, weight=color_distance)
    
    # del color_distance
    # Getting min distance between every pair of clusters
    min_color_distance = nx.shortest_path(main_graph, weight='weight')
    # Declaring some variable
    geodesic = np.zeros((num_clusters, num_clusters), dtype=float)
    spatial = np.zeros((num_clusters, num_clusters), dtype=float)
    smoothness = np.zeros((num_clusters, num_clusters), dtype=float)
    adjacency = np.zeros((num_clusters, num_clusters), dtype=float)

    sigma_clr = 10.0     # sigma color region
    sigma_bndcon = 0.1   # sigma boundary connectivity
    sigma_spa = 1.0      # sigma spatial
    mu = 0.1             # small constant to erase noise in both background and foreground
    max_distance = math.sqrt(h**2 + w**2)

    for cluster_i in unique_clusters:
        for cluster_j in unique_clusters:
            if cluster_i == cluster_j:
                geodesic[cluster_i, cluster_j] = 0
                spatial[cluster_i, cluster_j] = 0
                smoothness[cluster_i, cluster_j] = 0
            else:
                geodesic[cluster_i, cluster_j] = retrieve_node_distance(min_color_distance[cluster_i][cluster_j], main_graph)
                spatial[cluster_i, cluster_j] = distance.euclidean(cluster_centers[cluster_i], cluster_centers[cluster_j]) / max_distance
                smoothness[cluster_i, cluster_j] = math.exp(-(geodesic[cluster_i, cluster_j])**2 / (2*sigma_clr**2)) + mu
             
    for edge in edges:
        pixel_i, pixel_j = edge[0], edge[1]
        adjacency[pixel_i, pixel_j] = 1
        adjacency[pixel_j, pixel_i] = 1
    
    for cluster_i in unique_clusters:
        for cluster_j in unique_clusters:
            smoothness[cluster_i, cluster_j] *= adjacency[cluster_i, cluster_j]
    
    # Declaring some needed variables
    area = dict()       # spanning area of each superpixel p
    len_bnd = dict()    # length along the boundary
    bnd_con = dict()    # boundary connectivity
    w_bg = dict()       # probability which is mapped from the boundary connectivity value of superpixel p
    Ctr = dict()        # superpixel's contrast against its surrounding
    wCtr = dict()       # background weighted contrast
    d_app, d_spa, w_spa = None, None, None

    for cluster_i in unique_clusters:
        area[cluster_i] = 0
        len_bnd[cluster_i] = 0
        Ctr[cluster_i] = 0
        for cluster_j in unique_clusters:
            d_app = geodesic[cluster_i, cluster_j]
            d_spa = spatial[cluster_i, cluster_j]
            w_spa = math.exp(-(d_spa**2) / (2*sigma_spa**2))
            area_i = math.exp(-(geodesic[cluster_i, cluster_j])**2 / (2*sigma_clr**2))
            area[cluster_i] += area_i
            len_bnd[cluster_i] += area_i * bitwise_boundary[cluster_j]
            Ctr[cluster_i] += d_app * w_spa
        bnd_con[cluster_i] = len_bnd[cluster_i] / math.sqrt(area[cluster_i])
        w_bg[cluster_i] = 1 - math.exp(-(bnd_con[cluster_i])**2 / (2*sigma_bndcon**2))

    for cluster_i in unique_clusters:
        wCtr[cluster_i] = 0
        for cluster_j in unique_clusters:
            d_app = geodesic[cluster_i, cluster_j]
            d_spa = spatial[cluster_i, cluster_j]
            w_spa = math.exp(-(d_spa**2) / (2*sigma_spa**2))
            wCtr[cluster_i] += d_app * w_spa * w_bg[cluster_j] 

    min_value, max_value = min(wCtr.values()), max(wCtr.values())
    for cluster in unique_clusters:
        wCtr[cluster] = (wCtr[cluster] - min_value) / (max_value - min_value)

    saliency = gray_image.copy()
    x = compute_saliency_cost(w_bg, wCtr, smoothness)
    for cluster in unique_clusters:
        saliency[SEGMENTS_IMAGE == cluster] = x[cluster]
    saliency_norm = cv2.normalize(saliency, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_8UC1)
    
    saliency_norm = np.uint8(saliency_norm)
    _, saliency_norm_binary = cv2.threshold(saliency_norm, 0, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(saliency_norm_binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    contours_image = np.zeros((saliency_norm_binary.shape[0], saliency_norm_binary.shape[1], 3))
    # filtering contours by areas
    filter_contours = [contours[i] for i in range(len(contours)) if 200 <= cv2.contourArea(contours[i]) <= 50000]
    cv2.drawContours(contours_image, filter_contours, -1, (0, 255, 0), 1)
    saliency_anomalies = cv2.cvtColor(np.float32(contours_image), cv2.COLOR_BGR2GRAY)
    saliency_anomalies[saliency_anomalies == 0.0] = 0
    saliency_anomalies[saliency_anomalies != 0.0] = 1
    saliency_anomalies = ndimage.binary_fill_holes(saliency_anomalies).astype(np.float32)
    saliency_anomalies = cv2.normalize(saliency_anomalies, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_8UC1)
    # saliency_anomalies = saliency_anomalies.astype(int)

    # return SEGMENTS_IMAGE, unique_clusters, wCtr
    return saliency_anomalies