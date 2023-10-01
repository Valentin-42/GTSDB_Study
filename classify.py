import os 
import cv2
import matplotlib.pyplot as plt
import pandas
import shutil
import numpy as np
from skimage.feature import hog
from sklearn.cluster import KMeans

# this function takes directory of images and 
# For all image in a folder, get average size, color and shape
def get_average_size_color_shape(img_dir) :
    # get the average size of the images
    img_size = []
    img_color = []
    img_shape = []
    for img in os.listdir(img_dir) :
        img = cv2.imread(os.path.join(img_dir, img))
        img_size.append(img.shape)
        img_color.append(img.mean(axis=0).mean(axis=0))
        img_shape.append(img.shape)
    img_size = np.array(img_size)
    img_color = np.array(img_color)
    img_shape = np.array(img_shape)
    print("Average size : {}".format(img_size.mean(axis=0)))
    print("Average color : {}".format(img_color.mean(axis=0)))
    print("Average shape : {}".format(img_shape.mean(axis=0)))
    print(f"Done {len(os.listdir(img_dir))} images")
    return img_size, img_color, img_shape

def create_fld() :
    #create a classify folder
    # remove it if it already exists
    if os.path.exists("./classify/") :
        shutil.rmtree("./classify/")
    os.mkdir("./classify/")

    return "./classify/"

def extract_ROIS(img_path, label, output_fld):
    # extract each object in ROIs
    img_name = img_path.split("/")[-1]
    i = 0
    show=False
    for _, row in label.iterrows() :
        # extract the ROI
        x_center = int(row[1])
        y_center = int(row[2])
        width = int(row[3])
        height = int(row[4])
        print(img_path)
        img = cv2.imread(img_path)
        sign = img[(y_center - height//2):(y_center + height//2), (x_center - width//2):(x_center + width//2)]
        
        if show==True :
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.subplot(1,2,1).imshow(img)
            plt.subplot(1,2,2).imshow(sign)        
            
            plt.show()
            key = input("Press [Enter] : continue [q] : quit [p] : finish")
            if key == "q" :
                exit()
            if key == "p" :
                print("gp")
                show=False
            plt.close()


        # save the ROI
        cv2.imwrite(output_fld + img_name.split(".")[0] + "_" + str(i) + ".png", sign)
        print(output_fld + img_name.split(".")[0] + "_" + str(i) + ".png")
        i+=1

def extract_hog_features(image_path):
    show = False
    # Convert the image to grayscale
    image = cv2.imread(image_path)

    # Split the color image into its Red, Green, and Blue channels
    blue_channel, green_channel, red_channel = cv2.split(image)

    # Calculate the grayscale image using the specified formula
    grayscale_image = 0.5 * red_channel + 0.5 * blue_channel + 0.0 * green_channel

    # Convert the grayscale image to the correct data type (uint8)
    gray_image = grayscale_image.astype('uint8')
   
    kernel_right   = np.array([[0, 1, 2],
                               [1, 0, 1],
                               [2, 1, 0]])

    kernel_left    = np.array([[2, 0, 2],
                               [0, 0, 0],
                               [2, 0, 2]])

    # Compute the diagonal gradients
    depth = 1
    gradient_right = gray_image
    gradient_left = gray_image
    for i in range(depth) : 
        gradient_right = cv2.filter2D(gradient_right, cv2.CV_64F,kernel_right)
        gradient_left  = cv2.filter2D(gradient_left, cv2.CV_64F,kernel_left)
    
    gradient_left = np.sqrt(gradient_left**2)
    gradient_right = np.sqrt(gradient_right**2)
    gradient_left = cv2.normalize(gradient_left, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    gradient_right = cv2.normalize(gradient_right, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')

    left_arrow = np.argmax(gradient_left)
    right_arrow = np.argmax(gradient_right)

    arrow_length = 50  # Length of the arrow
    arrow_scale = 0.1  # Scale factor to adjust the arrow length
    x_center, y_center = gradient_left.shape[1] // 2, gradient_left.shape[0] // 2  # Center of the image
    x_end = x_center + int(arrow_length * arrow_scale * np.cos(left_arrow))
    y_end = y_center + int(arrow_length * arrow_scale * np.sin(left_arrow))

    x_center2, y_center2 = gradient_right.shape[1] // 2, gradient_right.shape[0] // 2  # Center of the image
    x_end_2 = x_center2 + int(arrow_length * arrow_scale * np.cos(right_arrow))
    y_end_2 = y_center2 + int(arrow_length * arrow_scale * np.sin(right_arrow))
    if show :
        plt.axis("off")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.subplot(1,6,1).imshow(image, cmap="gray")
        plt.subplot(1,6,2).imshow(gray_image, cmap="gray")
        plt.subplot(1,6,3).imshow(gradient_right, cmap="gray")
        plt.subplot(1,6,4).imshow(gradient_left, cmap="gray")
        plt.subplot(1,6,5).imshow(image, cmap="gray")
        plt.subplot(1,6,5).arrow(x_center, y_center, x_end - x_center, y_end - y_center, head_width=5, head_length=5, fc='r', ec='r')  # Adjust parameters as needed
        # plt.subplot(1,6,5).arrow(x_center2, y_center2, x_end_2 - x_center2, y_end_2 - y_center2, head_width=5, head_length=5, fc='r', ec='r')  # Adjust parameters as needed
        
        plt.show()
        k = input("Press Enter to continue...")
        plt.close()
        if k == "q" :
            exit()
    

    features = hog(
        gray_image,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2,2),
        block_norm='L2-Hys',
        visualize=False,
        transform_sqrt=True,
    )

    return features

def cluster(feature_vector) :

    # check length of feature vector
    print("Number of images : ", len(feature_vector))
    length_inside = [len(x) for x in feature_vector]
    print("Number of feature per image : ", length_inside)

    # Create a feature matrix by stacking the pairs (row:image index, col:feature vector)
    feature_matrix = np.vstack(feature_vector)
    feature_matrix = np.array(feature_matrix, dtype=np.float32)
    
    # Apply K-means clustering
    num_clusters = 2
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(feature_matrix)

    cluster_labels = kmeans.labels_
    return cluster_labels

def color_based_feature_extraction(img_path, label) :

    show=False
    img_name = img_path.split("/")[-1]
    feature_w = 30 
    feature_h = 30  

    feature_vector = []
    signs = []
    idx = 2
    for _, row in label.iterrows() :
        # extract the ROI
        x_center = int(row[1])
        y_center = int(row[2])
        width = int(row[3])
        height = int(row[4])

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        sign = img[(y_center - height//2):(y_center + height//2), (x_center - width//2):(x_center + width//2)]
        sign_top_right_cell = img[(y_center - height//3):(y_center + height//3), (x_center - width//3):(x_center + width//3)]
        # sign_top_right_cell = img[(y_center - height//3):(y_center), (x_center):(x_center + width//3)]
        # get number of red pixels

        max_red = get_max_pixel_value(sign_top_right_cell,'red')
        max_blue = get_max_pixel_value(sign_top_right_cell,'blue')
        max_green = get_max_pixel_value(sign_top_right_cell,'green')

        red_channel   = (sign_top_right_cell[:,:,0] ) / max_red # btw 0 and 255
        blue_channel  = (sign_top_right_cell[:,:,2] ) / max_blue
        green_channel = (sign_top_right_cell[:,:,1] ) / max_green

        red_channel = cv2.resize(red_channel, (feature_w, feature_h))
        blue_channel = cv2.resize(blue_channel, (feature_w, feature_h))
        green_channel = cv2.resize(blue_channel, (feature_w, feature_h))

        n_im = ((1)*red_channel + (0)*blue_channel + (0)*green_channel)

        sobel_kernel = np.array([[-1,  0,  1],
                                 [-2,  0,  2],
                                 [-1,  0,  1]])
        
        diagonal_edges = cv2.filter2D(n_im, -1, sobel_kernel)


        img = cv2.imread(img_path)
        plt.subplot(1,len(label.columns),1).imshow(img)
        plt.subplot(1,len(label.columns),idx).imshow(sign)
        idx +=1
        if show==True :
            plt.show()
            key = input("Press [Enter] : continue [q] : quit [p] : finish")
            plt.close()
            if key == "q" :
                exit()
        else :
            # plt.savefig("./classify/" + img_name.split(".")[0] + "_" + str(len(feature_vector)) + ".png")
            # feature_vector.append((red_channel.flatten(), blue_channel.flatten(), green_channel.flatten()))
            
            # feature_vector.append((red_channel+blue_channel+green_channel).flatten())
            feature_vector.append(diagonal_edges.flatten())
            signs.append(sign)


    return feature_vector,signs

def get_max_pixel_value(img, color) :
    max = 0

    if color == 'red' :
        c = 0
    elif color == 'blue' :
        c = 1
    elif color == 'green' :
        c = 2

    for i in range(img.shape[0]) :
        for j in range(img.shape[1]) :
            if img[i,j,c] > max :
                max = img[i,j,c]

    return max
