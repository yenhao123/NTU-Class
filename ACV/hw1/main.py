from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import cv2
import argparse

SEARCH_RANGE = 50

# why minus windw_size
def spilt_image_to_blocks(img, window_size, stride):
    pixel, position = [], []
    for x in range(0, img.shape[0] - window_size, stride):
        for y in range(0, img.shape[1] - window_size, stride):
            position.append((x,y))
            pixel.append(img[x:x+window_size,y:y+window_size])

    return {"pixel":pixel, "position":position}

def count_motion_vector(blocks1, blocks2):
    motion_vectors = []
    count = 0
    for pixel1, position1 in zip(blocks1["pixel"], blocks1["position"]):
        cost = 999999
        for pixel2, position2 in zip(blocks2["pixel"], blocks2["position"]):
            distance = ((position2[0] - position1[0])**2 + (position2[1] - position1[1]) ** 2) ** 0.5
            if distance <= SEARCH_RANGE:
                difference = np.sum(abs(pixel2 - pixel1))
                if difference <= cost:
                    cost = difference
                    match_position = position2
        motion_vector = (match_position[0] - position1[0], match_position[1] - position1[1])
        motion_vectors.append((position1, motion_vector))

    return motion_vectors

def get_motion_vector(img1, img2, window_size):
    blocks1 = spilt_image_to_blocks(img1, window_size, window_size)
    blocks2 = spilt_image_to_blocks(img2, window_size, 1)
    motion_vectors = count_motion_vector(blocks1, blocks2)

    return motion_vectors

def plot(motion_vectors, window_size):
    X = [-block[0][0] for block in motion_vectors]
    Y = [block[0][1] for block in motion_vectors]
    U = [-block[1][0] for block in motion_vectors]
    V = [block[1][1] for block in motion_vectors]

    plt.figure(figsize=(10,10))
    plt.quiver(X, Y, U, V)
    plt.title("motion vector for {}x{} block.".format(window_size,window_size))
    plt.axis('off')
    plt.savefig('./output/block_{}.png'.format(str(window_size)))
    plt.show()

def set_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img1_path", default="./trucka.bmp", type=str)
    parser.add_argument("--img2_path", default="./truckb.bmp", type=str)
    parser.add_argument("--window_size", default=31, type=int)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = set_arg()
    img1 = cv2.imread(args.img1_path, cv2.IMREAD_GRAYSCALE).astype("float")
    img2 = cv2.imread(args.img2_path, cv2.IMREAD_GRAYSCALE).astype("float")
    '''
    img1 = cv2.imread("./trucka.bmp").astype("float")
    img2 = cv2.imread("./truckb.bmp").astype("float")
    '''
    window_size = args.window_size
    motion_vectors = get_motion_vector(img1, img2, window_size)
    print(motion_vectors)
    plot(motion_vectors,window_size)



