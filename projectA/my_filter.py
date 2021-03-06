import numpy as np

def convolve(img,rows, cols, kernel):
    # assume kernel is 3x3

    

    right = np.hstack([img[:, 1:], np.zeros((rows, 1))]) *kernel[1,2]
    left = np.hstack([np.zeros((rows, 1)), img[:, :-1]]) *kernel[1,0]

    top = np.vstack([np.zeros((1, cols)), img[:-1,:]]) * kernel[0,1]
    bottom = np.vstack([img[1:,:], np.zeros((1, cols))]) * kernel[2,1]

    top_left = np.hstack([np.zeros((rows, 1)), top[:,:-1]]) * kernel[0,0]
    top_right = np.hstack([top[:, 1:], np.zeros((rows, 1))]) * kernel[0,2]

    bottom_right = np.hstack([bottom[:, 1:], np.zeros((rows, 1))]) *kernel[2,2]
    bottom_left = np.hstack([np.zeros((rows, 1)), bottom[:, :-1]]) *kernel[2,0]

    middle = img.copy() * kernel[1,1]

    return middle + right + left + top + bottom + top_left + top_right + bottom_right + bottom_left



if __name__ == "__main__":
    img = np.ones((25,25))
    k = np.ones((3,3))

    out = convolve(img, 25, 25, k)

    print(out)
