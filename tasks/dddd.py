import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import time as t
path = "duck.jpg"
ducky = cv.imread(path)
cv.imshow('duck', ducky)
cv.waitKey(0)
cv.destroyAllWindows()
#task 1
imageBGR = cv.imread(path, 1)
cv.imshow('bgr', imageBGR)
cv.waitKey(0)
cv.destroyAllWindows()
grey= cv.imread(path, 0)
cv.imshow('grey duck', grey)
cv.waitKey(0)
cv.destroyAllWindows()
#task 2
down = cv.resize(ducky, (int(ducky.shape[1]*0.6), int(ducky.shape[0]*0.6)))
up = cv.resize(ducky, (int(ducky.shape[1]*2), int(ducky.shape[0]*2)))
cv.imshow('downsize', down)
cv.waitKey(0)
cv.destroyAllWindows()
cv.imshow('upsized', up)
cv.waitKey(0)
cv.destroyAllWindows()
#task 3
width = cv.resize(ducky, (100, ducky.shape[0]))
height = cv.resize(ducky, (ducky.shape[1], 200))
both = cv.resize(ducky, (200,200))
cv.imshow('width pixels', width)
cv.waitKey(0)
cv.destroyAllWindows()
cv.imshow('height pixels', height)
cv.waitKey(0)
cv.destroyAllWindows()
cv.imshow('both', both)
cv.waitKey(0)
cv.destroyAllWindows()
#task 4
scaled_f_down = cv.resize(ducky, None, fx= 0.6, fy= 0.6, interpolation= cv.INTER_LINEAR)
scaled_f_up = cv.resize(ducky, None, fx= 1.2, fy= 1.2, interpolation= cv.INTER_LINEAR)
cv.imshow("scale up ", scaled_f_up)
cv.waitKey(0)
cv.destroyAllWindows()
cv.imshow("scale down ", scaled_f_down)
cv.waitKey(0)
cv.destroyAllWindows()
#task 5
Cropped = ducky [20:200, 50:200]
cv.imshow("Cropped img ", Cropped)
cv.waitKey(0)
cv.destroyAllWindows()
#task 6
h, w, _ = ducky.shape 
mid_h, mid_w = h // 2, w // 2
top_left = ducky[0:mid_h, 0:mid_w]
top_right = ducky[0:mid_h, mid_w:w]
bottom_left = ducky[mid_h:h, 0:mid_w]
bottom_right = ducky[mid_h:h, mid_w:w]
cv.imshow("Top Left", top_left)
cv.imshow("Top Right", top_right)
cv.imshow("Bottom Left", bottom_left)
cv.imshow("Bottom Right", bottom_right)
cv.waitKey(0)
cv.destroyAllWindows()
top = np.hstack((top_left, top_right))
bottom = np.hstack((bottom_left, bottom_right))
stitched = np.vstack((top, bottom))
cv.imshow("Stitched Image", stitched)
cv.waitKey(0)
cv.destroyAllWindows()
#task 7
h, w, _ = ducky.shape 
center = (w/2, h/2) 
deg45 = cv.getRotationMatrix2D(center, 45.0, 1.0)
rot45 = cv.warpAffine(ducky, deg45, (w, h))
deg90 = cv.getRotationMatrix2D(center, 90.0, 1.0)
rot90 = cv.warpAffine(ducky, deg90, (w, h))
deg180 = cv.getRotationMatrix2D(center, 180.0, 1.0)
rot180 = cv.warpAffine(ducky, deg180, (w, h))
cv.imshow('45 deg',rot45)
cv.waitKey(0)
cv.destroyAllWindows()
cv.imshow('90 deg', rot90)
cv.waitKey(0)
cv.destroyAllWindows()
cv.imshow('180 deg',rot180)
cv.waitKey(0)
cv.destroyAllWindows()
#task 8
scale45 = cv.getRotationMatrix2D(center, 45.0, 0.5)
scaled45 = cv.warpAffine(ducky, scale45, (w, h))
scaled_f_downn = cv.resize(ducky, None, fx= 0.5, fy= 0.5, interpolation= cv.INTER_LINEAR)
(h2, w2) = scaled_f_downn.shape[:2]
center2 = (w2 // 2, h2 // 2)
longscale = cv.getRotationMatrix2D(center, 45.0, 1.0)
scalerot = cv.warpAffine(ducky, longscale, (w, h))
cv.imshow('chopped and screwed',scaled45)
cv.imshow('chopped then screwed', scalerot)
cv.waitKey(0)
cv.destroyAllWindows()
#task 9
rgb = cv.cvtColor(ducky,cv.COLOR_BGR2RGB)
hsv = cv.cvtColor(ducky, cv.COLOR_BGR2HSV)
lab = cv.cvtColor(ducky, cv.COLOR_BGR2LAB)
gray = cv.cvtColor(ducky,cv.COLOR_BGR2GRAY)
titles = ['rgb','hsv','lab','Grayscale']
images = [rgb, hsv, lab, gray]
plt.figure(figsize=(12,6))
for i in range(len(images)):
    plt.subplot(2,3,i+1)
    if titles[i] == 'Grayscale':
        plt.imshow(images[i], cmap='gray')
    else:
        plt.imshow(images[i])
    plt.title(titles[i])
    plt.axis('off')
plt.tight_layout()
plt.show()
#task 10
blurr = cv.blur(ducky,(5,5))
mild = np.array([[0, -1, 0],
                    [-1, 5, -1],
                    [0, -1, 0]]) 
intense = np.array([[0, -2, 0],
                    [-2, 9, -2],
                    [0, -2, 0]])
sharp1= cv.filter2D(blurr, -1, mild)
sharp2= cv.filter2D(blurr, -1, intense)
cv.imshow('blurred', blurr)
cv.imshow('mild sharpening', sharp1)
cv.imshow('strong sharpening', sharp2)
cv.waitKey(0) 
cv.destroyAllWindows()
#task 11
def add_salt_pepper_noise(ducky, density=0.05):
    noisy = ducky.copy()
    num_noise = int(density*h*w) 
    coords = [np.random.randint(0, i - 1, num_noise) for i in (h, w)]
    noisy[coords[0], coords[1]] = [255, 255, 255]
    coords = [np.random.randint(0, i - 1, num_noise) for i in (h, w)]
    noisy[coords[0], coords[1]] = [0, 0, 0]
    return noisy
noisy_img = add_salt_pepper_noise(ducky, density=0.05)
cv.imshow("Salt & Pepper Noise", noisy_img)
#task 12 
median3 = cv.medianBlur(noisy_img,3)
median5 = cv.medianBlur(noisy_img,5)
median7 = cv.medianBlur(noisy_img,7)
cv.imshow('3x3 median',median3)
cv.imshow('5x5 median',median5)
cv.imshow('7x7 median',median7)
cv.waitKey(0)
cv.destroyAllWindows()
#task 13
def adaptive_median_filter(ducky, max = 7):
    padded = np.pad(ducky, max//2, mode = 'constant', constant_values = 0)
    filtered = np.zeros_like(ducky)
    row, col = ducky.shape
    for i in range (row):
        for j in range (col):
            window_size = 3
            done = False
            while not done:
                half = window_size // 2
                window = padded[i:i+window_size, j:j+window_size]
                minn = np.min(window)
                maxx = np.max(window)
                median = np.median(window)
                x1= median - minn
                x2 = median - maxx
                if x1>0 and x2<0 :
                    y1 = ducky[i, j] - minn
                    y2 = ducky[i, j] - maxx
                    if y1>0 and y2<0 :
                        filtered[i, j] = ducky[i, j]
                    else:
                        filtered[i, j] = median
                    done = True
                else:
                    window_size+=  2
                    if window_size > max:
                        filtered[i, j] = median
                        done = True

    return filtered.astype(np.uint8)
ducky_gray = cv.cvtColor(ducky, cv.COLOR_BGR2GRAY)  
adaptive_filtered = adaptive_median_filter(ducky_gray, max =7)
median_filtered = cv.medianBlur(ducky_gray, 5)
cv.imshow("Original Noisy Ducky (Gray)", ducky_gray)
cv.imshow("Median Filter (5x5)", median_filtered)
cv.imshow("Adaptive Median Filter", adaptive_filtered)
cv.waitKey(0)
cv.destroyAllWindows()
#task 14
def bilateral_filter(ducky, diameter, sigma_color, sigma_space):
    ducky2 = ducky.astype(np.float32)
    radius = diameter // 2
    filtered2 = np.zeros_like(ducky2, dtype = np.float32)
    ax = np.arange(-radius, radius+1)
    xx, yy = np.meshgrid(ax, ax)
    spatial_weights = np.exp(-(xx**2 + yy**2) / (2* sigma_space**2))
    for i in range(radius, ducky2.shape[0] - radius):
        for j in range(radius, ducky2.shape[1] - radius):
            region = ducky2[i - radius: i+radius+1, j-radius : j+radius+1]
            intensity_diff = region - ducky2[i, j]
            range_weights = np.exp(-(intensity_diff**2) / (2* sigma_color**2))
            weights = spatial_weights * range_weights 
            weights /= np.sum(weights)
            filtered2[i, j] = np.sum(region*weights)
    return np.clip(filtered2, 0, 255).astype(np.uint8)
custom = bilateral_filter(grey, diameter = 7, sigma_color = 30, sigma_space = 30)
opencv_bilateral = cv.bilateralFilter(grey, d=7, sigmaColor = 30, sigmaSpace = 30)
cv.imshow('og grey ducky', grey)
cv.imshow('custom bilateral', custom)
cv.imshow('opencv bilateral', opencv_bilateral)
cv.waitKey(0)
cv.destroyAllWindows()
#task 17
#im going to use the functions we created earlier
noisy_ducky = add_salt_pepper_noise(ducky, density = 0.1)
start = t.time()
median3 = cv.medianBlur(noisy_ducky, 3)
end = t.time()
print('standard median(3x3) time:', end - start, 'seconds')
start = t.time()
adaptive = adaptive_median_filter(cv.cvtColor(noisy_ducky, cv.COLOR_BGR2GRAY), max=7)
end = t.time()
print('adaptive median time:', end - start, 'seconds')
cv.imshow('og noisy', noisy_ducky)
cv.imshow('standard median 3x3', median3)
cv.imshow('adaptive median', adaptive)
cv.waitKey(0)
cv.destroyAllWindows() 


