import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
# def fish2equi(img, K, D, size, aperture):
#     h_src, w_src = img.shape[:2]
#     w_dst, h_dst = size

#     dst_img = np.zeros((h_dst, w_dst, 3))

#     for y in reversed(range(h_dst)):
#         y_dst_norm = lerp(-1, 1, 0, h_dst, y)

#         for x in range(w_dst):
#             x_dst_norm = lerp(-1, 1, 0, w_dst, x)

#             longitude = x_dst_norm * np.pi + np.pi / 2
#             latitude = y_dst_norm * np.pi / 2 
            
#             p_x = np.cos(latitude) * np.cos(longitude)
#             p_y = np.cos(latitude) * np.sin(longitude)
#             p_z = np.sin(latitude)

#             rot = np.array([[1,0,0], [0, 0, -1], [0, 1, 0]])
#             p_x = -p_x
#             p_x,p_y,p_z = np.matmul(rot, [p_x,p_y,p_z])
#             p_xz = np.sqrt(p_x**2 + p_z**2)

#             theta = np.arctan2(p_xz, p_y)
#             r = ((2 * theta/ aperture))
#             phi = np.arctan2(p_z, p_x)

#             u,v = rad2vec(r, theta, phi, K, D)

#             if y > h_dst/2 - 1 : # under 0 degree, half of the vertical position, no represetation 
#                 pass
#             elif y == 255:
#                 dst_img[y][x] = (255,0,0)
#             else:
#                 # ##################################
#                 # ### bilinear interpolation
#                 tx = np.minimum(w_src - 1, np.floor(u).astype(int))
#                 ty = np.minimum(w_src - 1, np.floor(v).astype(int))

#                 a = u - tx
#                 b = v - ty

#                 if(tx >= 0 and tx < w_src -1 and ty >= 0 and ty < h_src-1):                    
#                     if tx == w_src -1:
#                         tx-=1
#                     if ty == h_src -1:
#                         ty-=1
                    
#                     c_top = img[ty+1][tx] * (1. - a) + img[ty+1][tx+1] * (a)
#                     c_bot = img[ty][tx] * (1. - a) + img[ty][tx+1] * (a)
#                     dst_img[y][x] = c_bot * (1. -b) + c_top * (b)
                    
#     return dst_img

def distortion(src):
    h, w, c = src.shape

    pi = np.math.pi
    
    unit_w = 2*pi/w
    unit_h = pi/h

    rho = np.math.tan(unit_w)

    v = np.matrix([0,1,0])

    r_grid = np.array([[-1,-1],[0,-1],[1,-1],[-1,0],[0,0],[1,0],[-1,1],[0,1],[1,1]])
    # r_grid = np.array([[-1,-1],[-1,0],[-1,1],[0,-1],[0,0],[0,1],[1,-1],[1,0],[1,1]])
    
    
    x = int(w/2)

    kernel = []

    # for x in w:
    for y in range(h):

        # radian
        theta = (x- w*0.5) * unit_w
        phi   = (h*0.5 - y) * unit_h

        x_u = np.math.cos(phi)*np.math.sin(theta)
        y_u = np.math.sin(phi)
        z_u = np.math.cos(phi)*np.math.cos(theta)

        p_u = np.array([x_u, y_u, z_u])

        t_x = np.abs(np.cross(v, p_u))
        t_y = np.abs(np.cross(p_u,t_x))
        # t_x = np.cross(v, p_u)
        # t_y = np.cross(p_u,t_x)
        
        r_sphere = list()
        for r in r_grid:
            r_sphere.append(rho * (r[0]*t_x + r[1]*t_y))

        r_sphere = np.squeeze(r_sphere)
        p_ur = p_u + r_sphere
        p_ur /= np.linalg.norm(p_u + r_sphere)
        
        k = []
        for ur_i in p_ur:

            if ur_i[0] >= 0:
                theta_r = np.math.atan2(ur_i[0], ur_i[2])
            else:
                theta_r = np.math.atan2(ur_i[0], ur_i[2]) + pi
            
            phi_r = np.math.asin(ur_i[1])

            x_r = (np.divide(theta_r, 2*pi) + 0.5)* w
            y_r = (0.5 - np.divide(phi_r, pi))* h

            k.append([x_r - x, y_r - y])

        kernel.append(k)

    return kernel
    
        
def bilinear_interpolation(p, shape):

    h, w, c = shape

    x, y = p

    out = 0                                   
    if(x >= 0 and x < w-1 and y >= 0 and y < h-1):                    
        if x == w-1:
            x-=1
        if y == h-1:
            y-=1
        
        xc = np.ceil(x)
        xf = np.floor(x)
        yc = np.ceil(y)
        yf = np.floor(y)
        
        out += np.max(1-np.abs(), initial=0.)

if __name__ == "__main__":

    img = cv2.imread("pano.png", -1)
    img = cv2.resize(img, (1664, 832))

    h, w, c = img.shape

    newimg = np.zeros_like(img)

    kernel_offset = distortion(img)

    print(kernel_offset)
    # x = int(w/2)
    
    # for k in kernel_offset: # same as height in this test
    #     for p in k:
    #         bilinear_interpolation(p, img.shape)

    # print()
    
    # plt.imshow(img)
    # plt.show()

