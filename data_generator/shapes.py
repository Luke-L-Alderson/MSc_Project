import numpy as np
       

def triangle(center, width):
    side = 9
    def check_triangle(v, v0, v1, v2):
        def det(u, v):
            return u[0]*v[1] - u[1]*v[0]
        a = (det(v, v2) - det(v0, v2))/det(v1, v2)
        b = -(det(v, v1) - det(v0, v1))/det(v1, v2)
        return bool(a>=0 and b>=0 and a+b <= 1)
    blank_frame = np.zeros((width, width))
    v0 = np.array([center[0], center[1] + side/2*np.sqrt(1/2)])
    v1 = -side*np.array([np.cos(45), np.sin(45)])
    v2 = -side*np.array([-np.cos(45), np.sin(45)])
    for i in range(width):
        for j in range(width):
            v = np.array([i, j])
            if check_triangle(v, v0, v1, v2):
                blank_frame[i,j] = 1 
    return blank_frame


def square(center, width):
    side = 7
    blank_frame = np.zeros((width, width))
    for i in range(width):
        for j in range(width):
            x0, y0 = center
            if abs(x0 - i) <= side/2 and    abs(y0 - j) <= side/2:
                blank_frame[i,j] = 1
    return blank_frame


def circle(center, width):
    radius = 4
    blank_frame = np.zeros((width, width))
    for i in range(width):
        for j in range(width):
            x0, y0 = center
            if ((x0 - i)**2 + (j - y0)**2) <= radius**2:
                blank_frame[i,j] = 1
    return blank_frame