# used to calculate conversion
import numpy as np

def Quaternion(m):
    q = np.zeros((4,), np.float32)
    q[0] = np.sqrt(1+m[0][0]+m[1][1]+m[2][2])/2
    q[1] = (m[2][1]-m[1][2])/(4*q[0])
    q[2] = (m[0][2]-m[2][0])/(4*q[0])
    q[3] = (m[1][0]-m[0][1])/(4*q[0])
    return q

def Matrix33(q):
    w,x,y,z = q 
    x2 = x*x
    y2 = y*y
    z2 = z*z
    xy = x*y
    xz = x*z
    yz = y*z
    wx = w*x
    wy = w*y
    wz = w*z
    return np.array([[1-2*(y2+z2), 2*(xy-wz), 2*(xz+wy)],
                     [2*(xy+wz), 1-2*(x2+z2), 2*(yz-wx)],
                     [2*(xz-wy), 2*(yz+wx), 1-2*(x2+y2)]
                    ])


