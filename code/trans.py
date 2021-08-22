'''
>>> alpha, beta, gamma = 0.123, -1.234, 2.345
>>> origin, xaxis, yaxis, zaxis = [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]
>>> I = identity_matrix()
>>> Rx = rotation_matrix(alpha, xaxis)
>>> Ry = rotation_matrix(beta, yaxis)
>>> Rz = rotation_matrix(gamma, zaxis)
>>> R = concatenate_matrices(Rx, Ry, Rz)
>>> qx = quaternion_about_axis(alpha, xaxis)
>>> qy = quaternion_about_axis(beta, yaxis)
>>> qz = quaternion_about_axis(gamma, zaxis)
>>> q = quaternion_multiply(qx, qy)
>>> q = quaternion_multiply(q, qz)
>>> Rq = quaternion_matrix(q)
>>> is_same_transform(R, Rq)
True
>>> T = translation_matrix([1, 2, 3])
'''
from __future__ import division, print_function
import math
import numpy
import torch
import scipy
# epsilon for testing whether a number is close to zero
_EPS = numpy.finfo(float).eps * 4.0
#4
def axisangle2quaternion(axis,angle):
    q = numpy.array([0.0, axis[0], axis[1], axis[2]])
    qlen = vector_norm(q)
    if qlen > _EPS:
        q *= math.sin(angle/2.0) / qlen
    q[0] = math.cos(angle/2.0)
    return q
#1
def quaternion2axisangle(para):
    '''
    input_shape:(1,4)
    output_shape:(1,3) axis vector
                 (1,1) angle scalar value
    '''
    try:
        rad = 2.0 * math.acos(para[0])
    except:
        if para[0] > 1:
            para[0] = 1
        elif para[0] < -1:
            para[0] = -1
        rad = 2.0 * math.acos(para[0])
        print('warning:',para)
    # angle = rad * 180.0/numpy.pi
    sin_ = math.sin(rad*0.5)
    sin = numpy.sqrt(1 - math.cos(rad*0.5) * math.cos(rad*0.5))
    if sin_ <= _EPS:
        sin_ += _EPS
    x = para[1] / sin_
    y = para[2] / sin_
    z = para[3] / sin_
    axis = numpy.array([x,y,z])
    return axis,rad
#6
def quaternion2matrix_torch(quater):
    '''
    input_shape: (batch,4)
    output_shape:(3,3)
    '''
    # quaternion = quaternion.detach().cpu()
    # q = numpy.array(quaternion, dtype=numpy.float64, copy=True)
    # n = numpy.dot(q, q)
    # print(quaternion.shape) #64,4
    # print(torch.transpose(quaternion,-2,-1).shape) #4,64
    quaternion = quater.cpu().detach().numpy()
    matrix = numpy.array((4,4))
    mm = []
    for i in range(quaternion.shape[0]):
        n = numpy.dot(quaternion[i], quaternion[i])
        q = quaternion[i]
        if n < _EPS:
            #TO DO
            matrix = numpy.identity(4)
        else:
            q *= math.sqrt(2.0 / n)
            q = numpy.outer(q, q)
            matrix = numpy.array([
                [1.0-q[2, 2]-q[3, 3],     q[1, 2]-q[3, 0],     q[1, 3]+q[2, 0],0],
                [    q[1, 2]+q[3, 0], 1.0-q[1, 1]-q[3, 3],     q[2, 3]-q[1, 0],0],
                [    q[1, 3]-q[2, 0],     q[2, 3]+q[1, 0], 1.0-q[1, 1]-q[2, 2],0],
                [                0.0,                 0.0,                 0.0,1.0]
                ])
        mm.append(matrix)
    mm = numpy.array(mm)
    return torch.from_numpy(mm)

#5
def translation2matrix(direction):
    
    
    M = numpy.identity(4)
    M[:3, 3] = direction[:3]
    return M
    
#7
def transform_pts_torch(points,transform):
    '''
    input:pointclouds,rotation_matrix
    shape:(2048,3) (3,3)
    output:transformed pointcloud
    shape: (2048,3)
    '''
    points = points.float()
    transform = transform.float()

    if len(transform.shape) == 3:
        rot = transform[:, :3, :3]
        trans = transform[:, :3, 3]
    # single transform
    else:
        rot = transform[:3, :3]
        trans = transform[:3, 3]
    point = torch.matmul(points, torch.transpose(rot,-2,-1)) + torch.unsqueeze(trans, -2)
    return point
#2
def transform_pts(points,transform):
    '''
    input:pointclouds,translation/rotation_matrix
    shape:(2048,3) (3,3)
    output:transformed pointcloud
    shape: (2048,3)
    '''
    if len(transform.shape) == 3:
        rot = transform[:, :3, :3]
        trans = transform[:, :3, 3]
    # single transform
    else:
        rot = transform[:3, :3]
        trans = transform[:3, 3]
    point = numpy.matmul(points, numpy.transpose(rot)) + numpy.expand_dims(trans, axis=-2)
    # point = numpy.matmul(points, numpy.transpose(rot)) 
    # point = torch.matmul(points, torch.transpose(rot,-2,-1)) + torch.unsqueeze(trans, -2)
    return point
#3
def quaternion_inv(quaternion):

    """Return inverse of quaternion.

    >>> q0 = random_quaternion()
    >>> q1 = quaternion_inverse(q0)
    >>> numpy.allclose(quaternion_multiply(q0, q1), [1, 0, 0, 0])
    True

    """
    q = numpy.array(quaternion, dtype=numpy.float64, copy=True)
    numpy.negative(q[1:], q[1:])
    return q / numpy.dot(q, q)

def vector_norm(data, axis=None, out=None):
    """Return length, i.e. Euclidean norm, of ndarray along axis.

    >>> v = numpy.random.random(3)
    >>> n = vector_norm(v)
    >>> numpy.allclose(n, numpy.linalg.norm(v))
    True
    >>> v = numpy.random.rand(6, 5, 3)
    >>> n = vector_norm(v, axis=-1)
    >>> numpy.allclose(n, numpy.sqrt(numpy.sum(v*v, axis=2)))
    True
    >>> n = vector_norm(v, axis=1)
    >>> numpy.allclose(n, numpy.sqrt(numpy.sum(v*v, axis=1)))
    True
    >>> v = numpy.random.rand(5, 4, 3)
    >>> n = numpy.empty((5, 3))
    >>> vector_norm(v, axis=1, out=n)
    >>> numpy.allclose(n, numpy.sqrt(numpy.sum(v*v, axis=1)))
    True
    >>> vector_norm([])
    0.0
    >>> vector_norm([1])
    1.0

    """
    data = numpy.array(data, dtype=numpy.float64, copy=True)
    if out is None:
        if data.ndim == 1:
            return math.sqrt(numpy.dot(data, data))
        data *= data
        out = numpy.atleast_1d(numpy.sum(data, axis=axis))
        numpy.sqrt(out, out)
        return out
    else:
        data *= data
        numpy.sum(data, axis=axis, out=out)
        numpy.sqrt(out, out)
def unit_vector(data, axis=None, out=None):
    """Return ndarray normalized by length, i.e. Euclidean norm, along axis.

    >>> v0 = numpy.random.random(3)
    >>> v1 = unit_vector(v0)
    >>> numpy.allclose(v1, v0 / numpy.linalg.norm(v0))
    True
    >>> v0 = numpy.random.rand(5, 4, 3)
    >>> v1 = unit_vector(v0, axis=-1)
    >>> v2 = v0 / numpy.expand_dims(numpy.sqrt(numpy.sum(v0*v0, axis=2)), 2)
    >>> numpy.allclose(v1, v2)
    True
    >>> v1 = unit_vector(v0, axis=1)
    >>> v2 = v0 / numpy.expand_dims(numpy.sqrt(numpy.sum(v0*v0, axis=1)), 1)
    >>> numpy.allclose(v1, v2)
    True
    >>> v1 = numpy.empty((5, 4, 3))
    >>> unit_vector(v0, axis=1, out=v1)
    >>> numpy.allclose(v1, v2)
    True
    >>> list(unit_vector([]))
    []
    >>> list(unit_vector([1]))
    [1.0]

    """
    if out is None:
        data = numpy.array(data, dtype=numpy.float64, copy=True)
        if data.ndim == 1:
            data /= math.sqrt(numpy.dot(data, data))
            return data
    else:
        if out is not data:
            out[:] = numpy.array(data, copy=False)
        data = out
    length = numpy.atleast_1d(numpy.sum(data*data, axis))
    numpy.sqrt(length, length)
    if axis is not None:
        length = numpy.expand_dims(length, axis)
    data /= length
    if out is None:
        return data
def axisangle2matrix(angle, direction, point=None):
    """Return matrix to rotate about axis defined by point and direction.

    >>> R = rotation_matrix(math.pi/2, [0, 0, 1], [1, 0, 0])
    >>> numpy.allclose(numpy.dot(R, [0, 0, 0, 1]), [1, -1, 0, 1])
    True
    >>> angle = (random.random() - 0.5) * (2*math.pi)
    >>> direc = numpy.random.random(3) - 0.5
    >>> point = numpy.random.random(3) - 0.5
    >>> R0 = rotation_matrix(angle, direc, point)
    >>> R1 = rotation_matrix(angle-2*math.pi, direc, point)
    >>> is_same_transform(R0, R1)
    True
    >>> R0 = rotation_matrix(angle, direc, point)
    >>> R1 = rotation_matrix(-angle, -direc, point)
    >>> is_same_transform(R0, R1)
    True
    >>> I = numpy.identity(4, numpy.float64)
    >>> numpy.allclose(I, rotation_matrix(math.pi*2, direc))
    True
    >>> numpy.allclose(2, numpy.trace(rotation_matrix(math.pi/2,
    ...                                               direc, point)))
    True

    """
    sina = math.sin(angle)
    cosa = math.cos(angle)
    direction = unit_vector(direction[:3])
    # rotation matrix around unit vector
    R = numpy.diag([cosa, cosa, cosa])
    R += numpy.outer(direction, direction) * (1.0 - cosa)
    direction *= sina
    R += numpy.array([[ 0.0,         -direction[2],  direction[1]],
                      [ direction[2], 0.0,          -direction[0]],
                      [-direction[1], direction[0],  0.0]])
    M = numpy.identity(4)
    M[:3, :3] = R
    if point is not None:
        # rotation not around origin
        point = numpy.array(point[:3], dtype=numpy.float64, copy=False)
        M[:3, 3] = point - numpy.dot(R, point)
    return M

def matrix2quaternion(matrix, isprecise=False):
    """Return quaternion from rotation matrix.

    If isprecise is True, the input matrix is assumed to be a precise rotation
    matrix and a faster algorithm is used.

    >>> q = quaternion_from_matrix(numpy.identity(4), True)
    >>> numpy.allclose(q, [1, 0, 0, 0])
    True
    >>> q = quaternion_from_matrix(numpy.diag([1, -1, -1, 1]))
    >>> numpy.allclose(q, [0, 1, 0, 0]) or numpy.allclose(q, [0, -1, 0, 0])
    True
    >>> R = rotation_matrix(0.123, (1, 2, 3))
    >>> q = quaternion_from_matrix(R, True)
    >>> numpy.allclose(q, [0.9981095, 0.0164262, 0.0328524, 0.0492786])
    True
    >>> R = [[-0.545, 0.797, 0.260, 0], [0.733, 0.603, -0.313, 0],
    ...      [-0.407, 0.021, -0.913, 0], [0, 0, 0, 1]]
    >>> q = quaternion_from_matrix(R)
    >>> numpy.allclose(q, [0.19069, 0.43736, 0.87485, -0.083611])
    True
    >>> R = [[0.395, 0.362, 0.843, 0], [-0.626, 0.796, -0.056, 0],
    ...      [-0.677, -0.498, 0.529, 0], [0, 0, 0, 1]]
    >>> q = quaternion_from_matrix(R)
    >>> numpy.allclose(q, [0.82336615, -0.13610694, 0.46344705, -0.29792603])
    True
    >>> R = random_rotation_matrix()
    >>> q = quaternion_from_matrix(R)
    >>> is_same_transform(R, quaternion_matrix(q))
    True
    >>> is_same_quaternion(quaternion_from_matrix(R, isprecise=False),
    ...                    quaternion_from_matrix(R, isprecise=True))
    True
    >>> R = euler_matrix(0.0, 0.0, numpy.pi/2.0)
    >>> is_same_quaternion(quaternion_from_matrix(R, isprecise=False),
    ...                    quaternion_from_matrix(R, isprecise=True))
    True

    """
    M = numpy.array(matrix, dtype=numpy.float64, copy=False)[:4, :4]
    if isprecise:
        q = numpy.empty((4, ))
        t = numpy.trace(M)
        if t > M[3, 3]:
            q[0] = t
            q[3] = M[1, 0] - M[0, 1]
            q[2] = M[0, 2] - M[2, 0]
            q[1] = M[2, 1] - M[1, 2]
        else:
            i, j, k = 0, 1, 2
            if M[1, 1] > M[0, 0]:
                i, j, k = 1, 2, 0
            if M[2, 2] > M[i, i]:
                i, j, k = 2, 0, 1
            t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
            q[i] = t
            q[j] = M[i, j] + M[j, i]
            q[k] = M[k, i] + M[i, k]
            q[3] = M[k, j] - M[j, k]
            q = q[[3, 0, 1, 2]]
        q *= 0.5 / math.sqrt(t * M[3, 3])
    else:
        m00 = M[0, 0]
        m01 = M[0, 1]
        m02 = M[0, 2]
        m10 = M[1, 0]
        m11 = M[1, 1]
        m12 = M[1, 2]
        m20 = M[2, 0]
        m21 = M[2, 1]
        m22 = M[2, 2]
        # symmetric matrix K
        K = numpy.array([[m00-m11-m22, 0.0,         0.0,         0.0],
                         [m01+m10,     m11-m00-m22, 0.0,         0.0],
                         [m02+m20,     m12+m21,     m22-m00-m11, 0.0],
                         [m21-m12,     m02-m20,     m10-m01,     m00+m11+m22]])
        K /= 3.0
        # quaternion is eigenvector of K that corresponds to largest eigenvalue
        w,V = numpy.linalg.eigh(K)
		# w, V = numpy.linalg.eigh(K)
        q = V[[3, 0, 1, 2], numpy.argmax(w)]
    if q[0] < 0.0:
        numpy.negative(q, q)
    return q


def quaternion2matrix(quaternion):
    '''
    input_shape: (1,4)
    output_shape:(3,3)
    '''
    q = numpy.array(quaternion, dtype=numpy.float64, copy=True)
    n = numpy.dot(q, q)
    # print(n)
    if n < _EPS:
        return numpy.identity(4)
    q *= math.sqrt(2.0 / n)
    q = numpy.outer(q, q)
    return numpy.array([
        [1.0-q[2, 2]-q[3, 3],     q[1, 2]-q[3, 0],     q[1, 3]+q[2, 0], 0.0],
        [    q[1, 2]+q[3, 0], 1.0-q[1, 1]-q[3, 3],     q[2, 3]-q[1, 0], 0.0],
        [    q[1, 3]-q[2, 0],     q[2, 3]+q[1, 0], 1.0-q[1, 1]-q[2, 2], 0.0],
        [                0.0,                 0.0,                 0.0, 1.0]])

def qt2mat(qt):
#   q = qt[:4] / numpy.linalg.norm(qt[:4])
#   # t = qt[4:]
#   T = [q[0] ** 2 + q[1] ** 2 - q[2] ** 2 - q[3] ** 2,
#        2 * (q[1] * q[2] - q[0] * q[3]),
#        2 * (q[1] * q[3] + q[0] * q[2]),
#        0, 

#        2 * (q[1] * q[2] + q[0] * q[3]),
#        q[0] ** 2 - q[1] ** 2 + q[2] ** 2 - q[3] ** 2,
#        2 * (q[2] * q[3] - q[0] * q[1]),
#        0,

#        2 * (q[1] * q[3] - q[0] * q[2]),
#        2 * (q[2] * q[3] + q[0] * q[1]),
#        q[0] ** 2 - q[1] ** 2 - q[2] ** 2 + q[3] ** 2,
#        0,

#        0,0,0,1       
#        ]
#   return numpy.reshape(T, (4, 4))
    q = qt[:4] / numpy.linalg.norm(qt[:4])
    # t = qt[4:]
    T = [q[0] ** 2 + q[1] ** 2 - q[2] ** 2 - q[3] ** 2,
        2 * (q[1] * q[2] - q[0] * q[3]),
        2 * (q[1] * q[3] + q[0] * q[2]),

        2 * (q[1] * q[2] + q[0] * q[3]),
        q[0] ** 2 - q[1] ** 2 + q[2] ** 2 - q[3] ** 2,
        2 * (q[2] * q[3] - q[0] * q[1]),

        2 * (q[1] * q[3] - q[0] * q[2]),
        2 * (q[2] * q[3] + q[0] * q[1]),
        q[0] ** 2 - q[1] ** 2 - q[2] ** 2 + q[3] ** 2,   
        ]
    return numpy.reshape(T, (3, 3))

def mat2qt(R):
  # t = T[:3, :]
  # R = T[:3, :3]
  w = numpy.sqrt(1 + numpy.trace(R)) / 2
  x = (R[2, 1] - R[1, 2]) / ((4 * w)+1e-8)
  y = (R[0, 2] - R[2, 0]) / ((4 * w)+1e-8)
  z = (R[1, 0] - R[0, 1]) / ((4 * w)+1e-8)
  return numpy.stack([w, x, y, z], 0)

def transform_points(points, transform):
	# batch transform
	if len(transform.shape) == 3:
		rot = transform[:, :3, :3]
		trans = transform[:, :3, 3]
	# single transform
	else:
		rot = transform[:3, :3]
		trans = transform[:3, 3]
	point = torch.matmul(points, torch.transpose(rot,-2,-1)) + torch.unsqueeze(trans, -2)
	return point

''' #Numpy Version# '''
def transform_points_np(points, transform):
	# batch transform
	if len(transform.shape) == 3:
		rot = transform[:, :3, :3]
		# trans = transform[:, :3, 3]
	# single transform
	else:
		rot = transform[:3, :3]
		# trans = transform[:3, 3]
	point = numpy.matmul(points, numpy.transpose(rot))
	return point

def geometric_error(points, T, T_gt):
  trans_points = transform_points(points, T)
  trans_points_gt = transform_points(points, T_gt)
  dist = numpy.linalg.norm(trans_points - trans_points_gt, axis=-1)
  return numpy.reduce_mean(dist, axis=-1)


def rotation_error(R, R_gt):
  cos_theta = (numpy.trace(numpy.matmul(R, R_gt.T)) - 1) / 2
  return 1-cos_theta#tf.acos(cos_theta) * 180 / np.pi


def translation_error(t, t_gt):
  return numpy.linalg.norm(t - t_gt, axis=-1)

def identity_matrix():
    """Return 4x4 identity/unit matrix.

    >>> I = identity_matrix()
    >>> numpy.allclose(I, numpy.dot(I, I))
    True
    >>> numpy.sum(I), numpy.trace(I)
    (4.0, 4.0)
    >>> numpy.allclose(I, numpy.identity(4))
    True

    """
    return numpy.identity(4)

def translation_from_matrix(matrix):
    """Return translation vector from translation matrix.

    >>> v0 = numpy.random.random(3) - 0.5
    >>> v1 = translation_from_matrix(translation_matrix(v0))
    >>> numpy.allclose(v0, v1)
    True

    """
    return numpy.array(matrix, copy=False)[:3, 3].copy()

def euler_from_quaternion(quaternion, axes='sxyz'):
    """Return Euler angles from quaternion for specified axis sequence.

    >>> angles = euler_from_quaternion([0.99810947, 0.06146124, 0, 0])
    >>> numpy.allclose(angles, [0.123, 0, 0])
    True

    """
    return euler_from_matrix(quaternion_matrix(quaternion), axes)

def quaternion_from_euler(ai, aj, ak, axes='sxyz'):
    """Return quaternion from Euler angles and axis sequence.

    ai, aj, ak : Euler's roll, pitch and yaw angles
    axes : One of 24 axis sequences as string or encoded tuple

    >>> q = quaternion_from_euler(1, 2, 3, 'ryxz')
    >>> numpy.allclose(q, [0.435953, 0.310622, -0.718287, 0.444435])
    True

    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
    except (AttributeError, KeyError):
        _TUPLE2AXES[axes]  # noqa: validation
        firstaxis, parity, repetition, frame = axes

    i = firstaxis + 1
    j = _NEXT_AXIS[i+parity-1] + 1
    k = _NEXT_AXIS[i-parity] + 1

    if frame:
        ai, ak = ak, ai
    if parity:
        aj = -aj

    ai /= 2.0
    aj /= 2.0
    ak /= 2.0
    ci = math.cos(ai)
    si = math.sin(ai)
    cj = math.cos(aj)
    sj = math.sin(aj)
    ck = math.cos(ak)
    sk = math.sin(ak)
    cc = ci*ck
    cs = ci*sk
    sc = si*ck
    ss = si*sk

    q = numpy.empty((4, ))
    if repetition:
        q[0] = cj*(cc - ss)
        q[i] = cj*(cs + sc)
        q[j] = sj*(cc + ss)
        q[k] = sj*(cs - sc)
    else:
        q[0] = cj*cc + sj*ss
        q[i] = cj*sc - sj*cs
        q[j] = cj*ss + sj*cc
        q[k] = cj*cs - sj*sc
    if parity:
        q[j] *= -1.0

    return q

def quaternion_about_axis(angle, axis):
    """Return quaternion for rotation about axis.

    >>> q = quaternion_about_axis(0.123, [1, 0, 0])
    >>> numpy.allclose(q, [0.99810947, 0.06146124, 0, 0])
    True

    """
    q = numpy.array([0.0, axis[0], axis[1], axis[2]])
    qlen = vector_norm(q)
    if qlen > _EPS:
        q *= math.sin(angle/2.0) / qlen
    q[0] = math.cos(angle/2.0)
    return q


def quaternion_matrix(quaternion):
    """Return homogeneous rotation matrix from quaternion.

    >>> M = quaternion_matrix([0.99810947, 0.06146124, 0, 0])
    >>> numpy.allclose(M, rotation_matrix(0.123, [1, 0, 0]))
    True
    >>> M = quaternion_matrix([1, 0, 0, 0])
    >>> numpy.allclose(M, numpy.identity(4))
    True
    >>> M = quaternion_matrix([0, 1, 0, 0])
    >>> numpy.allclose(M, numpy.diag([1, -1, -1, 1]))
    True

    """
    q = numpy.array(quaternion, dtype=numpy.float64, copy=True)
    n = numpy.dot(q, q)
    if n < _EPS:
        return numpy.identity(4)
    q *= math.sqrt(2.0 / n)
    q = numpy.outer(q, q)
    return numpy.array([
        [1.0-q[2, 2]-q[3, 3],     q[1, 2]-q[3, 0],     q[1, 3]+q[2, 0], 0.0],
        [    q[1, 2]+q[3, 0], 1.0-q[1, 1]-q[3, 3],     q[2, 3]-q[1, 0], 0.0],
        [    q[1, 3]-q[2, 0],     q[2, 3]+q[1, 0], 1.0-q[1, 1]-q[2, 2], 0.0],
        [                0.0,                 0.0,                 0.0, 1.0]])

def quaternion_from_matrix(matrix, isprecise=False):
    """Return quaternion from rotation matrix.

    If isprecise is True, the input matrix is assumed to be a precise rotation
    matrix and a faster algorithm is used.

    >>> q = quaternion_from_matrix(numpy.identity(4), True)
    >>> numpy.allclose(q, [1, 0, 0, 0])
    True
    >>> q = quaternion_from_matrix(numpy.diag([1, -1, -1, 1]))
    >>> numpy.allclose(q, [0, 1, 0, 0]) or numpy.allclose(q, [0, -1, 0, 0])
    True
    >>> R = rotation_matrix(0.123, (1, 2, 3))
    >>> q = quaternion_from_matrix(R, True)
    >>> numpy.allclose(q, [0.9981095, 0.0164262, 0.0328524, 0.0492786])
    True
    >>> R = [[-0.545, 0.797, 0.260, 0], [0.733, 0.603, -0.313, 0],
    ...      [-0.407, 0.021, -0.913, 0], [0, 0, 0, 1]]
    >>> q = quaternion_from_matrix(R)
    >>> numpy.allclose(q, [0.19069, 0.43736, 0.87485, -0.083611])
    True
    >>> R = [[0.395, 0.362, 0.843, 0], [-0.626, 0.796, -0.056, 0],
    ...      [-0.677, -0.498, 0.529, 0], [0, 0, 0, 1]]
    >>> q = quaternion_from_matrix(R)
    >>> numpy.allclose(q, [0.82336615, -0.13610694, 0.46344705, -0.29792603])
    True
    >>> R = random_rotation_matrix()
    >>> q = quaternion_from_matrix(R)
    >>> is_same_transform(R, quaternion_matrix(q))
    True
    >>> is_same_quaternion(quaternion_from_matrix(R, isprecise=False),
    ...                    quaternion_from_matrix(R, isprecise=True))
    True
    >>> R = euler_matrix(0.0, 0.0, numpy.pi/2.0)
    >>> is_same_quaternion(quaternion_from_matrix(R, isprecise=False),
    ...                    quaternion_from_matrix(R, isprecise=True))
    True

    """
    M = numpy.array(matrix, dtype=numpy.float64, copy=False)[:4, :4]
    if isprecise:
        q = numpy.empty((4, ))
        t = numpy.trace(M)
        if t > M[3, 3]:
            q[0] = t
            q[3] = M[1, 0] - M[0, 1]
            q[2] = M[0, 2] - M[2, 0]
            q[1] = M[2, 1] - M[1, 2]
        else:
            i, j, k = 0, 1, 2
            if M[1, 1] > M[0, 0]:
                i, j, k = 1, 2, 0
            if M[2, 2] > M[i, i]:
                i, j, k = 2, 0, 1
            t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
            q[i] = t
            q[j] = M[i, j] + M[j, i]
            q[k] = M[k, i] + M[i, k]
            q[3] = M[k, j] - M[j, k]
            q = q[[3, 0, 1, 2]]
        q *= 0.5 / math.sqrt(t * M[3, 3])
    else:
        m00 = M[0, 0]
        m01 = M[0, 1]
        m02 = M[0, 2]
        m10 = M[1, 0]
        m11 = M[1, 1]
        m12 = M[1, 2]
        m20 = M[2, 0]
        m21 = M[2, 1]
        m22 = M[2, 2]
        # symmetric matrix K
        K = numpy.array([[m00-m11-m22, 0.0,         0.0,         0.0],
                         [m01+m10,     m11-m00-m22, 0.0,         0.0],
                         [m02+m20,     m12+m21,     m22-m00-m11, 0.0],
                         [m21-m12,     m02-m20,     m10-m01,     m00+m11+m22]])
        K /= 3.0
        # quaternion is eigenvector of K that corresponds to largest eigenvalue
        w, V = numpy.linalg.eigh(K)
        q = V[[3, 0, 1, 2], numpy.argmax(w)]
    if q[0] < 0.0:
        numpy.negative(q, q)
    return q
def is_same_quaternion(q0, q1):
    """Return True if two quaternions are equal."""
    q0 = numpy.array(q0)
    q1 = numpy.array(q1)
    return numpy.allclose(q0, q1) or numpy.allclose(q0, -q1)

if __name__ == '__main__':
    # import doctest
    # import random  # noqa: used in doctests
    # try:
    #     numpy.set_printoptions(suppress=True, precision=5, legacy='1.13')
    # except TypeError:
    #     numpy.set_printoptions(suppress=True, precision=5)
    # doctest.testmod()
    a = [0.95,0.3,0.2,0.3]
    b = [1.02664936,0.24730698,0.16487132,0.24730698]
    # print(quaternion2matrix(b))
    # print(qt2mat(b))
    c = b / numpy.sqrt(numpy.sum(numpy.multiply(b,b)))
    print(c)
    print(mat2qt(qt2mat(b)))
    print(matrix2quaternion(quaternion2matrix(b)))
    print(matrix2quaternion(quaternion2matrix(c)))
    
    # print(mat2qt(qt2mat(a)))
    # print(matrix2quaternion(quaternion2matrix(a)))
    # import numpy as np
    # compose_matrix = np.identity(4)
    # compose_matrix_b = np.repeat(np.expand_dims(compose_matrix,axis=0),4,axis=0)
    # print(compose_matrix_b[1])

    # dd = compose_matrix
    # dd[0,0] = 0.3
    # dd[1,1] = 0.4
    # dd[2,2] = 0.5
    # print(matrix2quaternion(dd))
    # print(quaternion2matrix(matrix2quaternion(dd)))

    # dd[0,0] = 0.4
    # dd[1,1] = 0.7
    # dd[2,2] = 0.2
    # print(matrix2quaternion(dd))
    # print(quaternion2matrix(matrix2quaternion(dd)))
    # # print(qt2mat([0.95,0.3,0.2,0.3]))
    # # a = qt2mat([0.95,0.3,0.2,0.3])
    # # b = numpy.array([0.4,0.5,0.2])
    # # # print(transform_points())
    # # print(transform_points_np(b,a))
    # # print(quaternion2axisangle([0.5,0.4,0.2,0.3]))
    # rot_m = axisangle2matrix(10.0*np.pi/180,[0,0,1])
    # print(rot_m)
    # print(rot_m * compose_matrix)
    # print(np.dot(rot_m, compose_matrix))
    import numpy as np
    # print(quaternion2axisangle([0,0,0,0]))
    # out_axis1,b = quaternion2axisangle([0,0,0,0])
    # gt_axis1 = np.array([0.2,0.4,0.5])
    # error_axis = np.abs(np.arccos(np.dot(out_axis1, gt_axis1)/(np.linalg.norm(out_axis1) * np.linalg.norm(gt_axis1)+_EPS)))
    # print(error_axis)
    # print(_EPS)
    # print(np.finfo(float).eps)
    import torch.nn.functional as F
    a = np.array([[1.,2.,3.],[2.,2.,2.]])
    b = np.array([[2.,1.,1.],[3.,1.,0]])
    out_axis1 = a[0]
    gt_axis1 = b[0]
    error_axis = np.abs(np.arccos(np.dot(out_axis1, gt_axis1)/(np.linalg.norm(out_axis1) * np.linalg.norm(gt_axis1)+_EPS)))
    print(np.dot(out_axis1, gt_axis1)/(np.linalg.norm(out_axis1) * np.linalg.norm(gt_axis1)+_EPS))
    print(error_axis)
    out_para_r1 = torch.from_numpy(a[0])
    gt_para_r1 = torch.from_numpy(b[0])
    dist_list = []
    out = (out_para_r1 * out_para_r1).sum().sqrt()
    gt = (gt_para_r1*gt_para_r1).sum().sqrt()
    cos = out_para_r1.dot(gt_para_r1)/(out*gt+_EPS)
    arccos = torch.acos(cos)
    print(1 - cos)
    print(arccos)
   
    
    
    
    
    
