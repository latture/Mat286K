__author__ = 'Ryan'

import numpy as np
from math import cos, sin, sqrt, acos

'''
Utility Functions
'''
def deleteDuplicates(input_array):
    """
    Returns the unique rows of input_array.

    Parameters
    ----------
    input_array : 2D array.
    """
    # because unique flattens the array, each row is first transformed to into
    #a view (simply a slice of the array with a specific type). np.unique is then called on the array of views.
    #This allows the rows of the input array to be compared without flattening the dimensions.
    unique_rows = np.unique(input_array.view([('', input_array.dtype)] * input_array.shape[1]))

    #change back to normal 2D array with same type as input_array
    return unique_rows.view(input_array.dtype).reshape((unique_rows.shape[0], input_array.shape[1]))


def replaceWithIndex(nodes, elem_coords):
    """
    Returns an array with the indices of the nodal positions of the points specified in elemCoords within nodes.

    Parameters
    ----------
    nodes      : A 2D array containing the (x,y,z) coordinates of the nodal positions.
    elemCoords : A 3D array of the from [[e1_pt1,el1_pt2,el1_ptN],...,[eN_pt1,el1_pt2,elN_ptN]], where elN_ptN contains the set of (x,y,z) coordinates of the point that is located within the variable `nodes`.
    """
    numElems = elem_coords.shape[0]
    nodesPerElem = elem_coords.shape[1]
    output = np.empty((numElems, nodesPerElem), dtype=np.int32)
    for i in xrange(numElems):
        for j in xrange(nodesPerElem):
            output[i][j] = np.where(np.all(nodes == elem_coords[i, j], axis=1))[0]

    return output


def getYZRotation(vec):
    """
    Returns the rotations around the z and y axes required such that the local x-axis is aligned along the vector's length.

    Parameters
    ----------
    vec : Numpy array with in the form of [x,y,z] corresponding to vector's scale values along the global coordinate axes.
    """
    # get the length of the x-y components of the vector
    xy_length = sqrt(vec[0] * vec[0] + vec[1] * vec[1])
    #get the length of the entire vector
    vec_length = sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2])
    #calculate the rotations about the y and z axes:
    #if there is no xyLength then the element is already aligned in the z direction
    if xy_length == 0:
        zrot = 0.0
        yrot = np.pi / 2.0
        #check if vector along z points up or down
        if vec[0] < 0:
            yrot *= -1.0
    #if not aligned with z-axis calculate the rotations normally
    else:
        zrot = -acos(vec[0] / xy_length)
        val = (vec[0] * vec[0] + vec[1] * vec[1]) / (xy_length * vec_length)
        #check that the value is in the valid range of acos (mainly for rounding errors when value approaches +/-1)
        val = min(1, max(val, -1))
        yrot = acos(val)
    #adjust rotation to correct sign based on quadrant the vector is in
    if vec[1] < 0.0:
        zrot *= -1.0
    if vec[2] < 0.0:
        yrot *= -1.0
    #put results in numpy array
    return np.array([yrot, zrot])


def rotateCoords(xRot, yRot, zRot):
    """
    Returns the local x, y, and z coordinate vectors that result from rotating the global x,y, and z vectors by the amounts xRot, yRot, and zRot, respectively.

    Parameters
    ----------
    xRot : Clockwise rotation in radians to apply about the x-axis
    yRot : Clockwise rotation in radians to apply about the y-axis
    zRot : Clockwise rotation in radians to apply about the z-axis
    """
    # define the unit vectors along the global x,y,z directions
    globalCoords = np.matrix([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    #get rotation matrices
    Rx = np.matrix([[1.0, 0.0, 0.0],
                    [0.0, cos(xRot), -sin(xRot)],
                    [0.0, sin(xRot), cos(xRot)]])
    Ry = np.matrix([[cos(yRot), 0.0, sin(yRot)],
                    [0.0, 1.0, 0.0],
                    [-sin(yRot), 0, cos(yRot)]])
    Rz = np.matrix([[cos(zRot), -sin(zRot), 0.0],
                    [sin(zRot), cos(zRot), 0.0],
                    [0.0, 0.0, 1.0]])
    #combine rotation matrices and apply to global coordinates
    return (Rx * Ry * Rz) * globalCoords


def rotate(pt1, pt2, xRot):
    """
    A vector is defined from pt1 to pt2, and a local coordinate system is defined where the x-axis runs parallel to the vector and the z-axis is coplanar with the global z-axis.
    The variable xRot will the apply a rotation about the x-axis by the specified amount and return the resultant unit vectors in the rotated coordinate system.

    Parameters
    ----------
    pt1  : Point in global (x,y,z) space
    pt2  : Point in global (x,y,z) space
    xRot : Clockwise rotation about the x-axis that should be applied in local coordinate space.
    """
    vec = pt2 - pt1
    yzRot = getYZRotation(vec)
    return rotateCoords(xRot, yzRot[0], yzRot[1])


def getMajorAxis(pt1, pt2, xRot):
    """
    A vector is defined from pt1 to pt2, and a local coordinate system is defined where the x-axis runs parallel to the vector and the z-axis is coplanar with the global z-axis.
    The variable xRot will the apply a rotation about the x-axis by the specified amount and return the resultant z-axis unit vector in the local rotated coordinate system.

    Parameters
    ----------
    pt1  : Point in global (x,y,z) space
    pt2  : Point in global (x,y,z) space
    xRot : Clockwise rotation about the x-axis that should be applied in local coordinate space.
    """
    vec = pt2 - pt1
    yzRot = getYZRotation(vec)
    return rotateCoords(xRot, yzRot[0], yzRot[1])[2]


def assignMajorAxes(mesh, xRot):
    """
    Takes a mesh an input and returns an array holding the major bending axis (in [x,y,z] form) for each element.

    Parameters
    ----------
    mesh : an object which has the member variable nodes which holds the (x,y,z) coordinates of each node in the mesh and
    the member variable elems holding an array of describing the connectivity of the nodal coordinates.
    xRot : The amount the major axis is rotated clockwise relative to the z-axis of the element if it were aligned along the x-axis
    and you were looking down the element from the end point to the first point.
    """
    numElems = mesh.elems.shape[0]
    majorAxes = np.empty((numElems, 3))
    for i in range(numElems):
        pt1 = mesh.nodes[mesh.elems[i, 0]]
        pt2 = mesh.nodes[mesh.elems[i, 1]]
        majorAxes[i] = getMajorAxis(pt1, pt2, xRot)
    # set small values in transformation matrix to 0
    majorAxes[np.abs(majorAxes) < 1e-15] = 0
    return majorAxes