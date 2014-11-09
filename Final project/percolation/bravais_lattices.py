__author__ = 'Ryan'

from utility_functions import *

'''
Parent Bravais lattice class
'''

class BravaisLattice(object):
    def __init__(self, a=1.0, b=1.0, c=1.0, alpha=np.pi / 2.0, beta=np.pi / 2.0, gamma=np.pi / 2.0, numElems=1):
        self.a = a
        self.b = b
        self.c = c
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.numElems = numElems
        self.volume = sqrt(
            1.0 - cos(alpha) ** 2 - cos(beta) ** 2 - cos(gamma) ** 2 + 2.0 * cos(alpha) * cos(beta) * cos(gamma))
        self.transformMatrix = np.array([[a, b * cos(gamma), c * cos(beta)],
                                         [0.0, b * sin(gamma), c * (cos(alpha) - cos(beta) * cos(gamma)) / sin(gamma)],
                                         [0.0, 0.0, c * self.volume / sin(gamma)]])
        # set small values in transformation matrix to 0
        self.transformMatrix[np.abs(self.transformMatrix) < 1e-15] = 0

    def createBasisMesh(self):
        # create arrays to hold the elements/nodes along each span of basis strut
        spanNodes = np.empty((self.numElems + 1, 3))
        spanElems = np.empty((self.numElems, 2, 3))

        #loop through each span and create numElems elements along the strut
        for i in xrange(len(self.basisElems)):
            pt1 = self.basisNodes[self.basisElems[i][0]]
            pt2 = self.basisNodes[self.basisElems[i][1]]
            xPts = np.linspace(pt1[0], pt2[0], self.numElems + 1)
            yPts = np.linspace(pt1[1], pt2[1], self.numElems + 1)
            zPts = np.linspace(pt1[2], pt2[2], self.numElems + 1)

            #form (x,y,z) coordinates
            for j in xrange(self.numElems + 1):
                spanNodes[j, 0] = xPts[j]
                spanNodes[j, 1] = yPts[j]
                spanNodes[j, 2] = zPts[j]
            #add form element list for span (contains [x,y,z] entries for each node in the element)
            for j in xrange(self.numElems):
                spanElems[j, 0, :] = spanNodes[j]
                spanElems[j, 1, :] = spanNodes[j + 1]
            #add the spans nodes/elements to the instance's nodes/elements
            self.nodes[i] = spanNodes
            self.elems[i] = spanElems

        #flatten the node list into an array of (x,y,z) points
        self.nodes = self.nodes.reshape((self.basisElems.shape[0] * (self.numElems + 1), 3))
        #cleanup any duplicate nodes
        self.nodes = deleteDuplicates(self.nodes)
        # flatten the element list which contains arrays of elements list for each
        # span to one array with all the elements (still contains (x,y,z) coordinates as entries)
        elemsShape = self.elems.shape
        self.elems = self.elems.reshape((elemsShape[0] * elemsShape[1], elemsShape[2], elemsShape[3]))
        #replace the (x,y,z) coordinate with the index it occurs in the node list
        self.elems = replaceWithIndex(self.nodes, self.elems)
        #transform nodes from fraction coordinate system to Cartesion coordinates
        for i in xrange(self.nodes.shape[0]):
            self.nodes[i] = self.transformMatrix.dot(self.nodes[i])

'''
Subclasses
'''

class SimpleCubic(BravaisLattice):
    def __init__(self, a, numElems=1):
        BravaisLattice.__init__(self, a=a, b=a, c=a, numElems=numElems)
        self.basisNodes = np.array([[0.0, 0.0, 0.0],
                                    [0.0, 0.0, 1.0],
                                    [0.0, 1.0, 0.0],
                                    [1.0, 0.0, 0.0],
                                    [0.0, 1.0, 1.0],
                                    [1.0, 0.0, 1.0],
                                    [1.0, 1.0, 0.0],
                                    [1.0, 1.0, 1.0]])
        self.basisElems = np.array([[0, 1],
                                    [0, 2],
                                    [0, 3],
                                    [1, 4],
                                    [1, 5],
                                    [2, 4],
                                    [2, 6],
                                    [3, 5],
                                    [3, 6],
                                    [4, 7],
                                    [5, 7],
                                    [6, 7]])
        self.nodes = np.empty((self.basisElems.shape[0], numElems + 1, 3))
        self.elems = np.empty((self.basisElems.shape[0], numElems, 2, 3))
        self.createBasisMesh()


class FCC(BravaisLattice):
    def __init__(self, a, numElems=1):
        BravaisLattice.__init__(self, a=a, b=a, c=a, numElems=numElems)
        self.basisNodes = np.array([[0.0, 0.0, 0.0],
                                    [0.0, 0.0, 1.0],
                                    [0.0, 1.0, 0.0],
                                    [1.0, 0.0, 0.0],
                                    [0.0, 1.0, 1.0],
                                    [1.0, 0.0, 1.0],
                                    [1.0, 1.0, 0.0],
                                    [1.0, 1.0, 1.0],
                                    [0.0, 0.5, 0.5],
                                    [0.5, 0.0, 0.5],
                                    [0.5, 0.5, 0.0],
                                    [1.0, 0.5, 0.5],
                                    [0.5, 1.0, 0.5],
                                    [0.5, 0.5, 1.0]])
        self.basisElems = np.array(
            [[0, 1], [0, 2], [0, 3], [1, 4], [1, 5], [2, 4], [2, 6], [3, 5], [3, 6], [4, 7], [5, 7], [6, 7],
             [0, 8], [1, 8], [2, 8], [4, 8], [0, 9], [1, 9], [3, 9], [5, 9], [0, 10], [2, 10], [3, 10], [6, 10],
             [3, 11], [5, 11], [6, 11], [7, 11], [2, 12], [4, 12], [6, 12], [7, 12], [1, 13], [4, 13], [5, 13],
             [7, 13]])
        self.nodes = np.empty((self.basisElems.shape[0], numElems + 1, 3))
        self.elems = np.empty((self.basisElems.shape[0], numElems, 2, 3))
        self.createBasisMesh()


class BCC(BravaisLattice):
    def __init__(self, a, numElems=1):
        BravaisLattice.__init__(self, a=a, b=a, c=a, numElems=numElems)
        self.basisNodes = np.array([[0.0, 0.0, 0.0],
                                    [0.0, 0.0, 1.0],
                                    [0.0, 1.0, 0.0],
                                    [1.0, 0.0, 0.0],
                                    [0.0, 1.0, 1.0],
                                    [1.0, 0.0, 1.0],
                                    [1.0, 1.0, 0.0],
                                    [1.0, 1.0, 1.0],
                                    [0.5, 0.5, 0.5]])
        self.basisElems = np.array(
            [[0, 1], [0, 2], [0, 3], [1, 4], [1, 5], [2, 4], [2, 6], [3, 5], [3, 6], [4, 7], [5, 7], [6, 7],
             [0, 8], [1, 8], [2, 8], [3, 8], [4, 8], [5, 8], [6, 8], [7, 8]])
        self.nodes = np.empty((self.basisElems.shape[0], numElems + 1, 3))
        self.elems = np.empty((self.basisElems.shape[0], numElems, 2, 3))
        self.createBasisMesh()


class Hexagonal(BravaisLattice):
    def __init__(self, a, c, numElems=1):
        BravaisLattice.__init__(self, a=a, b=a, c=c, gamma=np.pi / 3, numElems=numElems)
        self.basisNodes = np.array([[0.0, 0.0, 0.0],
                                    [0.0, 0.0, 1.0],
                                    [0.0, 1.0, 0.0],
                                    [1.0, 0.0, 0.0],
                                    [0.0, 1.0, 1.0],
                                    [1.0, 0.0, 1.0],
                                    [1.0, 1.0, 0.0],
                                    [1.0, 1.0, 1.0]])
        self.basisElems = np.array(
            [[0, 1], [0, 2], [0, 3], [1, 4], [1, 5], [2, 4], [2, 6], [3, 5], [3, 6], [4, 7], [5, 7], [6, 7],
             [2, 3], [4, 5]])
        self.nodes = np.empty((self.basisElems.shape[0], numElems + 1, 3))
        self.elems = np.empty((self.basisElems.shape[0], numElems, 2, 3))
        self.createBasisMesh()


class Rhombohedral(BravaisLattice):
    def __init__(self, a, alpha, numElems=1):
        BravaisLattice.__init__(self, a=a, b=a, c=a, alpha=alpha, beta=alpha, gamma=alpha, numElems=numElems)
        self.basisNodes = np.array([[0.0, 0.0, 0.0],
                                    [0.0, 0.0, 1.0],
                                    [0.0, 1.0, 0.0],
                                    [1.0, 0.0, 0.0],
                                    [0.0, 1.0, 1.0],
                                    [1.0, 0.0, 1.0],
                                    [1.0, 1.0, 0.0],
                                    [1.0, 1.0, 1.0]])
        self.basisElems = np.array(
            [[0, 1], [0, 2], [0, 3], [1, 4], [1, 5], [2, 4], [2, 6], [3, 5], [3, 6], [4, 7], [5, 7], [6, 7]])
        self.nodes = np.empty((self.basisElems.shape[0], numElems + 1, 3))
        self.elems = np.empty((self.basisElems.shape[0], numElems, 2, 3))
        self.createBasisMesh()


class PrimitiveTetragonal(BravaisLattice):
    def __init__(self, a, c, numElems=1):
        BravaisLattice.__init__(self, a=a, b=a, c=c, numElems=numElems)
        self.basisNodes = np.array([[0.0, 0.0, 0.0],
                                    [0.0, 0.0, 1.0],
                                    [0.0, 1.0, 0.0],
                                    [1.0, 0.0, 0.0],
                                    [0.0, 1.0, 1.0],
                                    [1.0, 0.0, 1.0],
                                    [1.0, 1.0, 0.0],
                                    [1.0, 1.0, 1.0]])
        self.basisElems = np.array(
            [[0, 1], [0, 2], [0, 3], [1, 4], [1, 5], [2, 4], [2, 6], [3, 5], [3, 6], [4, 7], [5, 7], [6, 7]])
        self.nodes = np.empty((self.basisElems.shape[0], numElems + 1, 3))
        self.elems = np.empty((self.basisElems.shape[0], numElems, 2, 3))
        self.createBasisMesh()


class BodyCenteredTetragonal(BravaisLattice):
    def __init__(self, a, c, numElems=1):
        BravaisLattice.__init__(self, a=a, b=a, c=c, numElems=numElems)
        self.basisNodes = np.array([[0.0, 0.0, 0.0],
                                    [0.0, 0.0, 1.0],
                                    [0.0, 1.0, 0.0],
                                    [1.0, 0.0, 0.0],
                                    [0.0, 1.0, 1.0],
                                    [1.0, 0.0, 1.0],
                                    [1.0, 1.0, 0.0],
                                    [1.0, 1.0, 1.0],
                                    [0.5, 0.5, 0.5]])
        self.basisElems = np.array(
            [[0, 1], [0, 2], [0, 3], [1, 4], [1, 5], [2, 4], [2, 6], [3, 5], [3, 6], [4, 7], [5, 7], [6, 7],
             [0, 8], [1, 8], [2, 8], [3, 8], [4, 8], [5, 8], [6, 8], [7, 8]])
        self.nodes = np.empty((self.basisElems.shape[0], numElems + 1, 3))
        self.elems = np.empty((self.basisElems.shape[0], numElems, 2, 3))
        self.createBasisMesh()


class PrimitiveOrthorhombic(BravaisLattice):
    def __init__(self, a, b, c, numElems=1):
        BravaisLattice.__init__(self, a=a, b=b, c=c, numElems=numElems)
        self.basisNodes = np.array([[0.0, 0.0, 0.0],
                                    [0.0, 0.0, 1.0],
                                    [0.0, 1.0, 0.0],
                                    [1.0, 0.0, 0.0],
                                    [0.0, 1.0, 1.0],
                                    [1.0, 0.0, 1.0],
                                    [1.0, 1.0, 0.0],
                                    [1.0, 1.0, 1.0]])
        self.basisElems = np.array(
            [[0, 1], [0, 2], [0, 3], [1, 4], [1, 5], [2, 4], [2, 6], [3, 5], [3, 6], [4, 7], [5, 7], [6, 7]])
        self.nodes = np.empty((self.basisElems.shape[0], numElems + 1, 3))
        self.elems = np.empty((self.basisElems.shape[0], numElems, 2, 3))
        self.createBasisMesh()


class BodyCenteredOrthorhombic(BravaisLattice):
    def __init__(self, a, b, c, numElems=1):
        BravaisLattice.__init__(self, a=a, b=b, c=c, numElems=numElems)
        self.basisNodes = np.array([[0.0, 0.0, 0.0],
                                    [0.0, 0.0, 1.0],
                                    [0.0, 1.0, 0.0],
                                    [1.0, 0.0, 0.0],
                                    [0.0, 1.0, 1.0],
                                    [1.0, 0.0, 1.0],
                                    [1.0, 1.0, 0.0],
                                    [1.0, 1.0, 1.0],
                                    [0.5, 0.5, 0.5]])
        self.basisElems = np.array(
            [[0, 1], [0, 2], [0, 3], [1, 4], [1, 5], [2, 4], [2, 6], [3, 5], [3, 6], [4, 7], [5, 7], [6, 7],
             [0, 8], [1, 8], [2, 8], [3, 8], [4, 8], [5, 8], [6, 8], [7, 8]])
        self.nodes = np.empty((self.basisElems.shape[0], numElems + 1, 3))
        self.elems = np.empty((self.basisElems.shape[0], numElems, 2, 3))
        self.createBasisMesh()


class BaseCenteredOrthorhombic(BravaisLattice):
    def __init__(self, a, b, c, numElems=1):
        BravaisLattice.__init__(self, a=a, b=b, c=c, numElems=numElems)
        self.basisNodes = np.array([[0.0, 0.0, 0.0],
                                    [0.0, 0.0, 1.0],
                                    [0.0, 1.0, 0.0],
                                    [1.0, 0.0, 0.0],
                                    [0.0, 1.0, 1.0],
                                    [1.0, 0.0, 1.0],
                                    [1.0, 1.0, 0.0],
                                    [1.0, 1.0, 1.0],
                                    [0.5, 0.5, 0.0],
                                    [0.5, 0.5, 1.0]])
        self.basisElems = np.array(
            [[0, 1], [0, 2], [0, 3], [1, 4], [1, 5], [2, 4], [2, 6], [3, 5], [3, 6], [4, 7], [5, 7], [6, 7],
             [0, 8], [2, 8], [3, 8], [6, 8], [1, 9], [4, 9], [5, 9], [7, 9]])
        self.nodes = np.empty((self.basisElems.shape[0], numElems + 1, 3))
        self.elems = np.empty((self.basisElems.shape[0], numElems, 2, 3))
        self.createBasisMesh()


class FaceCenteredOrthorhombic(BravaisLattice):
    def __init__(self, a, b, c, numElems=1):
        BravaisLattice.__init__(self, a=a, b=b, c=c, numElems=numElems)
        self.basisNodes = np.array([[0.0, 0.0, 0.0],
                                    [0.0, 0.0, 1.0],
                                    [0.0, 1.0, 0.0],
                                    [1.0, 0.0, 0.0],
                                    [0.0, 1.0, 1.0],
                                    [1.0, 0.0, 1.0],
                                    [1.0, 1.0, 0.0],
                                    [1.0, 1.0, 1.0],
                                    [0.0, 0.5, 0.5],
                                    [0.5, 0.0, 0.5],
                                    [0.5, 0.5, 0.0],
                                    [1.0, 0.5, 0.5],
                                    [0.5, 1.0, 0.5],
                                    [0.5, 0.5, 1.0]])
        self.basisElems = np.array(
            [[0, 1], [0, 2], [0, 3], [1, 4], [1, 5], [2, 4], [2, 6], [3, 5], [3, 6], [4, 7], [5, 7], [6, 7],
             [0, 8], [1, 8], [2, 8], [4, 8], [0, 9], [1, 9], [3, 9], [5, 9], [0, 10], [2, 10], [3, 10], [6, 10],
             [3, 11], [5, 11], [6, 11], [7, 11], [2, 12], [4, 12], [6, 12], [7, 12], [1, 13], [4, 13], [5, 13],
             [7, 13]])
        self.nodes = np.empty((self.basisElems.shape[0], numElems + 1, 3))
        self.elems = np.empty((self.basisElems.shape[0], numElems, 2, 3))
        self.createBasisMesh()


class PrimitiveMonoclinic(BravaisLattice):
    def __init__(self, a, b, c, beta, numElems=1):
        BravaisLattice.__init__(self, a=a, b=b, c=c, beta=beta, numElems=numElems)
        self.basisNodes = np.array([[0.0, 0.0, 0.0],
                                    [0.0, 0.0, 1.0],
                                    [0.0, 1.0, 0.0],
                                    [1.0, 0.0, 0.0],
                                    [0.0, 1.0, 1.0],
                                    [1.0, 0.0, 1.0],
                                    [1.0, 1.0, 0.0],
                                    [1.0, 1.0, 1.0]])
        self.basisElems = np.array(
            [[0, 1], [0, 2], [0, 3], [1, 4], [1, 5], [2, 4], [2, 6], [3, 5], [3, 6], [4, 7], [5, 7], [6, 7]])
        self.nodes = np.empty((self.basisElems.shape[0], numElems + 1, 3))
        self.elems = np.empty((self.basisElems.shape[0], numElems, 2, 3))
        self.createBasisMesh()


class BaseCenteredMonoclinic(BravaisLattice):
    def __init__(self, a, b, c, beta, numElems=1):
        BravaisLattice.__init__(self, a=a, b=b, c=c, beta=beta, numElems=numElems)
        self.basisNodes = np.array([[0.0, 0.0, 0.0],
                                    [0.0, 0.0, 1.0],
                                    [0.0, 1.0, 0.0],
                                    [1.0, 0.0, 0.0],
                                    [0.0, 1.0, 1.0],
                                    [1.0, 0.0, 1.0],
                                    [1.0, 1.0, 0.0],
                                    [1.0, 1.0, 1.0],
                                    [0.5, 0.5, 0.0],
                                    [0.5, 0.5, 1.0]])
        self.basisElems = np.array(
            [[0, 1], [0, 2], [0, 3], [1, 4], [1, 5], [2, 4], [2, 6], [3, 5], [3, 6], [4, 7], [5, 7], [6, 7],
             [0, 8], [2, 8], [3, 8], [6, 8], [1, 9], [4, 9], [5, 9], [7, 9]])
        self.nodes = np.empty((self.basisElems.shape[0], numElems + 1, 3))
        self.elems = np.empty((self.basisElems.shape[0], numElems, 2, 3))
        self.createBasisMesh()


class Triclinic(BravaisLattice):
    def __init__(self, a, b, c, alpha, beta, gamma, numElems=1):
        BravaisLattice.__init__(self, a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma, numElems=numElems)
        self.basisNodes = np.array([[0.0, 0.0, 0.0],
                                    [0.0, 0.0, 1.0],
                                    [0.0, 1.0, 0.0],
                                    [1.0, 0.0, 0.0],
                                    [0.0, 1.0, 1.0],
                                    [1.0, 0.0, 1.0],
                                    [1.0, 1.0, 0.0],
                                    [1.0, 1.0, 1.0]])
        self.basisElems = np.array([[0, 1],
                                    [0, 2],
                                    [0, 3],
                                    [1, 4],
                                    [1, 5],
                                    [2, 4],
                                    [2, 6],
                                    [3, 5],
                                    [3, 6],
                                    [4, 7],
                                    [5, 7],
                                    [6, 7]])
        self.nodes = np.empty((self.basisElems.shape[0], numElems + 1, 3))
        self.elems = np.empty((self.basisElems.shape[0], numElems, 2, 3))
        self.createBasisMesh()
