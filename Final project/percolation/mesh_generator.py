__author__ = 'Ryan'

from utility_functions import *


class Job:
    def __init__(self, nodes, elems, props=None):
        self.nodes = nodes
        self.elems = elems
        self.props = props

'''
Tile unit cells
'''

def meshBravais(lattice, dimX, dimY, dimZ):
    # form anchors
    anchors = np.empty((dimX * dimY * dimZ, 3))
    idx = 0
    for i in xrange(dimX):
        for j in xrange(dimY):
            for k in xrange(dimZ):
                anchors[idx, 0] = i
                anchors[idx, 1] = j
                anchors[idx, 2] = k
                anchors[idx] = lattice.transformMatrix.dot(anchors[idx])
                idx += 1
    #form element lists, expand lattice points about basis
    elemList = np.empty((dimX * dimY * dimZ, lattice.elems.shape[0], lattice.elems.shape[1], 3))
    nodeList = np.empty((dimX * dimY * dimZ, lattice.nodes.shape[0], lattice.nodes.shape[1]))
    for i in xrange(anchors.shape[0]):
        nodeList[i] = lattice.nodes + anchors[i]
        for j in xrange(elemList.shape[1]):
            for k in xrange(elemList.shape[2]):
                elemList[i, j, k] = nodeList[i][lattice.elems[j, k]]
                #delete duplicates
    nodeList = nodeList.reshape((nodeList.shape[0] * nodeList.shape[1], 3))
    nodeList = deleteDuplicates(nodeList)
    #replace the elemList (x,y,z) coordinates with the index the point occurs in the node list
    elemsShape = elemList.shape
    elemList = elemList.reshape((elemsShape[0] * elemsShape[1], elemsShape[2], elemsShape[3]))
    elemList = replaceWithIndex(nodeList, elemList)
    return Job(nodeList, elemList)