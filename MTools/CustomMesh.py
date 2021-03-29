import os
import sys
import numpy as np
import uvw as uvw
from tqdm import tqdm
import vtk as vtk
import dolfin as dlf
import logging


class MeshCreator():
    """Creates custom dolfin mesh"""

    def __init__(self, log_level = 10):
        #configure loggin
        logging.basicConfig(format="MeshCreator - %(levelname)s - %(message)s", level = log_level)

    def create_dolfin_mesh(self, tp, N, size):
        self.size = size
        if tp == 'hex':
            return self._create_hex_mesh(N)
    
    def _create_hex_mesh(self, N):
        """Creates custom 3D hex mesh with of size with elements N

        :N: Number of points (number of elements - 1)
        :returns: Dolfin Mesh
        """
        x = np.linspace(0, self.size, N)

        mesh = dlf.Mesh()
        editor = dlf.MeshEditor()
        editor.open(mesh, 'hexahedron', 3, 3)
        mat_vert = np.full((N, N, N), -1)

        def get_vertex_id(mat_vert, xi, yi, zi):
            """
            input: 
                xi,yi,zi: vertex indizes 
                mat_Vert: matrix storing the indices of the vetrices (-1 -> not used)
            return: vertex id
            """
            return mat_vert[xi, yi, zi]

        vertindex = np.full((N, N, N), 0)
        matindex = np.full((N-1, N-1, N-1), 1)
        #check which vertrices are needed -> can be done smarter
        logging.info("Check which vertices are needed")
        for zi in tqdm(range(N-1), ncols=80):
            for yi in range(N-1):
                for xi in range(N-1):
                    if matindex[xi, yi, zi] >= 1:
                        vertindex[xi, yi, zi] = 1
                        vertindex[xi + 1, yi, zi] = 1
                        vertindex[xi, yi + 1, zi] = 1
                        vertindex[xi + 1, yi + 1, zi] = 1
                        vertindex[xi, yi, zi + 1] = 1
                        vertindex[xi + 1, yi, zi + 1] = 1
                        vertindex[xi, yi + 1, zi + 1] = 1
                        vertindex[xi + 1, yi + 1, zi + 1] = 1

        editor.init_vertices(np.sum(vertindex))
        editor.init_cells(np.sum(matindex))

        #create all nodes/vertices
        logging.info('Init {} vertices:'.format(np.sum(vertindex)))
        vertex_id = 0
        for zi in tqdm(range(N), ncols=80):
            zcoor = x[zi]
            for yi in range(N):
                ycoor = x[yi]
                for xi in range(N):
                    xcoor = x[xi]
                    if vertindex[xi, yi, zi] == 1:
                        editor.add_vertex(vertex_id, [xcoor, ycoor, zcoor])
                        mat_vert[xi, yi, zi] = vertex_id
                        vertex_id += 1

        #create elements/cells
        logging.info('Init {} cells:'.format(np.sum(matindex)))
        cell_id = 0
        for zi in tqdm(range(N-1), ncols=80):
            for yi in range(N-1):
                for xi in range(N-1):
                    if matindex[xi, yi, zi] >= 1:
                        ind0 = get_vertex_id(mat_vert, xi, yi, zi)
                        ind1 = get_vertex_id(mat_vert, xi + 1, yi, zi)
                        ind2 = get_vertex_id(mat_vert, xi, yi + 1, zi)
                        ind3 = get_vertex_id(mat_vert, xi + 1, yi + 1, zi)
                        ind4 = get_vertex_id(mat_vert, xi, yi, zi + 1)
                        ind5 = get_vertex_id(mat_vert, xi + 1, yi, zi + 1)
                        ind6 = get_vertex_id(mat_vert, xi, yi + 1, zi + 1)
                        ind7 = get_vertex_id(mat_vert, xi + 1, yi + 1, zi + 1)

                        cell_index = [ind0, ind1, ind2,
                                      ind3, ind4, ind5, ind6, ind7]

                        #Debug check: is index in correct order
                        if not ind0 < ind1 < ind2 < ind3 < ind4 < ind5 < ind6 < ind7:
                            print(cell_id, cell_index)

                        editor.add_cell(cell_id, cell_index)
                        cell_id += 1

        editor.close(order=True)
        return mesh
