import os
import sys
import numpy as np
import uvw as uvw
from tqdm import tqdm
import vtk as vtk
import dolfin as dlf
import logging


class VTKSampler():
    """Creates dolfin hexhedron mesh from vtk file"""

    def __init__(self, name, path='', write=True, autoload=True, log_level = 10):
        #configure loggin
        logging.basicConfig(format="VTKSampler - %(levelname)s - %(message)s", level = log_level)
    
        #set file paths
        self.vtk_file_path = os.path.join(path, name + '.vtk')
        self.npz_file_path = os.path.join(path, name + '.npz')
        logging.info("Set the vtk file path to: {}".format(self.vtk_file_path))
        logging.info("Set the npz file path to: {}".format(self.npz_file_path))

        if not (os.path.isfile(self.vtk_file_path) or os.path.isfile(self.npz_file_path)):
            logging.critical("Path for mesh is wrong, either no npz or vtk file")
            sys.exit()

    def sample(self, N, write=True, load=True):
        """
        Creates a vector x and a matrix matindex which define the
        discretization and the volume elements located inside the vtk
        N: number of vertices per edge
        """
        if load and os.path.isfile(self.npz_file_path):
            loaded = np.load(self.npz_file_path)
            self.x = loaded['x']
            self.matindex = loaded['matindex']
            logging.info("Found npz file, loaded numpy data with size N={}".format(len(self.x)-1))
            return

        elif os.path.isfile(self.vtk_file_path):
            self._sample_vtk(N)
            if write: 
                logging.info("Writing sampled vkt to {}".format(self.vtk_file_path))
                self._writeState()

        else:
            logging.critical("Path for mesh is wrong, either no npz or vtk file")
            sys.exit()

    def _loadState(self):
        loaded = np.load(self.npz_file_path)
        self.x = loaded['x']
        self.matindex = loaded['matindex']

    def _writeState(self):
        np.savez_compressed(self.npz_file_path, x=self.x,
                            matindex=self.matindex)

    def _sample_vtk(self, N):
        """Defines discretizatioon and checks if volume elements are inside the
        vtk mesh by comparing their center point

        N: Number of vertices (number of volume elements + 1)
        returns: x, matindex
         
        -----------------------------------------------------------------------

        self.x: vector defining the discretization

        self.matindex: matrix (N-1)x(N-1)x(N-1) of {0,1} indicating the
        material of the volume cell (0: material 0, 1: matrial 1)

        """
        #stores the matrial index for each element
        matindex = np.zeros((N-1, N-1, N-1), dtype=int)

        reader = vtk.vtkUnstructuredGridReader()
        reader.SetFileName(self.vtk_file_path)
        reader.ReadAllScalarsOn()
        reader.ReadAllVectorsOn()
        reader.Update()
        mesh = reader.GetOutput()
        bounds = mesh.GetBounds()  # [x0, xmax, y0, ymax, z0, zmax]

        #get dimensions of problem
        delX = bounds[1] - bounds[0]
        delY = bounds[3] - bounds[2]
        delZ = bounds[5] - bounds[4]
        a = np.min([delX, delY, delZ])

        #crop to size of smallest edge
        x = np.linspace(0.0, a, N)

        #setup to locate the volume cells in the vtk file
        genCell = vtk.vtkGenericCell()
        pc = [0, 0, 0]  # no further start information is known
        weights = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # interpolation function
        subID = vtk.reference(0)  # 0 since it has no more basic cell
        cellmat = mesh.GetCellData().GetArray('material')

        xcoor = np.zeros(3)
        logging.info("Checking the placement of volume elements in the vtk file")
        for i in tqdm(range(N-1), ncols=80):
            xcoor[0] = (x[i]+x[i+1])/2
            for j in range(N-1):
                xcoor[1] = (x[j]+x[j+1])/2
                for k in range(N-1):
                    xcoor[2] = (x[k]+x[k+1])/2
                    cellno = mesh.FindCell(
                        xcoor, genCell, -1, 0.0001, subID, pc, weights)
                    if cellno != -1:  # -1 -> cell not found
                        if int(cellmat.GetValue(cellno)) == 2:
                            # center of volume element -> material = 1
                            matindex[i, j, k] = 1
        self.x = x
        self.matindex = matindex

    def create_dolfin_mesh(self):
        """
        Creates a three dimensional regular fenics mesh of 2 materials, defined by
        a spacing vector and an index vector

        :x: axis vector, defining the uniform spacing of the nodes
        :matindex: three dimensional matrix of {0;1} indicating the material allocation
        :returns: fenics mesh
        """
        N = len(self.x)

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

        #check which vertrices are needed -> can be done smarter
        logging.info("Check which vertices are needed")
        for zi in tqdm(range(N-1), ncols=80):
            for yi in range(N-1):
                for xi in range(N-1):
                    if self.matindex[xi, yi, zi] >= 1:
                        vertindex[xi, yi, zi] = 1
                        vertindex[xi + 1, yi, zi] = 1
                        vertindex[xi, yi + 1, zi] = 1
                        vertindex[xi + 1, yi + 1, zi] = 1
                        vertindex[xi, yi, zi + 1] = 1
                        vertindex[xi + 1, yi, zi + 1] = 1
                        vertindex[xi, yi + 1, zi + 1] = 1
                        vertindex[xi + 1, yi + 1, zi + 1] = 1

        editor.init_vertices(np.sum(vertindex))
        editor.init_cells(np.sum(self.matindex))

        #create all nodes/vertices
        logging.info('Init {} vertices:'.format(str(N**3)))
        vertex_id = 0
        for zi in tqdm(range(N), ncols=80):
            zcoor = self.x[zi]
            for yi in range(N):
                ycoor = self.x[yi]
                for xi in range(N):
                    xcoor = self.x[xi]
                    if vertindex[xi, yi, zi] == 1:
                        editor.add_vertex(vertex_id, [xcoor, ycoor, zcoor])
                        mat_vert[xi, yi, zi] = vertex_id
                        vertex_id += 1

        #create elements/cells
        logging.info('Init {} cells:'.format(str((N-1)**3)))
        cell_id = 0
        for zi in tqdm(range(N-1), ncols=80):
            for yi in range(N-1):
                for xi in range(N-1):
                    if self.matindex[xi, yi, zi] >= 1:
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
        return mesh, max(self.x)
