import pyvista as pv

def remove_duplicate_cells(netw):
    cells = np.array(netw.cells.reshape(-1, 3))[:,1:]
    cells.sort(axis=1)
    unique_cells = np.unique(cells, axis=0)
    netw.cells = np.pad(unique_cells, pad_width=((0,0), (1,0)), constant_values=2)
    
def read_vtk_network(filename, rescale_mm2m=True):
    """Read the VTK file given by filename, return a FEniCS 1D Mesh representing the network, a FEniCS MeshFunction (double) representing the radius of each vessel segment (defined over the mesh cells), and a FEniCS MeshFunction (size_t) defining the roots of the network (defined over the mesh vertices, roots are labelled by 2 or 1.) 

    rescale_mm2m is set to True by default, in which case the information on file is rescaled by a factor 1.e-3. MER: This is error-prone, I suggest no rescaling behind the scenes.
"""
    print("Reading network mesh from %s" % filename)
    netw = pv.read(filename)
    if rescale_mm2m:
        netw.points *= 1e-3 # scale to m
    remove_duplicate_cells(netw)
    mesh = Mesh()
    ed = MeshEditor()
    ed.open(mesh, 'interval', 1, 3)
    ed.init_vertices(netw.number_of_points)
    cells = netw.cells.reshape((-1,3))
    ed.init_cells(cells.shape[0])

    for vid, v in enumerate(netw.points):
        ed.add_vertex(vid, v)
    for cid, c in enumerate(cells[:,1:]):
        ed.add_cell(cid, c)
    ed.close()

    # Cell Function
    radii = MeshFunction('double', mesh, 1, 0)
    roots = MeshFunction('size_t', mesh, 0, 0)

    roots.array()[:] = netw["root"]
    netw = netw.point_data_to_cell_data()

    if rescale_mm2m:
        radii.array()[:] = netw["radius"] * 1e-3 # scale to m
    else:
        radii.array()[:] = netw["radius"]

    print("... with %d nodes, %d edges" % (mesh.num_vertices(), mesh.num_cells()))
    return mesh, radii, roots
