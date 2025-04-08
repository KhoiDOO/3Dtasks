import open3d as o3d
import numpy as np
import os

dct = {
    'nut' : "examples/obj/objaverse_nut.obj",
    'pig' : "examples/obj/objaverse_pig.obj",
}

def formgraph(triangles, vertices):
    """
    Form a graph from the triangles and vertices of a mesh.
    The graph is represented as an adjacency matrix.
    """
    num_vertices = vertices.shape[0]
    graph = np.zeros((num_vertices, num_vertices), dtype=np.uint8)

    for triangle in triangles:
        for i in range(3):
            for j in range(i + 1, 3):
                graph[triangle[i], triangle[j]] = 1
                graph[triangle[j], triangle[i]] = 1

    return graph

def lower_arr2mat(arr):
    """
    Convert a 1D array to a lower triangular matrix.
    """
    n = int((1 + (1 + 8 * arr.size) ** 0.5) / 2)
    mat = np.zeros((n, n), dtype=arr.dtype)
    tril_indices = np.tril_indices(n, k=-1)
    mat[tril_indices] = arr
    return mat

def group01(lst):
    idx = 0
    old_idx = 0
    output = []

    check = 0 if lst[0] == 0 else 1

    while idx < len(lst): 
        for i in range(idx, len(lst)):
            if lst[i] == check:
                idx += 1
            else:
                output.append(f'{len(lst[old_idx:idx])}${check}')
                old_idx = idx
                check = 1 if check == 0 else 0
                break

    return output

if __name__ == '__main__':

    nut_mesh = o3d.io.read_triangle_mesh(os.path.join(os.getcwd(), dct['nut']))
    pig_mesh = o3d.io.read_triangle_mesh(os.path.join(os.getcwd(), dct['pig']))

    nut_triangles = np.asarray(nut_mesh.triangles)
    nut_vertices = np.asarray(nut_mesh.vertices)
    pig_triangles = np.asarray(pig_mesh.triangles)
    pig_vertices = np.asarray(pig_mesh.vertices)

    # print(f"nut_triangles: {nut_triangles.shape}")
    # print(f"nut_vertices: {nut_vertices.shape}")
    # print(f"pig_triangles: {pig_triangles.shape}")
    # print(f"pig_vertices: {pig_vertices.shape}")

    # print(f"nut_triangles sample: {nut_triangles[0]}")
    # print(f"pig_triangles sample: {nut_triangles[0]}")

    nut_graph = formgraph(nut_triangles, nut_vertices)
    pig_graph = formgraph(pig_triangles, pig_vertices)

    print(f"nut_graph: {nut_graph.shape}")
    print(f"pig_graph: {pig_graph.shape}")

    print(f"#zero elements in nut_graph: {np.count_nonzero(nut_graph == 0)}/{nut_graph.shape[0]**2}")
    print(f"#zero elements in pig_graph: {np.count_nonzero(pig_graph == 0)}/{pig_graph.shape[0]**2}")

    # print(f'symmetric nut_graph: {np.all(nut_graph == nut_graph.T)}')
    # print(f'symmetric pig_graph: {np.all(pig_graph == pig_graph.T)}')

    flattened_lower_nut_graph = nut_graph[np.tril_indices(nut_graph.shape[0], k=-1)]
    flattened_lower_pig_graph = pig_graph[np.tril_indices(pig_graph.shape[0], k=-1)]
    print(f"Flattened lower_nut_graph: {flattened_lower_nut_graph.shape}")
    print(f"Flattened lower_pig_graph: {flattened_lower_pig_graph.shape}")
    print(f"#zero elements in flattened_lower_nut_graph: {np.count_nonzero(flattened_lower_nut_graph == 0)}/{flattened_lower_nut_graph.size}")
    print(f"#zero elements in flattened_lower_pig_graph: {np.count_nonzero(flattened_lower_pig_graph == 0)}/{flattened_lower_pig_graph.size}")

    recon_lower_nut_graph = lower_arr2mat(flattened_lower_nut_graph)
    reconstructed_nut_graph = recon_lower_nut_graph + recon_lower_nut_graph.T

    print(f"Reconstruction successful: {np.array_equal(nut_graph, reconstructed_nut_graph)}")

    recon_lower_pig_graph = lower_arr2mat(flattened_lower_pig_graph)
    reconstructed_pig_graph = recon_lower_pig_graph + recon_lower_pig_graph.T

    print(f"Reconstruction successful: {np.array_equal(pig_graph, reconstructed_pig_graph)}")

    nut_seq = flattened_lower_nut_graph.tolist()
    pig_seq = flattened_lower_pig_graph.tolist()
    # print(f"nut_seq: {len(nut_seq)}")
    # print(f"pig_seq: {len(pig_seq)}")

    print(f"nut_seq: {nut_seq[:20]}")
    print(f"pig_seq: {pig_seq[:20]}")

    nut_gr_seq = group01(nut_seq)
    pig_gr_seq = group01(pig_seq)
    print(f"nut_gr_seq: {len(nut_gr_seq)}")
    print(f"pig_gr_seq: {len(pig_gr_seq)}")
    # print(f"nut_gr_seq: {nut_gr_seq[:10]}")
    # print(f"pig_gr_seq: {pig_gr_seq[:10]}")

    