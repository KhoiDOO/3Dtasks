import numpy as np
import random

class PointSampler(object):
    def __init__(self, output_size):
        assert isinstance(output_size, int)
        self.output_size = output_size
    
    def triangle_area(self, pt1, pt2, pt3):
        side_a = np.linalg.norm(pt1 - pt2)
        side_b = np.linalg.norm(pt2 - pt3)
        side_c = np.linalg.norm(pt3 - pt1)
        s = 0.5 * ( side_a + side_b + side_c)
        return max(s * (s - side_a) * (s - side_b) * (s - side_c), 0)**0.5

    def sample_point(self, pt1, pt2, pt3):
        # barycentric coordinates on a triangle
        # https://mathworld.wolfram.com/BarycentricCoordinates.html
        s, t = sorted([random.random(), random.random()])
        f = lambda i: s * pt1[i] + (t-s)*pt2[i] + (1-t)*pt3[i]
        return (f(0), f(1), f(2))
    
    def __call__(self, mesh):
        verts: np.ndarray # 3 * N
        faces: np.ndarray # 3 * N
        verts, faces = mesh

        areas = np.zeros(faces.shape[-1])

        for i in range(len(areas)):
            areas[i] = self.triangle_area(verts[:, faces[0, i]], verts[:, faces[1, i]], verts[:, faces[2,i]])
            
        sampled_faces = random.choices(faces.T, weights=areas, cum_weights=None, k=self.output_size)
        
        sampled_points = np.zeros((3, self.output_size))

        for i in range(len(sampled_faces)):
            sampled_points[:, i] = self.sample_point(verts[:, sampled_faces[i][0]], verts[:, sampled_faces[i][1]], verts[:, sampled_faces[i][2]])
        
        return sampled_points

def point_sampler(verts: np.ndarray, faces: np.ndarray, output_size: int):
    return PointSampler(output_size)((verts, faces))