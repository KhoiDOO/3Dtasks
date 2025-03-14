import numpy as np

supported_formats = ['off']

def check_file_format(file_path):
    return file_path.split('.')[-1] in supported_formats

def read_off(file_path):

    if not check_file_format(file_path):
        raise ValueError(f'Unsupported file format: {file_path}, supported formats: {supported_formats}')
    
    with open(file_path, 'r') as file:
        off_header = file.readline().strip()
        if 'OFF' == off_header:
            n_verts, n_faces, __ = tuple([int(s) for s in file.readline().strip().split(' ')])
        else:
            n_verts, n_faces, __ = tuple([int(s) for s in off_header[3:].split(' ')])
        verts = [[float(s) for s in file.readline().strip().split(' ')] for _ in range(n_verts)]
        faces = [[int(s) for s in file.readline().strip().split(' ')][1:] for _ in range(n_faces)]
        return np.array(verts).T, np.array(faces).T

def read_obj(file_path):
    raise NotImplementedError()