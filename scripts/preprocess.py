import os
import glob
import math
import random
import numpy as np
from tqdm import tqdm
import bpy
from multiprocessing import Pool
import h5py

import sys
sys.path.append('.')
from src.utils import read_obj


DATA_DIR = './data'

def get_new_path_name(path, replace_word):
    head, tail = os.path.split(path)
    head = head.replace("/ShapeNetCore.v2", "").replace("/models", "").replace("raw", replace_word)

    head = head.split('/')
    cat  = head[-2]
    name = head[-1]
    ext = tail.split('.')[1]

    new_path = os.path.join(*head[:-2], cat, name + '.' + ext)

    return new_path

def clean_an_obj_file(src_path):
    dst_path = get_new_path_name(src_path, 'preclean')
    os.makedirs(os.path.split(dst_path)[0], exist_ok=True)
    vertices = []
    faces = []
    with open(src_path, 'r') as file:
        for line in file:
            if line == "":
                continue
            split = line.split()
            if len(split) > 0:
                if split[0] == 'v':
                    _, x, y, z = line.split()
                    # x, y, z = map(float, (x, y, z))
                    vertices.append((x, y, z))
                elif split[0] == 'f':
                    vs = split[1:]
                    vs = list(map(lambda v: v.split('/')[0], vs))
                    faces.append(vs)
                else:
                    pass
                    # print("not vertex or face, skip line:", line[:-1])
            else:
                pass
                # print("skip line: ", line[-1])
    
    #write file
    with open(dst_path, 'w') as file:
        for vertex in vertices:
            x, y, z = vertex
            file.write(f'v {x} {y} {z}\n')
        for face in faces:
            file.write('f ' + ' '.join(face) + '\n')
    return dst_path
        
def decimate(src_path, dst_path, angle=5):
    os.makedirs(os.path.split(dst_path)[0], exist_ok=True)
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    bpy.ops.import_scene.obj(filepath=src_path, filter_glob="*.obj")
    selected_objects = bpy.context.selected_objects
    assert len(selected_objects) == 1
    obj = selected_objects[0]
    bpy.context.view_layer.objects.active = obj
    
    n_faces_before = len(obj.data.polygons)
    bpy.ops.object.shade_flat()
    bpy.ops.object.editmode_toggle()
    bpy.ops.mesh.select_mode(use_extend=False, use_expand=False, type='FACE')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.editmode_toggle()
    modifier = obj.modifiers.new("decimate", "DECIMATE")
    modifier.decimate_type = "DISSOLVE"
    modifier.angle_limit = angle * math.pi / 180
    bpy.ops.object.modifier_apply(modifier=modifier.name)
    n_faces_after = len(obj.data.polygons)
    print(f"Before Decimating {n_faces_before} faces. After Decimating {n_faces_after}")
    bpy.ops.export_scene.obj(
        filepath=dst_path,
        filter_glob="*.obj",
        use_edges=False,
        use_normals=False,
        use_uvs=False,
        use_materials=False,
        use_selection=True
    )
    
def process_shapenet(shapenet_dir="shapenet/raw/ShapeNetCore.v2"):
    # obj_paths = glob.glob(os.path.join(DATA_DIR, shapenet_dir, "**/*.obj"), recursive=True)
    # random.shuffle(obj_paths)
    # obj_paths = obj_paths[:10]
    # new_paths = []
    # print("Pre cleaning")        
    # with Pool(22) as p:
    #     new_paths = list(tqdm(p.imap(clean_an_obj_file, obj_paths), total=len(obj_paths)))

    new_paths = glob.glob(os.path.join(DATA_DIR, "shapenet", "preclean" "/**/*.obj"), recursive=True)
    new_paths.sort()
    
    # 52472
    print("decimating")
    for obj_path in tqdm(new_paths[40000:]):
        decimate(obj_path, obj_path.replace("preclean", "decimated"), angle=10)
        
    # print("Post cleaning")
    # for obj_path in tqdm(new_paths):
    #     # print(f"Post cleaning {obj_path}")
    #     clean_an_obj_file(obj_path, obj_path.replace("decimated", "processed"))

def cache_h5py():
    obj_paths = glob.glob(os.path.join(DATA_DIR, "shapenet", "decimated" "/**/*.obj"), recursive=True)
    with h5py.File(os.path.join(DATA_DIR, "shapenet", "data2.h5"), 'w') as file:
        for obj_path in tqdm(obj_paths):
            vertices, faces = read_obj(obj_path)
            # if len(vertices) > 800 or len(faces) > 2800:
            #     continue
            head, tail = os.path.split(obj_path)
            cat = head.split('/')[-1]
            name = tail.split('.')[0]
            grp = file.create_group(f'/{cat}/{name}')

            print("# of vertices", len(vertices))
            print("# of faces", len(faces))
            # vertices: List[Tuple(str, str, str)]
            nodes = np.array([list(map(float, vertex)) for vertex in vertices])
            # faces: List[str]
            # tanspose for possible spatial locality
            # hyperedges = np.zeros((len(faces), nodes.shape[0]), dtype=int)
            hyperedges = []
            for n_face, face in enumerate(faces):
                face = map(int, face)
                for node in face:
                    hyperedges.append([n_face, node-1])
            # hyperedges = np.transpose(hyperedges)
            hyperedges = np.array(hyperedges)
            grp.create_dataset('node', data=nodes)
            grp.create_dataset('hyperedge', data=hyperedges)


if __name__ == "__main__":
    # clean_an_obj_file("/Users/qiuweikang/Downloads/table.obj", "/Users/qiuweikang/Downloads/table1.obj")
    # decimate("/Users/qiuweikang/Downloads/table1.obj", "/Users/qiuweikang/Downloads/table2.obj")
    # process_shapenet()
    cache_h5py()
    pass
