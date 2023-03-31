import torch
import torchvision
from torchvision.transforms.functional import resize
import bpy
import os
import sys
from os import dup, close
from contextlib import ContextDecorator
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import seaborn as sns

from tango.common import Registrable

from .utils import VerticesMutedError


class Visualizer(Registrable):
    ...


class stdout_suppress(ContextDecorator):
    def __init__(self) -> None:
        super().__init__()
    
    def __enter__(self):
        logfile = "/dev/null"
        open(logfile, 'a').close()
        self.old = os.dup(sys.stdout.fileno())
        sys.stdout.flush()
        os.close(sys.stdout.fileno())
        self.fd = os.open(logfile, os.O_WRONLY)
        
    def __exit__(self, exc_type, exc, exc_tb):
        # disable output redirection
        os.close(self.fd)
        os.dup(self.old)
        os.close(self.old)
            

@Visualizer.register("blender") 
class BlenderRenderer(Visualizer):
    def __init__(self, engine = 'BLENDER_EEVEE'):
        bpy.ops.wm.read_homefile()
        self.resolution_x = 256
        self.resolution_y = 256
        bpy.ops.object.delete(use_global=False)
        cam = bpy.data.objects["Camera"]
        bpy.context.view_layer.objects.active = cam
        bpy.context.object.location = bpy.context.object.location * 0.15
        bpy.context.scene.render.resolution_x = self.resolution_x
        bpy.context.scene.render.resolution_y = self.resolution_y
        bpy.context.scene.render.engine = engine
        
        bpy.ops.object.select_pattern(pattern="Light")
        bpy.ops.object.delete(use_global=False)
        
        # Rembrandt lighting
        self.lighting_setup("Key", (0.07474, -3.67952, 3.83413), 800)
        self.lighting_setup("Fill", (3.20319, 0.459794, 3.13854), 400)
        self.lighting_setup("Rim", (-2.18065, 1.44968, -0.096274), 50)
    
    def lighting_setup(self, name, location, energy):
        light_data = bpy.data.lights.new(name=name.lower() + '-data', type='POINT')
        light_data.energy = energy
        # Create new object, pass the light data 
        light_object = bpy.data.objects.new(name=name, object_data=light_data)
        # Link object to collection in context
        bpy.context.collection.objects.link(light_object)
        # Change light position
        light_object.location = location
        
    @staticmethod
    def make_face(obj, face):
        bpy.ops.mesh.select_all(action='DESELECT')
        bpy.ops.object.mode_set(mode='OBJECT')
        for vid in face:
            try:
                obj.data.vertices[vid].select = True
            except IndexError:
                raise VerticesMutedError

        bpy.ops.object.mode_set(mode='EDIT') 
        bpy.ops.mesh.edge_face_add()

    @stdout_suppress()
    def visualize_object(self, x, h, m, idx=None):
        in_mem_dir = "/dev/shm" # store file in memory so it would be faster and still compatible to API
        temp_dir = "hypergen_render"
        dir_path = os.path.join(in_mem_dir, temp_dir)
        dir_path = "./results/wavefront"
        obj_path = os.path.join(dir_path, f"{idx}.obj")

        h = h * m[None, :]
        x = x[m.bool()]        
        
        # This face's index is form 0 istead of 1!
        h = torch.unique(h, dim=0, sorted=False)
        faces = []
        for face in h:
            face = face.nonzero(as_tuple=True)[0].tolist()
            if len(face) > 0:
                faces.append(face)
        
        with open(obj_path, 'w') as file:
            for vert in x.tolist():
                file.write(f"v {vert[0]} {vert[1]} {vert[2]}\n")
            for face in faces:
                file.write(f"#f " + " ".join(map(lambda f: str(f+1), face)) + '\n')

        bpy.ops.import_scene.obj(filepath=obj_path, filter_glob="*.obj", split_mode="OFF")
        selected_objects = bpy.context.selected_objects
        assert len(selected_objects) == 1
        obj = selected_objects[0]
        bpy.context.view_layer.objects.active = obj
        
        # Fill Face
        bpy.ops.object.mode_set(mode='EDIT') 
        bpy.ops.mesh.select_mode(type="VERT")
        for face in faces:
            BlenderRenderer.make_face(obj, face)
        bpy.ops.mesh.select_mode(use_extend=False, use_expand=False, type='FACE')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.normals_make_consistent(inside=False)
        bpy.ops.object.mode_set(mode='OBJECT') 
    
        ind = os.path.split(obj_path)[-1].split('.')[0]
        
        outpath = f"results/render/{ind}.png"
        bpy.context.scene.render.filepath = outpath
        bpy.ops.render.render(write_still=True)  # save straight to file
        self.save_file(f"results/blender/{ind}.blend")
        
        bpy.ops.object.delete()
        if len(bpy.data.objects) > 4:
            for obj in bpy.data.objects:
                if obj.name not in ["Camera", "Key", "Fill", "Rim"]:
                    obj.select_set(True)
                    bpy.ops.object.delete()

        image = torchvision.io.read_image(outpath)
            
        return image
    
    @stdout_suppress()
    def save_file(self, path="results/blender/blender.blend"):
        bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(path))
        

@Visualizer.register("matplotlib")
class MatplotlibPlotter(Visualizer):
    def __init__(self) -> None:
        super().__init__()
        
    def visualize_object(self, x, h, idx=None):
        """
        x - [n_nodes, 3]
        h - [n_hyper, n_nodes]
        """
        device = x.device
        x = x.cpu().numpy()
        h = torch.unique(h, dim=0, sorted=False).bool().cpu().numpy()
        collection = []
        for face in h:
            collection.append(x[face])
        
        fig = plt.figure(figsize=(3, 3), dpi=96)
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111, projection='3d')
        
        plt_mesh = Poly3DCollection(collection)
        plt_mesh.set_edgecolor((0., 0., 0., 0.0))
        plt_mesh.set_facecolor((1, 0, 0, 0.0))
        ax.add_collection3d(plt_mesh)

        ax.scatter3D(
            x[:, 0],
            x[:, 1],
            x[:, 2],
            lw=0.,
            s=10,
            c=['b', 'g', 'r', 'c', 'm', 'y', 'k', 'darkorange'],
            alpha=0.75)
        
        ax_lims = 0.5
        ax.set_xlim(-ax_lims, ax_lims)
        ax.set_ylim(-ax_lims, ax_lims)
        ax.set_zlim(-ax_lims, ax_lims)

        ax.view_init(30, 120)
        
        canvas.draw()
        width, height = fig.get_size_inches() * fig.get_dpi()
        width = width.astype(int)
        height = height.astype(int)
        image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(height, width, 3) 
        image = torch.from_numpy(image)
        image = image.to(device).permute(2, 0, 1)
        image = resize(image, (256, 256))
        
        plt.close()
        
        return image
    
@Visualizer.register("heatmap")
class HeatmapVisualizer(Visualizer):
    def __init__(self):
        super().__init__()
        
    def visualize_object(self, x, h, idx=None):
        device = h.device
        fig = plt.figure(figsize=(3, 3), dpi=96)
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        h = h.float().detach().cpu().numpy()
        sns.heatmap(h, vmin=0., vmax=1., ax=ax)
        
        canvas.draw()
        width, height = fig.get_size_inches() * fig.get_dpi()
        width = width.astype(int)
        height = height.astype(int)
        image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(height, width, 3) 
        image = torch.from_numpy(image)
        image = image.to(device).permute(2, 0, 1)
        image = resize(image, (256, 256))
        
        plt.close()
        return image
        