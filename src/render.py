import torch
import torchvision
import bpy
import os
import sys
from os import dup, close
from contextlib import ContextDecorator

from .utils import VerticesMutedError


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
            
        
class BlenderRenderer:
    def __init__(self, engine = 'BLENDER_EEVEE'):
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
    def render_obj(self, path, faces):
        print(path)
        bpy.ops.import_scene.obj(filepath=path, filter_glob="*.obj", split_mode="OFF")
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
    
        ind = os.path.split(path)[-1].split('.')[0]
        
        outpath = f"results/render/{ind}.png"
        bpy.context.scene.render.filepath = outpath
        bpy.ops.render.render(write_still=True)  # save straight to file
        
        bpy.ops.object.delete()
        if len(bpy.data.objects) > 4:
            for obj in bpy.data.objects:
                if obj.name not in ["Camera", "Key", "Fill", "Rim"]:
                    obj.select_set(True)
                    bpy.ops.object.delete()

        image = torchvision.io.read_image(outpath)
            
        return image
    
    @stdout_suppress()
    def save_file(self):
        bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath("results/blender/blender.blend"))