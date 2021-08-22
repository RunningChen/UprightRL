import numpy as np
import bpy
import bmesh

class PointCloudMaker():
	def __init__(self):
		self.active_object = bpy.context.active_object
		bpy.app.debug_value = 0
		self.instancers = []
	

	# Utility function to generate numpy points from ascii point cloud files
	def generate_points_from_pts(self, filename):
		points = np.loadtxt(filename)
		return points

	# Convert points to spheres
	def convert_to_spheres(self, points=None, object_name=None, color='TransparentGray', sphere_radius=0.01):
		# start_time = time.time()

		bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=4, radius=sphere_radius)
		template_sphere = bpy.context.active_object

		bpy.ops.mesh.primitive_plane_add()
		instancer = bpy.context.active_object
		instancer_mesh = instancer.data
		bm = bmesh.new()
		for i in range(points.shape[0]):
			bm.verts.new().co = points[i, :]
		bm.to_mesh(instancer_mesh)
		template_sphere.parent = instancer
		instancer.instance_type = 'VERTS'
		# print(bpy.data.materials.keys())  #['AxisBlue', 'AxisGreen', 'AxisRed', 'Black', 'Blue', 'Gray', 'Orange', 'TransparentGray']
		template_sphere.active_material = bpy.data.materials[color]

		# If want to use diffuse_color,should write as below, color should be those materials.key
		# template_sphere.active_material = bpy.data.materials[color]
		# template_sphere.active_material.diffuse_color=(0.58,0.8,0.8,0.8) 

		self.instancers.append(instancer)

		# print('sphere convert time: ', time.time() - start_time)

	# Clear generated instancers (point spheres)
	def clear_instancers(self):
		active_object = bpy.context.active_object
		active_object_name = active_object.name
		active_object.select_set(False)

		for instancer in self.instancers:
			instancer.select_set(True)
			for child in instancer.children:
				child.select_set(True)

		bpy.ops.object.delete()  # Delete selected objects
		active_object = bpy.context.scene.objects[active_object_name]
		active_object.select_set(True)

	# Reselect active object
	def post_process(self):
		view_layer = bpy.context.view_layer
		bpy.ops.object.select_all(action='DESELECT')
		self.active_object.select_set(True)
		view_layer.objects.active = self.active_object