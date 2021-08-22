
import os
import sys
import time
import bpy
import h5py

ROOT_DIR = bpy.path.abspath('//')
sys.path.insert(1, ROOT_DIR)


import numpy as np
from point_cloud_maker import PointCloudMaker

# Apply transformation to points
def transform_pts(points, transform):
	# batch transform
	if len(transform.shape) == 3:
		rot = transform[:, :3, :3]
		trans = transform[:, :3, 3]
	# single transform
	else:
		rot = transform[:3, :3]
		trans = transform[:3, 3]
	point = np.matmul(points, np.transpose(rot)) + np.expand_dims(trans, axis=-2)
	return point

# Setup blender scene
def preset_scene(category):
	# 3D cursor settings
	bpy.context.scene.cursor.location = (0, 0, 0)
	bpy.context.scene.cursor.rotation_euler = (0, 0, 0)
	bpy.ops.object.select_all(action='DESELECT')
	bpy.data.objects['Origin'].select_set(True)
	bpy.context.view_layer.objects.active = bpy.data.objects['Origin']
	
	# Set up z axis length
	z_axis_length_table = {
		'02691156': 0.6,
		'02933112': 0.6,
		'02958343': 0.6,
		'03636649': 0.6,
		'03001627': 0.6,
		'04256520': 0.6,
		'04379243': 0.6,
		'04530566': 0.6,
		'base':0.6,
		'airplane': 0.6,
		'bathtub': 0.6,
		'bicycle': 0.6,
		'car': 0.6,
		'chair': 0.6,
		'cup': 0.6,
		'dog': 0.6,
		'fruit': 0.6,
		'person': 0.6,
		'table': 0.6
	}
	bpy.context.scene.objects['Z-Axis'].scale.z = z_axis_length_table[category]
	bpy.context.scene.objects['Arrow'].location.z = z_axis_length_table[category]

# Clear intermediate stuff
def reset(pcm=None, clear_instancers=False, clear_database=False):
	# start_time = time.time()

	if (clear_instancers):
		# Clear materials
		for material in bpy.data.materials:
			if ('Material' in material.name) or ('material' in material.name):
				bpy.data.materials.remove(material)

	if pcm != None:
		pcm.clear_instancers()
	if clear_database:
		# Clear meshes
		for mesh in bpy.data.meshes:
			if (not 'Cube' in mesh.name) and (not 'Cone' in mesh.name) \
					and (not 'Cylinder' in mesh.name):
				bpy.data.meshes.remove(mesh)

		# Clear images
		for image in bpy.data.images:
			bpy.data.images.remove(image)
	# print('reset time: ', time.time() - start_time)

if __name__ == "__main__":
	
	data_type = 'single' # complete / partial / single_scan / uprl ... just a folder name you like
	image_dir = os.path.join(ROOT_DIR, 'images_' + data_type)
	
	render_type = 'out'  # gt: means input points   #out means output points

	# the result h5 file
	h5_path = '/home/ubuntu/ws/upright/main/output/test/4_single_scan_l2__2021-07-07-18-08_3/test.h5'

	# the categories you want to render
	cats = ['03001627'] #,'02933112','02958343','03001627','03636649','04256520','04379243','04530566']
	# cate = '02691156'
	# cats = ['airplane','bathtub', 'car', 'chair', 'cup', 'dog', 'fruit','person','table', 'bicycle']
	# cats = ['chair']#['airplane','bathtub','bicycle','car','chair']#,['cup','dog','fruit','person','table']

	save_folder_name = "angle_4"
	render_num = 5
	start_index = 0

	# bpy.context.scene.cycles.device = 'GPU'
	for cate in cats:
		preset_scene(cate)

		with h5py.File(os.path.join( h5_path ),'r') as f:

			print(f.keys())  #['gt_pts','gt_up','out_pts',out_up']
			key_list = sorted(f.keys())
			for i in key_list:
				print(f[i].shape)
				length = (f[i].shape)[0]
				break

			for k in range(start_index, start_index + render_num):
				ii = k % 8
				i = int(k / 8)
				
				image_path = os.path.join(image_dir, cate + '_' + render_type, save_folder_name, str(i*8+ii)+'.png')
				os.makedirs( os.path.join(image_dir, cate + '_' + render_type, save_folder_name), exist_ok=True )
				
				if render_type == 'out':
					sphere_color = 'Orange'#'Orang'
				else:
					sphere_color = 'Gray'
					
				sphere_radius = 0.02

				# Generate point cloud from file
				print('Generating point cloud...')
				pcm = PointCloudMaker()
				
				pcd = f[render_type+'_pts'][i,ii,:,:]
				reset(clear_instancers=True)

				# Create spheres from point clouds
				print('Meshing...')
				pcm.convert_to_spheres(points=pcd, object_name=cate, color=sphere_color, sphere_radius=sphere_radius)
				pcm.post_process()

				reset(clear_instancers=True)

				############TO DO##########
				#picture the arrow indicate up 
				
				print('Rendering into image...')

				bpy.context.scene.render.filepath = image_path
				bpy.ops.render.render(write_still=True)

				reset(pcm, clear_instancers=True)  # Clear scene
				print("Done")
			

