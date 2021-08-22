import numpy as np
import warnings
import os
from torch.utils.data import Dataset
import trans
import h5py
warnings.filterwarnings('ignore')

def pc_normalize(pc):
	pmax = np.max(pc, axis=0)
	pmin = np.min(pc, axis=0)
	centroid = (pmax + pmin) / 2.0
	return - centroid

def random_matrix():
    random_matrix_10 = []
    for i in range(10):
        # np.random.seed()
        angle1 = np.random.uniform(-np.pi/2, np.pi/2)
        theta1 = np.random.uniform(0, np.pi * 2)
        phi1 = np.random.uniform(0, np.pi / 2)
        x1 = np.cos(theta1) * np.sin(phi1)
        y1 = np.sin(theta1) * np.sin(phi1)
        z1 = np.cos(phi1)
        axis1 = np.array([x1, y1, z1])
        quater1 = trans.axisangle2quaternion(axis=axis1, angle=angle1)
        matrix1_r = trans.quaternion2matrix(quater1)
        random_matrix_10.append(matrix1_r)
    
    return random_matrix_10

split_list = {'02691156':400,'02933112':150,'02958343':300,'03001627':600,'03636649':200,'04256520':300,'04379243':250,'04530566':150}

class DataLoader_Transform_Single_Scan(Dataset):
	def __init__(self, root, npoint=2048, split='train', category=['02691156'], assist_input=False):
		self.npoints = npoint
		self.split = split
		self.assist_input = assist_input

		if split != 'real':
			in_pts1 = np.zeros(shape=(0, 2048, 3))
			gt_up = np.zeros(shape=(0, 1, 3))
			pts_name = np.zeros(shape=(0))
			
			for cate in category:
				with h5py.File(os.path.join(root, cate+'.h5'), 'r') as f:
					print(f['gt'].shape)
					
					test_eind = split_list[cate]
					val_eind = split_list[cate]+split_list[cate]

					if split == 'test':
						in_pts1 = np.concatenate((in_pts1, np.array(f['gt'])[:test_eind,:,:]), axis=0).astype(np.float32)
						gt_up = np.concatenate((gt_up, np.array(f['gt_up'])[:test_eind,:,:]), axis=0).astype(np.float32)
						pts_name = np.concatenate((pts_name, np.array(f['name'][:test_eind])), axis=0)
					elif split == 'val':
						in_pts1 = np.concatenate((in_pts1, np.array(f['gt'])[test_eind:val_eind,:,:]), axis=0).astype(np.float32)
						gt_up = np.concatenate((gt_up, np.array(f['gt_up'])[test_eind:val_eind,:,:]), axis=0).astype(np.float32)
						pts_name = np.concatenate((pts_name, np.array(f['name'][test_eind:val_eind])), axis=0)
					else:
						# in_pts1 = np.concatenate((in_pts1, np.array(f['gt'])[val_eind:val_eind+1000,:,:]), axis=0).astype(np.float32)
						# gt_up = np.concatenate((gt_up, np.array(f['gt_up'])[val_eind:val_eind+1000,:,:]), axis=0).astype(np.float32)
						# pts_name = np.concatenate((pts_name, np.array(f['name'][val_eind:val_eind+1000])), axis=0)

						in_pts1 = np.concatenate((in_pts1, np.array(f['gt'])[val_eind:,:,:]), axis=0).astype(np.float32)
						gt_up = np.concatenate((gt_up, np.array(f['gt_up'])[val_eind:,:,:]), axis=0).astype(np.float32)
						pts_name = np.concatenate((pts_name, np.array(f['name'][val_eind:])), axis=0)

				print(os.path.join(root, split+'_'+cate+'.h5'), ' LOADED!')

			self.in_ptss1 = np.array(in_pts1)
			self.ptss_name = np.array(pts_name)
			self.in_ptss1_up = np.array(gt_up)

			print('The size of %s data is %d'%(split, len(self.in_ptss1)))
		else:
			with h5py.File(os.path.join(root, 'real', category[0]+'real.h5'), 'r') as f:
				in_pts = np.array(f['in'])
				gt_pts = np.array(f['gt'])
				pts_name = np.array(f['name'])
			print(os.path.join(root, 'real', category[0]+'real.h5'), ' LOADED!')
			in_pts[:,:,[1,2]] = in_pts[:,:,[2,1]]
			gt_pts[:,:,[1,2]] = gt_pts[:,:,[2,1]]
			self.in_ptss = np.array(in_pts)
			self.gt_ptss = np.array(gt_pts)
			self.ptss_name = np.array(pts_name)
			print('The size of %s data is %d'%(split, len(self.ptss_name)))

	def __len__(self):
		return len(self.in_ptss1)

	def __getitem__(self, index):
		in_pts1 = self.in_ptss1[index][:int(self.npoints)]
		gt_up = self.in_ptss1_up[index]

		return in_pts1,gt_up[0]  #becase gt_upright's shape is [1,3],so we should add [0]

	def get_name(self, index):
		# get corresponding pointcloud names from index
		return self.ptss_name[index]#.decode('utf-8')

class DataLoader_Transform(Dataset):
	def __init__(self, root, npoint=2048, split='train', isrotate=True, rotate_azimuth=False, category=['02691156'], assist_input=False):
		self.npoints = npoint
		self.split = split
		self.isrotate = isrotate
		self.rotate_azimuth = rotate_azimuth
		self.assist_input = assist_input

		if split != 'real':
			in_pts1 = np.zeros(shape=(0, 2048, 3))

			pts_name = np.zeros(shape=(0))

			for cate in category:
				with h5py.File(os.path.join(root, cate+'.h5'), 'r') as f:
					print(f['gt'].shape)

					test_eind = split_list[cate]
					val_eind = split_list[cate]+split_list[cate]
					
					if split == 'test':
						in_pts1 = np.concatenate((in_pts1, np.array(f['gt'])[:test_eind,:,:]), axis=0).astype(np.float32)
						pts_name = np.concatenate((pts_name, np.array(f['name'][:test_eind])), axis=0)
					elif split == 'val':
						in_pts1 = np.concatenate((in_pts1, np.array(f['gt'])[test_eind:val_eind,:,:]), axis=0).astype(np.float32)
						pts_name = np.concatenate((pts_name, np.array(f['name'][test_eind:val_eind])), axis=0)
					else:
						in_pts1 = np.concatenate((in_pts1, np.array(f['gt'])[val_eind:,:,:]), axis=0).astype(np.float32)
						pts_name = np.concatenate((pts_name, np.array(f['name'][val_eind:])), axis=0)
					
				print(os.path.join(root, split+'_'+cate+'.h5'), ' LOADED!')

			in_pts1[:,:,[1,2]] = in_pts1[:,:,[2,1]]

			self.in_ptss1 = np.array(in_pts1)
			self.ptss_name = np.array(pts_name)

			print('The size of %s data is %d'%(split, len(self.in_ptss1)))
		else:
			with h5py.File(os.path.join(root, 'real', category[0]+'real.h5'), 'r') as f:
				in_pts = np.array(f['in'])
				gt_pts = np.array(f['gt'])
				pts_name = np.array(f['name'])
			print(os.path.join(root, 'real', category[0]+'real.h5'), ' LOADED!')
			in_pts[:,:,[1,2]] = in_pts[:,:,[2,1]]
			gt_pts[:,:,[1,2]] = gt_pts[:,:,[2,1]]
			self.in_ptss = np.array(in_pts)
			self.gt_ptss = np.array(gt_pts)
			self.ptss_name = np.array(pts_name)
			print('The size of %s data is %d'%(split, len(self.ptss_name)))

	def __len__(self):
		return len(self.in_ptss1)

	def __getitem__(self, index):
		in_pts1 = self.in_ptss1[index][:int(self.npoints)]
		

		# 计算旋转
		if self.isrotate:

			angle1 = np.random.uniform(-np.pi/2, np.pi/2)
			
			theta1 = np.random.uniform(0, np.pi * 2)
			phi1 = np.random.uniform(0, np.pi / 2)
			x1 = np.cos(theta1) * np.sin(phi1)
			y1 = np.sin(theta1) * np.sin(phi1)
			z1 = np.cos(phi1)
			axis1 = np.array([x1, y1, z1])
		else:
			angle1 = 0
			axis1 = np.array([0.0,0.0,1.0])

		quater1 = trans.axisangle2quaternion(axis=axis1, angle=angle1)
		matrix1_r = trans.quaternion2matrix(quater1)

		origin = np.array([0.0,0.0,1.0])
		trans1 = pc_normalize(in_pts1)
		matrix1_t = trans.translation2matrix(trans1)

		in_pts1 = trans.transform_pts(in_pts1, matrix1_t)
		in_pts1 = trans.transform_pts(in_pts1, matrix1_r)
		gt_upright = trans.transform_pts(origin,matrix1_r)

		return in_pts1,gt_upright[0]

	def get_name(self, index):
		# get corresponding pointcloud names from index
		return self.ptss_name[index]#.decode('utf-8')

class DataLoader_Transform_Upright2(Dataset):
	def __init__(self, root, npoint=2048, split='train', isrotate=True, rotate_azimuth=False, category=['table'], assist_input=False):
		self.npoints = npoint
		self.split = split
		self.isrotate = isrotate
		self.rotate_azimuth = rotate_azimuth
		self.assist_input = assist_input

		if split != 'real':
			in_pts1 = np.zeros(shape=(0, 2048, 3))
			pts_name = np.zeros(shape=(0))

			for cate in category:
				
				with h5py.File(os.path.join(root, split+'_'+cate+'.h5'), 'r') as f:

					print(f['gt'].shape)
					# print(f.keys())
					# key_list = sorted(f.keys())
					# for i in key_list:
					# 	print(f[i].shape)
					
					in_pts1 = np.concatenate((in_pts1, np.array(f['gt'])[:]), axis=0).astype(np.float32)
					pts_name = np.concatenate((pts_name, np.array(f['name'][:])), axis=0)
					
				print(os.path.join(root, split+'_'+cate+'.h5'), ' LOADED!')
			in_pts1[:,:,[1,2]] = in_pts1[:,:,[2,1]]
			# print(in_pts1.shape)
			self.in_ptss1 = np.array(in_pts1)
			self.ptss_name = np.array(pts_name)

			print('The size of %s data is %d'%(split, len(self.in_ptss1)))
		else:
			with h5py.File(os.path.join(root, 'real', category[0]+'real.h5'), 'r') as f:
				in_pts = np.array(f['in'])
				gt_pts = np.array(f['gt'])
				pts_name = np.array(f['name'])
			print(os.path.join(root, 'real', category[0]+'real.h5'), ' LOADED!')
			in_pts[:,:,[1,2]] = in_pts[:,:,[2,1]]
			gt_pts[:,:,[1,2]] = gt_pts[:,:,[2,1]]
			self.in_ptss = np.array(in_pts)
			self.gt_ptss = np.array(gt_pts)
			self.ptss_name = np.array(pts_name)
			print('The size of %s data is %d'%(split, len(self.ptss_name)))

	def __len__(self):
		return len(self.in_ptss1)

	def __getitem__(self, index):
		in_pts1 = self.in_ptss1[index][:int(self.npoints)]
		# np.savetxt('../output/completion/test/inpts_gt'+str(index)+'.txt',in_pts1)	
		# gt = in_pts1
		# np.random.seed(0)
		# 计算旋转

		angle1 = np.random.uniform(-np.pi/2, np.pi/2)
		
		theta1 = np.random.uniform(0, np.pi * 2)
		phi1 = np.random.uniform(0, np.pi / 2)
		x1 = np.cos(theta1) * np.sin(phi1)
		y1 = np.sin(theta1) * np.sin(phi1)
		z1 = np.cos(phi1)
		axis1 = np.array([x1, y1, z1])

		quater1 = trans.axisangle2quaternion(axis=axis1, angle=angle1)
		matrix1_r = trans.quaternion2matrix(quater1)

		origin = np.array([0.0, 0.0, 1.0])
		trans1 = pc_normalize(in_pts1)
		matrix1_t = trans.translation2matrix(trans1)

		in_pts1 = trans.transform_pts(in_pts1, matrix1_t)
		in_pts1 = trans.transform_pts(in_pts1, matrix1_r)

		gt_upright = trans.transform_pts(origin,matrix1_r)

		return in_pts1,gt_upright[0]

	def get_name(self, index):
		# get corresponding pointcloud names from index
		return self.ptss_name[index]#.decode('utf-8')

class DataLoader_Transform_Upright(Dataset):
    def __init__(self, root, npoint=2048, split='train', isrotate=True, rotate_azimuth=False, category=['02691156'], small_set=False, assist_input=False,test_aug = False):
        self.npoints = npoint
        self.split = split
        self.isrotate = isrotate
        self.rotate_azimuth = rotate_azimuth
        self.assist_input = assist_input
        self.test_aug = test_aug

        if split != 'real':
            in_pts1 = np.zeros(shape=(0, 2048, 3))
            pts_name = np.zeros(shape=(0))

            for cate in category:
                with h5py.File(os.path.join(root, split+'_'+cate+'.h5'), 'r') as f:
                    print(f['gt'].shape)

                    in_pts1 = np.concatenate((in_pts1, np.array(f['gt'])[:]), axis=0).astype(np.float32)
                    pts_name = np.concatenate((pts_name, np.array(f['name'][:])), axis=0)

                print(os.path.join(root, split+'_'+cate+'.h5'), ' LOADED!')

            self.in_ptss1 = np.array(in_pts1)
            self.ptss_name = np.array(pts_name)

            print('The size of %s data is %d' % (split, len(self.in_ptss1)))
        else:
            with h5py.File(os.path.join(root, 'real', category[0]+'real.h5'), 'r') as f:
                in_pts = np.array(f['in'])
                gt_pts = np.array(f['gt'])
                pts_name = np.array(f['name'])
            print(os.path.join(root, 'real',
                               category[0]+'real.h5'), ' LOADED!')
            in_pts[:, :, [1, 2]] = in_pts[:, :, [2, 1]]
            gt_pts[:, :, [1, 2]] = gt_pts[:, :, [2, 1]]
            self.in_ptss = np.array(in_pts)
            self.gt_ptss = np.array(gt_pts)
            self.ptss_name = np.array(pts_name)
            print('The size of %s data is %d' % (split, len(self.ptss_name)))

    def __len__(self):
        return len(self.in_ptss1)

    def __getitem__(self, index):
        in_pts1 = self.in_ptss1[index][:int(self.npoints)]
        
        angle1 = np.random.uniform(-np.pi/2, np.pi/2)
		
        theta1 = np.random.uniform(0, np.pi * 2)
        phi1 = np.random.uniform(0, np.pi / 2)
        x1 = np.cos(theta1) * np.sin(phi1)
        y1 = np.sin(theta1) * np.sin(phi1)
        z1 = np.cos(phi1)
        axis1 = np.array([x1, y1, z1])
        quater1 = trans.axisangle2quaternion(axis=axis1, angle=angle1)
        matrix1_r = trans.quaternion2matrix(quater1)

        origin = np.array([0.0, 0.0, 1.0])
        trans1 = pc_normalize(in_pts1)
        matrix1_t = trans.translation2matrix(trans1)
        in_pts1 = trans.transform_pts(in_pts1, matrix1_t)
        in_pts1 = trans.transform_pts(in_pts1, matrix1_r)
	
        gt_upright = trans.transform_pts(origin, matrix1_r)
        return in_pts1, gt_upright[0]

    def get_name(self, index):
        # get corresponding pointcloud names from index
        return self.ptss_name[index]  # .decode('utf-8')

class DataLoader_Transform_TTA(Dataset):
	def __init__(self, root, npoint=2048, split='train',  category=['02691156'], assist_input=False):
		self.npoints = npoint
		self.split = split 
		self.assist_input = assist_input

		if split != 'real':
			in_pts1 = np.zeros(shape=(0, 2048, 3))
			gt_up = np.zeros(shape=(0, 3))
			pts_name = np.zeros(shape=(0))
			
			for cate in category:
				with h5py.File(os.path.join(root, split+'_'+cate+'.h5'), 'r') as f:
					print(f['gt'].shape)
	
					in_pts1 = np.concatenate((in_pts1, np.array(f['gt'])[:]), axis=0).astype(np.float32)
					pts_name = np.concatenate((pts_name, np.array(f['name'][:])), axis=0)
					gt_up = np.concatenate((gt_up, np.array(f['up'])[:]), axis=0).astype(np.float32)

			
				print(os.path.join(root, cate+'.h5'), ' LOADED!')

			self.in_ptss1 = np.array(in_pts1)
			self.ptss_name = np.array(pts_name)
			self.in_ptss1_up = np.array(gt_up)

			print('The size of %s data is %d'%(split, len(self.in_ptss1)))
		else:
			with h5py.File(os.path.join(root, 'real', category[0]+'real.h5'), 'r') as f:
				in_pts = np.array(f['in'])
				gt_pts = np.array(f['gt'])
				pts_name = np.array(f['name'])
			print(os.path.join(root, 'real', category[0]+'real.h5'), ' LOADED!')
			in_pts[:,:,[1,2]] = in_pts[:,:,[2,1]]
			gt_pts[:,:,[1,2]] = gt_pts[:,:,[2,1]]
			self.in_ptss = np.array(in_pts)
			self.gt_ptss = np.array(gt_pts)
			self.ptss_name = np.array(pts_name)
			print('The size of %s data is %d'%(split, len(self.ptss_name)))

	def __len__(self):
		return len(self.in_ptss1)

	def __getitem__(self, index):
		in_pts1 = self.in_ptss1[index][:int(self.npoints)]
		gt_up = self.in_ptss1_up[index]
		
		return in_pts1,gt_up  #becase gt_upright's shape is [1,3],so we should add [0]

	def get_name(self, index):
		# get corresponding pointcloud names from index
		return self.ptss_name[index]#.decode('utf-8')


if __name__ == '__main__':
	import torch
	#  /home/luanmin/projects/uprightRL/UprightRL/data/shapenet_virtualscan/
	data = DataLoader_Transform(root='/mnt/disk2/Upright_output/uprightRL_output/output/origin_h5/', npoint=2048, split='train', isrotate=True)
	DataLoader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=False)
	i = 0

	gt,in_pts1, gt_para_r1,gt_re = data.__getitem__(1)
	np.savetxt('../output/completion/test/1.txt',gt)
	np.savetxt('../output/completion/test/2.txt',gt_re)
	print(int(data.get_name(1)))