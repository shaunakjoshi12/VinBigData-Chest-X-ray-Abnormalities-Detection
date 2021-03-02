import os, pickle
from PIL import Image
import numpy as np
import pandas as pd
from tqdm import tqdm
from utils import makedir
import glob, torch, cv2
from augment import train_transforms, test_transforms, train_transforms_only_image, test_transforms_only_image 
from utils import split_data, class_id_name_mapping
from skimage.io import imread, imsave
from torch.utils.data import Dataset, DataLoader


def checkBadFiles(image_paths):
	bad_files=[]
	for image in tqdm(image_paths):
		try:
			img = Image.open(image)
			img.verify()
		except:
			bad_files.append(image)
	print('Bad files ',len(bad_files))

def collate(batch):
	
	images = [d['img'] for d in batch]
	annots = [d['annots'] for d in batch]
	max_boxes = max(box.shape[0] for box in annots)
		
	if max_boxes > 0: 
		padded_annotations = np.ones((len(batch), max_boxes, 5))*-1	
		for item,annot in enumerate(annots):
			if annot.shape[0] > 0:
				padded_annotations[item, :annot.shape[0], :]=annot

	else:
		padded_annotations = np.ones((len(batch), 1, 5))*-1

	return_dict = {'img':torch.FloatTensor(images), 'annot':torch.FloatTensor(padded_annotations)}
	# for k,v in return_dict.items():
	# 	print(k,':',v.size())
	return return_dict



class VinBigDataset(Dataset):
	def __init__(self, data_dir, pickle_file, class_id_to_name_map, save_path_viz, train=True, visualize=True):
		super(VinBigDataset, self).__init__()
		self.data_dir = data_dir
		self.pickle_file = pickle_file
		self.bboxes_info = pickle.load(open(pickle_file,'rb'))
		self.save_path_viz = None
		self.visualize = visualize
		self.train = train
		self.class_id_to_name_map = class_id_to_name_map

		if self.visualize:
			self.save_path_viz = save_path_viz
			makedir(self.save_path_viz)

		if self.train:
			self.transforms=train_transforms
			self.transforms_only_image = train_transforms_only_image
		else:
			self.transforms=test_transforms
			self.transforms_only_image = test_transforms_only_image

		self.image_paths = sorted(glob.glob(os.path.join(data_dir,'*')))
		#checkBadFiles(self.image_paths)
		


	def makedir(self, path):
		if not os.path.exists(path):
			os.makedirs(path)

	def num_classes(self):
		return len(self.class_id_to_name_map.keys())

	def __getitem__(self, index):
		image  = self.image_paths[index]
		image_key = image.split('/')[-1]
		bboxes, classes = self.bboxes_info[image_key]['bboxes'], np.array(self.bboxes_info[image_key]['class_id'])
		classes = np.expand_dims(classes, axis=1)

		img = imread(image, plugin='imageio')

		if np.all(bboxes==0):
			#print('No findings')
			transformed = self.transforms_only_image(image=img)
			img = transformed['image']				
			return {'img':img.transpose(2,0,1), 'annots':np.array([])}

		bboxes_with_classes = np.concatenate([bboxes, classes], axis=1)
		transformed = self.transforms(image=img, bboxes=bboxes_with_classes)
		bboxes_with_classes = transformed['bboxes']
		img = transformed['image']
		h,w,_ = img.shape

		if self.visualize:
			img_viz = img.copy()
			img_viz=((img_viz*(0.229, 0.224, 0.225) + (0.485, 0.456, 0.406))*255).astype(np.uint8)

			for bbox_class in bboxes_with_classes:
				
				img_viz = cv2.rectangle(img_viz, (int(bbox_class[0]), int(bbox_class[1])), (int(bbox_class[2]), int(bbox_class[3])), (255, 0, 0), 2)
				img_viz = cv2.putText(img_viz, self.class_id_to_name_map[bbox_class[-1]], (int(bbox_class[0]), int(bbox_class[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

			imsave(os.path.join(self.save_path_viz, image.split('/')[-1]), img_viz)

		
		if len(bboxes_with_classes)==0:			
			return {'img':img.transpose(2,0,1), 'annots':np.array([])}

		return {'img':img.transpose(2,0,1), 'annots':np.array(bboxes_with_classes)}


	def __len__(self):
		return len(self.image_paths)


if __name__ == '__main__':
	data_csv = '../data/train.csv'
	class_id_to_name_map = class_id_name_mapping(pd.read_csv(data_csv))
	data_dir, pickle_file, save_path_viz = '../data/all_data','../data/bboxes.pkl',  '../data/viz'
	train_dir, test_dir = '../data/train_data/','../data/test_data/'
	#train_dir, test_dir = split_data(data_dir)
	train_dataset = VinBigDataset(train_dir, pickle_file, class_id_to_name_map, save_path_viz, visualize=False)
	test_dataset = VinBigDataset(test_dir, pickle_file, class_id_to_name_map, save_path_viz, train=False)

	train_dataloader = DataLoader(train_dataset, batch_size = 2, num_workers=2, collate_fn=collate)
	for data in tqdm(train_dataloader):
		print(data['img_name'],' ',data['annot'])
		import pdb;pdb.set_trace()
		
						
