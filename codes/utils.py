import shutil
import pydicom, pickle
import numpy as np
import pandas as pd 
from tqdm import tqdm
import glob, os, cv2
from ensemble_boxes import *
from skimage.color import gray2rgb
from skimage.io import imread, imsave
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

def makedir(path):
	if not os.path.exists(path):
		os.makedirs(path)
		print(path+' created')

def visualize_class_dist(data_df):
	counts = dict()
	for _, row in tqdm(data_df.iterrows()):
		if row['class_id'] not in counts.keys():
			counts[row['class_id']]=1
		else:
			counts[row['class_id']]+=1
	print(counts.keys())
	for k in sorted(counts):
		writer.add_scalar('Num_ex_per_class/class', counts[k], k)
	writer.flush()
	writer.close()

def do_box_fusion(boxes, class_ids, classes, scores, weights, iou_thresh):
	boxes, scores, class_ids = nms(boxes, scores, class_ids, weights=weights, iou_thr = iou_thresh)
	return boxes, scores, class_ids

def class_id_name_mapping(data_df):
	map_dict = dict()
	grouped_df = data_df.groupby('class_id')
	for class_id, df in grouped_df:
		map_dict[class_id] = df['class_name'].iloc[0]
	return map_dict

def prepare_and_visualize_data(data_dir_, save_path_dest,  save_path_bboxes, visualize=False, df_file=None, class_map_dict=None):
	if df_file is None:
		data_df=None
	else:
		data_df = pd.read_csv(df_file)
	image_dicoms = sorted(glob.glob(os.path.join(data_dir_,'*')))
	bboxes_info = {}
	for dicom in tqdm(image_dicoms):
		dicom_data = pydicom.dcmread(dicom)
		image=dicom_data.pixel_array
		slope = dicom_data.RescaleSlope if "RescaleSlope" in dicom_data else 1.0
		intercept = dicom_data.RescaleIntercept if "RescaleIntercept" in dicom_data else 0.0
		if slope!=1.0:
			print(slope)
			image = (image*slope + intercept).astype(np.uint16)
		image = gray2rgb(image)
		h,w,_ = image.shape
		image -= image.min()
		image = image / image.max()
		image *= 255
		if data_df is not None:
			boxes, classes, class_ids = np.array(data_df.loc[data_df['image_id']==dicom.split('/')[-1].split('.')[0]][['x_min','y_min','x_max','y_max']].values).astype(np.uint16), data_df[data_df['image_id']==dicom.split('/')[-1].split('.')[0]]['class_name'].values, np.array(data_df[data_df['image_id']==dicom.split('/')[-1].split('.')[0]]['class_id'].values)
		else:
			boxes, classes, class_ids = [],[],[]

		boxes_int = boxes.copy()
		image_thresh = image.copy()
		if visualize:
			for box, class_ in zip(boxes_int, classes):	
				image = cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]),(255,0,0), 2)
				image = cv2.putText(image, class_, (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)			

			imsave(os.path.join(save_path, dicom.split('/')[-1]+'.jpg'), image)				

		#if visualize:
		if np.all(boxes==0) or len(boxes)==0:
				bboxes_info[dicom.split('/')[-1]+'.jpg']={'bboxes':boxes, 'class_id':[14]}
				imsave(os.path.join(save_path_dest, dicom.split('/')[-1]+'.jpg'), image_thresh)	
				continue


		scores = []
		weights = []
		weights_fusion = []
		scores_inner = []
		class_ids_fusion = []
		boxes_fusion = []

		for _ in range(len(boxes)):
			scores_inner.append(1)

		

		scores.append(scores_inner)
		class_ids_fusion.append(class_ids.tolist())
		
		boxes=boxes/(w,h,w,h)
		boxes_fusion.append(boxes.tolist())
		weights_fusion.append(1)
		
		boxes_fused, scores_fused, class_ids_fused = do_box_fusion(boxes_fusion, class_ids_fusion, classes, scores, weights_fusion, iou_thresh)
		boxes_fused = (np.array(boxes_fused)*(w,h,w,h)).astype(np.uint16)
		if visualize:
			for box, class_ in zip(boxes_fused, class_ids_fused):
				image_thresh = cv2.rectangle(image_thresh, (box[0], box[1]), (box[2], box[3]),(255,0,0), 2)
				image_thresh = cv2.putText(image_thresh, class_map_dict[class_], (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)		
		imsave(os.path.join(save_path_dest, dicom.split('/')[-1]+'.jpg'), image_thresh)	
		bboxes_info[dicom.split('/')[-1]+'.jpg'] = {'bboxes':boxes_fused, 'class_id':class_ids_fused}
		
	pickle.dump(bboxes_info, open(save_path_bboxes,'wb'))
			





def visualize(all_data_dir, bboxes_pickle):
	images = glob.glob(os.path.join(all_data_dir,'*'))
	bboxes = pickle.load(open(save_path_bboxes, 'rb'))
	#id_to_class = {v:k for k, v in class_map_dict.items()}
	for image in tqdm(images):
		img = imread(image,plugin='imageio')
		boxes = bboxes[image.split('/')[-1]]['bboxes']
		class_ids = bboxes[image.split('/')[-1]]['class_id']
		#print(boxes,' ',class_ids)
		for box, class_id in zip(boxes, class_ids):
			img = cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]),(255,0,0), 2)
			img = cv2.putText(img, class_map_dict[class_id], (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)		
		imsave(os.path.join('visualize',image.split('/')[-1]), img)

def split_data(data_dir, train_path, test_path, split_percent=0.2):
	print('Preparing train-test data.....')
	images = sorted(glob.glob(os.path.join(data_dir,'*')))
	test_images = images[:int(0.2*len(images))]
	train_images = [image for image in images if image not in test_images]
	
	makedir(train_path)
	makedir(test_path)

	print('Copying train images....')
	for img in tqdm(train_images):
		shutil.copy(img, train_path)

	print('Copying test images....')		
	for img in tqdm(test_images):
		shutil.copy(img, test_path)

	print('Train-test data prepared!')
	
	
	

if __name__=='__main__':
	iou_thresh = 0.4
	data_df = pd.read_csv('../data/train.csv')
	class_map_dict = class_id_name_mapping(data_df)
	save_path = '../data/train'
	save_path_dest = '../data/dummy_data/'
	save_path_bboxes = '../data/bboxes.pkl'
	#makedir(save_path_dest)
	#prepare_and_visualize_data(data_df, save_path, class_map_dict)
	#visualize(save_path_dest, save_path_bboxes)
	visualize_class_dist(data_df)