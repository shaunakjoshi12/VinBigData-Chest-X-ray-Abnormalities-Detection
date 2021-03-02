import torch
import numpy as np
import pandas as pd
import time
import os
import csv
import cv2
import argparse
import csv
import glob
from skimage.io import imread, imsave
from skimage.transform import resize
from tqdm import tqdm
from utils import class_id_name_mapping, makedir, prepare_and_visualize_data


# Draws a caption above the box in an image
def draw_caption(image, box, caption):
    b = np.array(box).astype(int)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)


def detect_image(image_path, model_path, class_id_to_name_mapping, save_path):
    print('Generating submission file for {} ......'.format(model_path.split('/')[-1]))
    submission_file = 'submission_{}.csv'.format(model_path.split('/')[-1])
    csvwriter = csv.writer(open(submission_file,'w'))
    csvwriter.writerow(['image_id','PredictionString'])
    model = torch.load(model_path)

    if torch.cuda.is_available():
        model = model.cuda()

    model.training = False
    model.eval()

    for img_name in tqdm(os.listdir(image_path)):

        image = imread(os.path.join(image_path, img_name), plugin='imageio')
        if image is None:
            continue
        image_orig = image.copy()

        rows, cols, cns = image.shape

        scale_h = rows/512
        scale_w = cols/512
        # resize the image with the computed scale
        image = resize(image, (512, 512), preserve_range=True)
        rows, cols, cns = image.shape

        image = image.astype(np.float32)
        image /= 255
        image -= [0.485, 0.456, 0.406]
        image /= [0.229, 0.224, 0.225]
        image = np.expand_dims(image, 0)
        image = np.transpose(image, (0, 3, 1, 2))

        with torch.no_grad():

            image = torch.from_numpy(image)
            if torch.cuda.is_available():
                image = image.cuda()

            st = time.time()
            scores, classification, transformed_anchors = model(image.cuda().float())
            #print('Elapsed time: {}'.format(time.time() - st))
            idxs = np.where(scores.cpu() > 0.5)


            boxes_to_csv = []
            for j in range(idxs[0].shape[0]):
                bbox = transformed_anchors[idxs[0][j], :]

                x1 = int(bbox[0] * scale_w)
                y1 = int(bbox[1] * scale_h)
                x2 = int(bbox[2] * scale_w)
                y2 = int(bbox[3] * scale_h)
                class_id = int(classification[idxs[0][j]])
                label_name = class_id_to_name_mapping[class_id]
                #print(bbox, classification.shape)
                score = scores[j]
                caption = '{} {:.3f}'.format(label_name, score)
                box = [class_id, round(score.item(),1), x1,y1,x2,y2]
                boxes_to_csv+=box
                # draw_caption(img, (x1, y1, x2, y2), label_name)
                # draw_caption(image_orig, (x1, y1, x2, y2), caption)
                # cv2.rectangle(image_orig, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
                # imsave(os.path.join(save_path,img_name), image_orig)

            if len(boxes_to_csv)==0:
            	csvwriter.writerow([img_name.split('.')[0],''.join(str(elem)+' ' for elem in [14,1,0,0,1,1]).strip()])	
            	continue
            csvwriter.writerow([img_name.split('.')[0],''.join(str(elem)+' ' for elem in boxes_to_csv).strip()])

    print('Generated submission file:{} \n'.format(submission_file))        



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Simple script for visualizing result of training.')

    parser.add_argument('--image_dir', help='Path to directory containing images')
    parser.add_argument('--model_path', help='Path to model')
    parser.add_argument('--viz_path')
    parser.add_argument('--prepare_test_data_from_dicom',action='store_true')
    parser.add_argument('--test_dicom_path')
    parser.add_argument('--save_path')

    parser = parser.parse_args()
    save_path_viz = parser.viz_path
    save_path = parser.save_path
    prepare_test_data_from_dicom = parser.prepare_test_data_from_dicom
    if prepare_test_data_from_dicom:
        test_dicom_path = parser.test_dicom_path
        makedir(save_path_viz)
        prepare_and_visualize_data(test_dicom_path, save_path_viz, '../data/bboxes_test.pkl')
    
    makedir(save_path)

    data_df = pd.read_csv('../data/train.csv')
    class_id_to_name_mapping = class_id_name_mapping(data_df)
    models = sorted(glob.glob(os.path.join(parser.model_path,'*')))
    for model_path in models:
        detect_image(parser.image_dir, model_path, class_id_to_name_mapping, save_path)
