import albumentations as A
image_size=512

train_transforms = A.Compose([   #flip horizontal, randomcrop, scaling, clahe, randombrightnesscontrast, shiftscalerotate, blur
		A.ShiftScaleRotate(p=0.1),
		A.HorizontalFlip(p=0.5),
		#A.RandomCrop(512, 512, p=0.05),
		A.CLAHE(p=0.5),
 		A.OneOf([
            A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit= 0.2, 
                                 val_shift_limit=0.2, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, 
                                       contrast_limit=0.2, p=0.5),
        ],p=0.5),		
		A.OneOf([
				A.MedianBlur(p=0.5),
				A.GaussianBlur(p=0.5)
			], p=0.5),
		A.Normalize(p=1.0),
		A.Resize(height=image_size, width=image_size, p=1)
	], bbox_params = A.BboxParams(format='pascal_voc', min_visibility=0.9))

train_transforms_only_image = A.Compose([   #flip horizontal, randomcrop, scaling, clahe, randombrightnesscontrast, shiftscalerotate, blur
		A.ShiftScaleRotate(p=0.1),
		A.HorizontalFlip(p=0.5),
		#A.RandomCrop(512, 512, p=0.05),
		A.CLAHE(p=0.5),
 		A.OneOf([
            A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit= 0.2, 
                                 val_shift_limit=0.2, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, 
                                       contrast_limit=0.2, p=0.5),
        ],p=0.5),		
		A.OneOf([
				A.MedianBlur(p=0.5),
				A.GaussianBlur(p=0.5)
			], p=0.5),
		A.Normalize(p=1.0),
		A.Resize(height=image_size, width=image_size, p=1)
	])

test_transforms = A.Compose([
		A.Normalize(p=1.0),
		A.Resize(height=image_size, width=image_size, p=1)
	], bbox_params = A.BboxParams(format='pascal_voc'))

test_transforms_only_image = A.Compose([
		A.Normalize(p=1.0),
		A.Resize(height=image_size, width=image_size, p=1)
	])


	