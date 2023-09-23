import os, random
from PIL import Image, ImageDraw
import numpy as np
import cv2
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

class ListDataset(data.Dataset):
    def __init__(self, root, annotation_lines, input_shape, transform, train, strides, limit_range, sample_radiu_ratio):
        super(ListDataset, self).__init__()
        
        self.root               = root
        self.annotation_lines   = annotation_lines
        self.length             = len(annotation_lines)
        self.input_shape        = input_shape
        self.transform          = transform
        self.train              = train
        self.strides            = strides
        self.limit_ranges        = limit_range
        self.sample_radiu_ratio = sample_radiu_ratio
        self.max_boxes          = 100

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        line    = self.annotation_lines[index].split()
        box     = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])
        image   = cv2.imread(os.path.join(self.root, line[0]))
        
        ih, iw, _  = image.shape
        h, w    = self.input_shape
        if self.train:
            image, box = self.random_data(image, box)
        image = self.transform(image)
        if self.train and random.randrange(1, 6) in [1, 5]:
            contrast_random = random.randrange(6, 15) / 10
            color_random = random.randrange(6, 15) / 10
            saturation_random = random.randrange(6, 15) / 10
            transform = transforms.Compose([
                    transforms.ColorJitter(brightness=color_random, contrast=contrast_random, saturation=saturation_random),
                ])
            image = transform(image)
        
        #---------------------------------#
        #   对真实框进行调整
        #---------------------------------#
        # box_data = np.zeros((self.max_boxes,5))
        if len(box)>0:
            np.random.shuffle(box)
            box[:, 0] = box[:, 0] / iw * w
            box[:, 2] = box[:, 2] / iw * w
            box[:, 1] = box[:, 1] / ih * h
            box[:, 3] = box[:, 3] / ih * h
            box[:, 0:2][box[:, 0:2]<0] = 0
            box[:, 2][box[:, 2]>w] = w
            box[:, 3][box[:, 3]>h] = h
            # box_w = box[:, 2] - box[:, 0]
            # box_h = box[:, 3] - box[:, 1]
            # box = box[np.logical_and(box_w>1, box_h>1)] # discard invalid box
            # if len(box)>self.max_boxes: box = box[:self.max_boxes]
            # box_data[:len(box)] = box
            
        box         = torch.FloatTensor(box)
        gt_boxes    = torch.FloatTensor(box[:, :4])
        classes     = torch.FloatTensor(box[:, 4])
        cls_targets, cnt_targets, reg_targets = self.preprocess_true_boxes(gt_boxes, classes)
#         return cls_targets, cnt_targets, reg_targets
        mask_pos    = cnt_targets > -1
        cls_targets = torch.cat([cls_targets, mask_pos], dim=-1)
        cnt_targets = torch.cat([cnt_targets, mask_pos], dim=-1)
        reg_targets = torch.cat([reg_targets, mask_pos], dim=-1)
        return image, cls_targets, cnt_targets, reg_targets
    
    @staticmethod
    def random_data(image, box):
        seq = iaa.Sequential([
                iaa.Sometimes(
                            0.5,
                            iaa.Affine(
                                rotate=(-4, 4),
                                scale={"x":(0.8,1.2),"y":(0.8,1.2)},
                                translate_percent={"x":(-0.2,0.2),"y":(-0.2,0.2)},
                                shear=(-4,4),
                                order=[0, 1],
                                cval=(0, 255),
                                mode="constant"),
                        ),
                iaa.Sometimes(
                            0.5,
                            # 锐化
                            iaa.Sharpen(alpha=(0, 0.2), lightness=(0.8, 1.2)),
                        ),
                iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 2.0)))
            
        ])
        bbs = BoundingBoxesOnImage([BoundingBox(x1=i[0], y1=i[1], x2=i[2], y2=i[3], label=i[4]) for i in box], shape=image.shape)
        image_aug, bbs_aug = seq(image=image, bounding_boxes=bbs)
        box_aug = np.array([np.array([int(b.x1),int(b.y1),int(b.x2),int(b.y2),int(b.label)]) for b in bbs_aug.bounding_boxes])
        
        return image_aug, box_aug
    
    @staticmethod
    def _get_grids(h, w, stride):
        shifts_x = torch.arange(0, w * stride, stride, dtype=torch.float32)
        shifts_y = torch.arange(0, h * stride, stride, dtype=torch.float32)

        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)

        shift_x = torch.reshape(shift_x, [-1])
        shift_y = torch.reshape(shift_y, [-1])
        grid    = torch.stack([shift_x, shift_y], -1) + stride // 2

        return grid
    
    def preprocess_true_boxes(self, gt_boxes, classes):
        cls_targets_all_level = []
        cnt_targets_all_level = []
        reg_targets_all_level = []
        
        output_length = ([int(self.input_shape[0]/stride) for stride in self.strides], [int(self.input_shape[1]/stride) for stride in self.strides])
        
        for level in range(len(output_length[0])):
            stride = self.strides[level]
            limit_range = self.limit_ranges[level]
            grids = self._get_grids(output_length[0][level], output_length[1][level], stride)
            h_mul_w     = output_length[0][level] * output_length[1][level]

            x           = grids[:, 0]
            y           = grids[:, 1]
            
            #----------------------------------------------------------------#
            #   左上点、右下点不可以差距很大
            #----------------------------------------------------------------#
            # 求真实框的左上角和右下角相比于特征点的偏移情况
            # [h*w, 1] - [1, m] --> [h*w, m]
            left_off    = x[:, None] - gt_boxes[..., 0][None, :]
            top_off     = y[:, None] - gt_boxes[..., 1][None, :]
            right_off   = gt_boxes[...,2][None, :] - x[:, None]
            bottom_off  = gt_boxes[...,3][None, :] - y[:, None]
            # [h*w, m, 4]
            ltrb_off = torch.stack([left_off, top_off, right_off, bottom_off],dim=-1)
            
            # 求每个框的面积
            # [h*w, m]
            areas = (ltrb_off[...,0] + ltrb_off[...,2]) * (ltrb_off[...,1] + ltrb_off[...,3])
            
            # [batch_size, h*w, m]
            off_min = torch.min(ltrb_off, dim=-1)[0]
            off_max = torch.max(ltrb_off, dim=-1)[0]
            
            # 将特征点不落在真实框内的特征点剔除。
            mask_in_gtboxes = off_min > 0
            # 前层特征适合小目标检测，深层特征适合大目标检测。
            mask_in_level = (off_max > limit_range[0]) & (off_max <= limit_range[1])
            
            #----------------------------------------------------------------#
            #   中心点不可以差距很大，求真实框中心相比于特征点的偏移情况
            #   计算真实框中心的x轴坐标
            #   计算真实框中心的y轴坐标
            #   [1,h*w,1] - [batch_size, 1, m] --> [batch_size,h * w, m]
            #----------------------------------------------------------------#
            radiu       = stride * self.sample_radiu_ratio
            # 计算真实框中心的x轴坐标+
            gt_center_x = (gt_boxes[...,0] + gt_boxes[...,2])/2
            # 计算真实框中心的y轴坐标
            gt_center_y = (gt_boxes[...,1] + gt_boxes[...,3])/2
            # [h*w,1] - [1,m] --> [h*w,m]
            c_left_off  = x[:, None] - gt_center_x[None, :]
            c_top_off   = y[:, None] - gt_center_y[None, :]
            c_right_off = gt_center_x[None, :] - x[:, None]
            c_bottom_off= gt_center_y[None, :] - y[:, None]
            
            # [h*w, m, 4]
            c_ltrb_off  = torch.stack([c_left_off, c_top_off, c_right_off, c_bottom_off],dim=-1)
            c_off_max   = torch.max(c_ltrb_off,dim=-1)[0]
            mask_center = c_off_max < radiu
            
            #   为正样本的特征点
            #   [h*w, m]
            mask_pos = mask_in_gtboxes & mask_in_level & mask_center
            
            #   将所有不是正样本的特征点，面积设成max
            #   [h*w, m]
            areas[~mask_pos] = 99999999
            #   选取该特征点对应面积最小的框
            #   [h*w]
            areas_min_ind = torch.min(areas, dim = -1)[1]
            
            # [h*w, 4]
            reg_targets = ltrb_off[torch.zeros_like(areas, dtype=torch.bool).scatter_(-1, areas_min_ind.unsqueeze(dim=-1), 1)]
            reg_targets = torch.reshape(reg_targets, (-1, 4))
            
            # [h*w, m]
            _classes    = torch.broadcast_tensors(classes[None, :], areas.long())[0]
            cls_targets = _classes[torch.zeros_like(areas, dtype=torch.bool).scatter_(-1, areas_min_ind.unsqueeze(dim=-1), 1)]
            
            # [h*w, 1]
            cls_targets = torch.reshape(cls_targets,(-1,1))
            
            # [h*w]
            left_right_min  = torch.min(reg_targets[..., 0], reg_targets[..., 2])
            left_right_max  = torch.max(reg_targets[..., 0], reg_targets[..., 2])
            top_bottom_min  = torch.min(reg_targets[..., 1], reg_targets[..., 3])
            top_bottom_max  = torch.max(reg_targets[..., 1], reg_targets[..., 3])
            
            # [h*w, 1]
            cnt_targets= ((left_right_min * top_bottom_min) / (left_right_max * top_bottom_max + 1e-10)).sqrt().unsqueeze(dim=-1)
            
            assert reg_targets.shape == (h_mul_w,4)
            assert cls_targets.shape == (h_mul_w,1)
            assert cnt_targets.shape == (h_mul_w,1)
            
            # process neg grids
            mask_pos_2 = mask_pos.long().sum(dim=-1) >= 1
            
            cls_targets[~mask_pos_2] = -1
            cnt_targets[~mask_pos_2] = -1
            reg_targets[~mask_pos_2] = -1

            cls_targets_all_level.append(cls_targets)
            cnt_targets_all_level.append(cnt_targets)
            reg_targets_all_level.append(reg_targets)
#         return cls_targets_all_level, cnt_targets_all_level, reg_targets_all_level
        return torch.cat(cls_targets_all_level), torch.cat(cnt_targets_all_level), torch.cat(reg_targets_all_level)
