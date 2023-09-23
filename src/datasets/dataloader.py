import mlconfig
import torchvision.transforms as transforms
from torch.utils import data
from .datagen import ListDataset

# def fcos_dataset_collate(batch):
#     images  = []
#     bboxes  = []
#     classes = []
#     for img, box, classe in batch:
#         images.append(img)
#         bboxes.append(box)
#         classes.append(classe)
#     images  = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
#     bboxes  = torch.from_numpy(np.array(bboxes)).type(torch.FloatTensor)
#     classes = torch.from_numpy(np.array(classes)).type(torch.LongTensor)
#     return images, bboxes, classes

@mlconfig.register
class Fcos_DataLoader(data.DataLoader):

    def __init__(self, root: str, list_file: str, train: bool, batch_size: int, scale: int, strides: list, limit_range: list, sample_radiu_ratio: int, **kwargs):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((scale,scale)),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
        
        with open(list_file) as f:
            annotation_lines = f.readlines()
        
        dataset = ListDataset(root, annotation_lines, (scale,scale), transform, train, strides, limit_range, sample_radiu_ratio)
        
        super(Fcos_DataLoader, self).__init__(dataset=dataset, batch_size=batch_size, shuffle=train, **kwargs)