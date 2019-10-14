from torchvision import transforms
from torch.utils.data import Dataset
from loguru import logger
from base import BaseDataLoader
from PIL import Image
import os


class CifarDataset(Dataset):
    def __init__(self, data_index_txt, transform=None):
        """[summary]
        Build a dataset for the dataload of pytorch
        Parameters
        ----------
        Dataset : [type]: torch.utils.data.Dataset
            [description]: The basic dataset class of pytorch
        data_index_txt : [type]: string
            [description]: The path of the data index file which is contain two words,
                        first, the abslute path of each image.
                        seconde, the index of the class of this image belong
        """
        self._transform = transform
        self._img_labels = list()
        with open(data_index_txt, 'r') as fh:
            data_lines = fh.readlines()
            for s_line in data_lines:
                s_line.strip()
                splited_line = s_line.split(" ")
                if(len(splited_line) != 2):
                    logger.error("Find a error data index, skip it...")
                    continue
                if not os.path.exists(splited_line[0]):
                    logger.error(
                        "Load image failed. {}".format(splited_line[0]))
                    continue
                self._img_labels.append(
                    (splited_line[0], int(splited_line[1])))
        logger.info("Prepare dataset done, the data size: {}"
                    .format(len(self._img_labels)))

    def __len__(self):
        return len(self._img_labels)

    def __getitem__(self, index):
        img_path, label = self._img_labels[index]
        img = Image.open(img_path).convert('RGB')
        if self._transform is not None:
            img = self._transform(img)

        return img, label


class CifarDataLoader(BaseDataLoader):
    def __init__(self,
                 data_dir,
                 batch_size,
                 shuffle=True,
                 validation_split=0.1,
                 num_workers=1):
        normMean = [0.4948052, 0.48568845, 0.44682974]
        normStd = [0.24580306, 0.24236229, 0.2603115]
        normTransform = transforms.Normalize(normMean, normStd)
        trsfm = transforms.Compose([
            transforms.Resize(32),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            normTransform])
        self._dataset = CifarDataset(data_dir, trsfm)
        super().__init__(self._dataset, batch_size, shuffle,
                         validation_split, num_workers)
