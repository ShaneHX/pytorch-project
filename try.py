
from loguru import logger
from data_loader.cifar_loader import CifarDataset, CifarDataLoader
from model import Demo
import multiprocessing
multiprocessing.set_start_method('spawn', True)

# logger.add("file_{time}.log")
if __name__ == "__main__":
    # txt_path = "/home/shanehan/workspace/project_ws/pytorch-project/data/train.txt"
    # dataset = CifarDataset(txt_path)
    # cifar_dataload = CifarDataLoader(data_index_txt=txt_path,
    #                                  batch_size=16)

    # net = Demo()
    # print(net)
