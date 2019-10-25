from torch.utils.data import DataLoader
from utils.data.Dataset import ComDataset
import numpy as np
from torchvision import transforms

def lstm_cycle_dataloader(config):


	# First robot data
	first_rob = np.load(config.DATALOADER.FIRTS_ROBOT_PATH)

	# Second robot data
	second_rob = np.load(config.DATALOADER.SECOND_ROBOT_PATH)

    main_inputs = [torch.from_numpy(x).float() for x in [y[0:10] for y in first_rob["robot"]] ]

	main_inputs_cor = torch.from_numpy(first_rob["relevant"]).int()
	real_inputs_cor = torch.from_numpy(second_rob["relevant"]).int() 
	real_inputs = [torch.from_numpy(x).float() for x in [y[0:10] for y in second_rob["robot"]] ]
    dataloader = torch.utils.data.DataLoader(ComDataset(real_inputs,main_inputs), batch_size=config.TRAIN.BATCH_SIZE,
                                         shuffle=config.DATALOADER.SHUFFLE, num_workers=config.DATALOADER.WORKERS)

    return dataloader
