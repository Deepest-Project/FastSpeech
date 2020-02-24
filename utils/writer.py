import os
from torch.utils.tensorboard import SummaryWriter

def get_writer(output_directory, log_directory):
    logging_path=f'{output_directory}/{log_directory}'
    
    if os.path.exists(logging_path):
        raise Exception('The experiment already exists')
    else:
        os.mkdir(logging_path)
        writer = SummaryWriter(logging_path)
            
    return writer