# STRec is implemented based on Recbole.
# https://recbole.io/
import sys
import time
import torch
from logging import getLogger
import logging
from recbole.utils import init_logger, init_seed
from recbole.trainer import Trainer
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from models import STRec_pre, STRec, print_time



if __name__ == '__main__':
    start_time = time.time()
    
    config = Config(model=STRec,config_file_list=['STRec.yaml'])
    init_seed(config['seed'], config['reproducibility'])
    logging.getLogger().setLevel(logging.INFO)
    
    init_logger(config)
    logger = getLogger()

    logger.info(config)
    

    dataset = create_dataset(config)
    logger.info(dataset)

    train_data, valid_data, test_data = data_preparation(config, dataset)
    

    if config['mode'] == 'pre_train':
        config['model'] = 'STRec_pre'

    elif config['mode'] == 'train':
        config['model'] = 'STRec'
    
    elif config['mode'] == 'test':
        checkpoint = torch.load(config['load_dir'])
        model = locals()[config['model']](config, train_data.dataset).to(config['device'])
        model.load_state_dict(checkpoint['state_dict'])
        model.load_other_parameter(checkpoint.get('other_parameter'))
        trainer = Trainer(config, model)
        test_result = trainer.evaluate(test_data, load_best_model=False, show_progress=True)
        logger.info('test result: {}'.format(test_result))
        sys.exit()

    elif config['mode'] == 'speed':
        print_time(start_time)
        checkpoint = torch.load(config['load_dir'])
        model = locals()[config['model']](config, train_data.dataset).to(config['device'])
        model.load_state_dict(checkpoint['state_dict'])
        model.load_other_parameter(checkpoint.get('other_parameter'))
        model.eval()
        for batch_idx, batched_data in enumerate(train_data):
            if batch_idx==2:
                print(torch.cuda.max_memory_allocated())
            batched_data = batched_data.to(config['device'])
            out = model.predictx(batched_data)
        print_time(start_time)
        sys.exit()

    else:
        raise NotImplementedError("Make sure 'mode' in ['pre_train', 'train', 'test', 'speed']!")

    model = locals()[config['model']](config, train_data.dataset).to(config['device'])
    logger.info(model)

    trainer = Trainer(config, model)

    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data)

    test_result = trainer.evaluate(test_data)

    logger.info('best valid result: {}'.format(best_valid_result))
    logger.info('test result: {}'.format(test_result))