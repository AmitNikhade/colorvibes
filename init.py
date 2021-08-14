import tensorflow
print(tensorflow.__version__)

import logging
logger = logging.getLogger()

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',level=logging.INFO )
logger.error('dataset path must be specified')
logger.info('Loading data..')