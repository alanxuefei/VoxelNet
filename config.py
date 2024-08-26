from easydict import EasyDict as edict

__C = edict()
cfg = __C

# Dataset Config
__C.DATASETS = edict()
__C.DATASETS.SHAPENET = edict()
# __C.DATASETS.SHAPENET.TAXONOMY_FILE_PATH = './datasets/ShapeNet_lamp.json'
__C.DATASETS.SHAPENET.TAXONOMY_FILE_PATH = './datasets/ShapeNet.json'
__C.DATASETS.SHAPENET.RENDERING_PATH = './datasets/ShapeNetRendering/%s/%s/rendering/%02d.png'
__C.DATASETS.SHAPENET.VOXEL_PATH = './datasets/ShapeNetVox32/%s/%s/model.binvox'

# Dataset
__C.DATASET = edict()
__C.DATASET.TRAIN_DATASET = 'ShapeNet'
__C.DATASET.TEST_DATASET = 'ShapeNet'

# Common
__C.CONST = edict()
__C.CONST.RNG_SEED = 0
__C.CONST.IMG_W = 224  # Image width for input
__C.CONST.IMG_H = 224  # Image height for input
__C.CONST.CROP_IMG_W = 128  # Dummy property for Pascal 3D
__C.CONST.CROP_IMG_H = 128  # Dummy property for Pascal 3D
__C.CONST.BATCH_SIZE_PER_GPU = 16
__C.CONST.N_VIEWS_RENDERING = 3
__C.CONST.NUM_WORKER = 20  # number of data workers
__C.CONST.WEIGHTS = ''

# Directories
__C.DIR = edict()
__C.DIR.OUT_PATH = './output/'

# Network
__C.NETWORK = edict()
__C.NETWORK.EMBED_DIM = 768
__C.NETWORK.COMBINED_DIM = 1024
__C.NETWORK.OUTPUT_SHAPE = (32, 32, 32)
__C.NETWORK.ATTENTION_HEADS = 8  # Add this line to define the number of attention heads
__C.NETWORK.DROPOUT_RATE = 0.1  # You might also want to define the dropout rate if it's used

# Training
__C.TRAIN = edict()
__C.TRAIN.RESUME_TRAIN = False
__C.TRAIN.SYNC_BN = True
__C.TRAIN.NUM_EPOCHS = 1500
__C.TRAIN.BRIGHTNESS = .4
__C.TRAIN.CONTRAST = .4
__C.TRAIN.SATURATION = .4
__C.TRAIN.NOISE_STD = .1
__C.TRAIN.RANDOM_BG_COLOR_RANGE = [[225, 255], [225, 255], [225, 255]]
__C.TRAIN.LEARNING_RATE = 1e-4
__C.TRAIN.LR_SCHEDULER = 'MilestonesLR'  # 'ExponentialLR' or 'MilestonesLR'
__C.TRAIN.WARMUP = 0
# for ExponentialLR
__C.TRAIN.EXPONENTIALLR = edict()
__C.TRAIN.EXPONENTIALLR.SCHEDULE_FACTOR = 1
# for MilestonesLR
__C.TRAIN.MILESTONESLR = edict()
__C.TRAIN.MILESTONESLR.LR_MILESTONES = [50, 120]
__C.TRAIN.MILESTONESLR.GAMMA = .1
__C.TRAIN.BETAS = (.9, .999)
__C.TRAIN.SAVE_FREQ = 30  # weights will be overwritten every save_freq epoch
__C.TRAIN.SHOW_TRAIN_STATE = 500
__C.TRAIN.LOSS = 2  # 1 for 'bce'; 2 for 'dice'; 3 for 'ce_dice'; 4 for 'focal'
__C.TRAIN.TEST_AFTER_TRAIN = True

# Testing options
__C.TEST = edict()
__C.TEST.RANDOM_BG_COLOR_RANGE = [[240, 240], [240, 240], [240, 240]]
__C.TEST.VOXEL_THRESH = [.3, .4, .5, .6]

# Refiner Configuration
__C.REFINER = edict()
__C.REFINER.LEARNING_RATE = 5e-5  # Learning rate specific to the refiner
__C.REFINER.LR_MILESTONES = [30, 100]  # Learning rate milestones for the refiner
__C.REFINER.GAMMA = .2  # Learning rate decay factor for the refiner
__C.REFINER.LEAKY_VALUE = 0.2  # LeakyReLU negative slope for the refiner
__C.REFINER.TCONV_USE_BIAS = False  # Whether to use bias in the transposed convolution layers
__C.REFINER.N_VOX = 32

# Add checkpoint file path to the configuration
__C.CHECKPOINT_FILE = '/workspace/output/checkpoints/2024-08-25T14:09:33.942981/checkpoint-epoch-027.pth'
__C.RESUME_TRAIN = True