import torchreid
import torch
from torchreid.data.datasets.video import iLIDSVID


# Imported Model
model = torchreid.models.build_model(
    name='osnet_x1_0',
    num_classes=751,
    pretrained=True
)

#torchreid.data.register_video_dataset('ilids-vid', iLIDSVID)
#torchreid.data.datasets.init_video_dataset('ilids-vid')


# Attempting to use our own data
datamanager = torchreid.data.VideoDataManager(
    root='.',
    sources='ilidsvid',
    height=128,
    width=64,
    batch_size_train=32,
    batch_size_test=30,
)

# Attempting to set up engine for testing
engine = torchreid.engine.ImageSoftmaxEngine(
    datamanager=datamanager,
    model=model,
    optimizer=None,  # No optimizer for testing
    scheduler=None,  # No scheduler for testing
    use_gpu=torch.cuda.is_available()
)

# Run testing
engine.run(
    max_epoch=0,       # No training
    save_dir='log/osnet_x1_0',
    print_freq=10,
    test_only=True     # Only test the model
)