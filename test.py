import torchreid
import torch
from torchreid.data.datasets.video import iLIDSVID


def main():
    # Imported Model
    model = torchreid.models.build_model(
        name='resnext50_32x4d',
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
        batch_size_test=5,
    )

    # Wrap the model to handle the reshaping
    class ReshapeModel(torch.nn.Module):
        def __init__(self, model):
            super(ReshapeModel, self).__init__()
            self.model = model

        def forward(self, x):
            # Reshape from (batch_size, seq_len, channels, height, width)
            # to (batch_size * seq_len, channels, height, width)
            batch_size, seq_len, channels, height, width = x.shape
            x = x.view(-1, channels, height, width)
            x = self.model(x)
            return x
    model = ReshapeModel(model)

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
        save_dir='log/resnext50_32x4d',
        print_freq=10,
        test_only=True     # Only test the model
    )

if __name__ == '__main__':
    main()