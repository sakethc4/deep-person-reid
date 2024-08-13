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
        def __init__(self, model, aggregation='avg'):
            super(ReshapeModel, self).__init__()
            self.model = model
            self.aggregation = aggregation

        def forward(self, x):
            batch_size, seq_len, channels, height, width = x.shape
            print(f'Original input shape: {x.shape}')
            
            # Reshape to (batch_size * seq_len, channels, height, width)
            x = x.view(-1, channels, height, width)
            print(f'Reshaped input shape: {x.shape}')
            
            # Forward pass through the model
            x = self.model(x)
            
            # Reshape back to (batch_size, seq_len, output_dim)
            output_dim = x.shape[1]
            x = x.view(batch_size, seq_len, output_dim)
            print(f'Output shape before aggregation: {x.shape}')
            
            # Aggregate features across the sequence dimension
            if self.aggregation == 'avg':
                x = x.mean(dim=1)  # Average pooling
            elif self.aggregation == 'max':
                x = x.max(dim=1)[0]  # Max pooling
            else:
                raise ValueError(f"Unknown aggregation method: {self.aggregation}")
            
            print(f'Output shape after aggregation: {x.shape}')
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