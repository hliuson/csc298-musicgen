import torch

### Autoencoder for a 32-step pianoroll (shape: (batch_size, 32, 128))
### Encoder: Convolutional -> Fully Connected
### Decoder: Fully Connected -> Convolutional
### We use the convolutional layers to extract local ideas

class ConvAutoEncoder(torch.nn.Module):
    def __init__(self, conv_depth = 2, fc_depth = 1, conv_width = 32, fc_width=1024,
                 encoded_length=512, dropout=0.5, pool = False): 
        super().__init__()
        self.encodeConv = torch.nn.Sequential()
        self.encodeFc = torch.nn.Sequential()
        self.decoderConv = torch.nn.Sequential()
        self.decoderFc = torch.nn.Sequential()
        
        self.poolFactor = 2**conv_depth

        
        print("Constructing ConvAutoEncoder")
        for i in range(conv_depth):
            if i == 0: 
                self.encodeConv.add_module(f'conv_{i}', torch.nn.Conv1d(in_channels=128, out_channels=conv_width, kernel_size=3, padding=1))
            else:
                self.encodeConv.add_module(f'conv_{i}', torch.nn.Conv1d(in_channels=conv_width, out_channels=conv_width, kernel_size=3, padding=1))
            #self.encoder.add_module(f'batchnorm_{i}', torch.nn.BatchNorm1d(conv_width))
            self.encodeConv.add_module(f'relu_{i}', torch.nn.ReLU())
            self.encodeConv.add_module(f'dropout_{i}', torch.nn.Dropout(dropout))
            self.encodeConv.add_module(f'pool_{i}', torch.nn.MaxPool1d(kernel_size=2))
        print("Convolutional layers constructed")
        for i in range(fc_depth):
            if i == 0:
                self.encodeFc.add_module(f'fc_{i}', torch.nn.Linear(32*conv_width // self.poolFactor, fc_width))
            elif i == fc_depth - 1:
                self.encodeFc.add_module(f'fc_{i}', torch.nn.Linear(fc_width, encoded_length))
            else:
                self.encodeFc.add_module(f'fc_{i}', torch.nn.Linear(fc_width, fc_width))
            self.encodeFc.add_module(f'relu_{i}', torch.nn.ReLU())
            self.encodeFc.add_module(f'dropout_{i}', torch.nn.Dropout(dropout))
        print("Fully connected layers constructed")
        for i in range(fc_depth):
            if i == 0:
                self.decoderFc.add_module(f'fc_{i}', torch.nn.Linear(encoded_length, fc_width))
            elif i == fc_depth - 1:
                self.decoderFc.add_module(f'fc_{i}', torch.nn.Linear(fc_width, 32*conv_width // self.poolFactor))
            else:
                self.decoderFc.add_module(f'fc_{i}', torch.nn.Linear(fc_width, fc_width))
            self.decoderFc.add_module(f'relu_{i}', torch.nn.ReLU())
            self.decoderFc.add_module(f'dropout_{i}', torch.nn.Dropout(dropout))
        print("Decoder fully connected layers constructed")
        for i in range(conv_depth):
            if i == 0: 
                self.decoderConv.add_module(f'conv_{i}', torch.nn.ConvTranspose1d(in_channels=conv_width, out_channels=128, kernel_size=3, padding=1))
            else:
                self.decoderConv.add_module(f'conv_{i}', torch.nn.ConvTranspose1d(in_channels=conv_width, out_channels=conv_width, kernel_size=3, padding=1))
            #self.decoder.add_module(f'batchnorm_{i}', torch.nn.BatchNorm1d(conv_width))
            self.decoderConv.add_module(f'relu_{i}', torch.nn.ReLU())
            self.decoderConv.add_module(f'dropout_{i}', torch.nn.Dropout(dropout))
            self.decoderConv.add_module(f'pool_{i}', torch.nn.MaxUnpool1d(kernel_size=2))
        print("Decoder convolutional layers constructed")
        
    def forward(self, x):
        x = self.encodeConv(x)
        x = x.view(x.size(0), -1)
        x = self.encodeFc(x)
        x = self.decoderFc(x)
        x = x.view(x.size(0), 32 // self.poolFactor, -1)
        x = self.decoderConv(x)
        return x
        
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, x):
        return self.decoder(x)