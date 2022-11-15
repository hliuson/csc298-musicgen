import torch
import torch.optim as optim
import pytorch_lightning as pl


### Autoencoder for a 32-step pianoroll (shape: (batch_size, 32, 128))
### Encoder: Convolutional -> Fully Connected
### Decoder: Fully Connected -> Convolutional
### We use the convolutional layers to extract local ideas

class ConvAutoEncoder(torch.nn.Module):
    def __init__(self, conv_depth = 2, fc_depth = 1, conv_width = 32, fc_width=1024,
                 encoded_length=512, dropout=0.5): 
        super().__init__()
        self.encodeConv = torch.nn.Sequential()
        self.encodeFc = torch.nn.Sequential()
        self.decoderConv = torch.nn.Sequential()
        self.decoderFc = torch.nn.Sequential()
        
        self.poolFactor = 1#2**conv_depth

    
        for i in range(conv_depth):
            in_channels = conv_width
            out_channels = conv_width
            if i == 0:
                in_channels = 128
            self.encodeConv.add_module(f'conv_{i}', torch.nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1))
            #self.encoder.add_module(f'batchnorm_{i}', torch.nn.BatchNorm1d(conv_width))
            self.encodeConv.add_module(f'relu_{i}', torch.nn.ReLU())
            self.encodeConv.add_module(f'dropout_{i}', torch.nn.Dropout(dropout))
            #self.encodeConv.add_module(f'pool_{i}', torch.nn.MaxPool1d(kernel_size=2))
            
        for i in range(fc_depth):
            in_dim = fc_width
            out_dim = fc_width
            if i == 0:
                in_dim = 32*conv_width // self.poolFactor
            if i == fc_depth - 1:
                out_dim = encoded_length
            self.encodeFc.add_module(f'fc_{i}', torch.nn.Linear(in_dim, out_dim))
            self.encodeFc.add_module(f'relu_{i}', torch.nn.ReLU())
            self.encodeFc.add_module(f'dropout_{i}', torch.nn.Dropout(dropout))
            
        for i in range(fc_depth):
            in_dim = fc_width
            out_dim = fc_width
            if i == 0:
                in_dim = encoded_length
            if i == fc_depth - 1:
                out_dim = 32*conv_width // self.poolFactor
            self.decoderFc.add_module(f'fc_{i}', torch.nn.Linear(in_dim, out_dim))
            self.decoderFc.add_module(f'relu_{i}', torch.nn.ReLU())
            self.decoderFc.add_module(f'dropout_{i}', torch.nn.Dropout(dropout))
            
        for i in range(conv_depth):
            in_channels = conv_width
            out_channels = conv_width
            if i == conv_depth - 1:
                out_channels = 128
            self.decoderConv.add_module(f'conv_{i}', torch.nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1))
            self.decoderConv.add_module(f'relu_{i}', torch.nn.ReLU())
            self.decoderConv.add_module(f'dropout_{i}', torch.nn.Dropout(dropout))
            #self.decoderConv.add_module(f'pool_{i}', torch.nn.ConvTranspose1d(in_channels=conv_width, out_channels=conv_width, kernel_size=2, stride=2))
        
    def forward(self, x):
        #Conv1d expects data in format (batch, channels, length)
        x = x.view(x.size(0), 128, -1)
        x = self.encodeConv(x)
        x = x.view(x.size(0), -1)
        x = self.encodeFc(x)
        x = self.decoderFc(x)
        x = x.view(x.size(0), -1, 32 // self.poolFactor)
        x = self.decoderConv(x)
        x = x.view(x.size(0), -1, 128)
        return x
        
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, x):
        return self.decoder(x)

class LightningConvAutoencoder(pl.LightningModule):
    def __init__(self, model = None):
        super().__init__()
        if model is None:
            model = ConvAutoEncoder()
        self.model = model
        self.loss = torch.nn.MSELoss()
        
        self.save_hyperparameters()
        
    def training_step(self, batch, batch_idx):
        output = self.model(batch)
        loss = self.loss(output, batch)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        output = self.model(batch)
        loss = self.loss(output, batch)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-5)