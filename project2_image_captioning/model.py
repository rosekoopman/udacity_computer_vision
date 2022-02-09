import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, batch_size=10, drop_prob=0):  # Rose: add batchsize        
        super(DecoderRNN, self).__init__()
        
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=drop_prob)
        self.fc = nn.Linear(hidden_size, vocab_size)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.hidden = self.init_hidden(batch_size)
    
    def forward(self, features, captions):
        
        # remove last word from caption as there is no ground truth next word
        captions = captions[:, :-1]
        
        # embed the captions
        embeddings = self.embedding(captions)
        
        # add extra dimension to features (length_caption), such that dimensions are the same as for the captions
        inputs = torch.cat((features.unsqueeze(dim=1), embeddings), dim=1)
        
        # feed through LSTM and Linear layers
        out, self.hidden = self.lstm(inputs, self.hidden)
        x = self.fc(out)
        
        return x
        
    def init_hidden(self, batch_size):
        ''' At the start of training, we need to initialize a hidden state;
            there will be none because the hidden state is formed based on perviously seen data.
            So, this function defines a hidden state with all zeroes and of a specified size.'''
        
        # The axes dimensions are (num_layers, batch_size, hidden_size). 
        if self.device == "cuda":
            return (torch.zeros(self.num_layers, batch_size, self.hidden_size).cuda(),
                    torch.zeros(self.num_layers, batch_size, self.hidden_size).cuda())
        else:
            return (torch.zeros(self.num_layers, batch_size, self.hidden_size),
                    torch.zeros(self.num_layers, batch_size, self.hidden_size))
    

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        
        self.hidden = self.init_hidden(1)
        
        outputs = []
        for i in range(max_len):
            
            # feed through decoder
            out, self.hidden = self.lstm(inputs, self.hidden)
            x = self.fc(out)
            
            # get idx of maximum value
            idx = torch.argmax(x)
            
            # break if stop-token
            if idx.item() == 1:
                break
                
            # append token-idx to output
            outputs.append(idx.item())
            
            # roll-over to the next state. Note that we need to embed the token using the word embedding layer
            inputs = self.embedding(idx.view(1,-1))
            
        
        return outputs

    

