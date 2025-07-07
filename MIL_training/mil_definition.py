'''
File to introduce a MIL model
Note that loads of hyperparameters could be included as arguments
It could be avg pooling size, hidden dim, etc...
Also encoder could be changed to a different model
'''

import torch
import torch.nn as nn
import timm



class MILsection(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers=1):
        super(MILsection, self).__init__()
        self.num_layers = num_layers
        # RNN cells, here we use a GRU  
        if num_layers > 0:
            self.rnn = nn.GRU(input_dim, input_dim//2, num_layers=num_layers,
                             batch_first=True, dropout=0.1, bidirectional=True)
        # attention layer, here we use a simple linear layer
        self.attention = nn.Sequential(
            nn.Tanh(),
            nn.Linear(input_dim, 1)
        )

    def forward(self, bags):
        """
        Args:
            bags: (batch_size, num_instances, input_dim)

        Returns:
            logits: (batch_size, num_classes)
        """

        # bags iterates in the RNN
        if self.num_layers > 0:
            bags_rnn, _ = self.rnn(bags)
        else:
            bags_rnn = bags
        
        # main attention
        attn_scores = self.attention(bags_rnn).squeeze(-1)  # [batch_size, num_instances]
        attn_weights = torch.softmax(attn_scores, dim=-1)  # [batch_size, num_instances]
        weighted_instances = torch.bmm(attn_weights.unsqueeze(1), bags_rnn).squeeze(1)  # [batch_size, input_dim]
        
        return weighted_instances


class MILmodel(nn.Module):
    # attention, a new encoder instance has to be created at each MIL creation
    # if not, the same encoder could be used for all MIL sections
    def __init__(self, encoder, num_layers=1):
        super(MILmodel, self).__init__()
        # encoder
        self.encoder = encoder
        # flattening layer, applying pooling and flattening
        # note here that we could try different pooling methods
        self.flatten = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1)
        )
        # size of the features after encoding
        # encoder outputs n feature maps 
        # that are flattened into a vector of size n 
        self.feature_size = self.encoder.num_features
        
        # MIL section, loads of hyperparameters here also
        self.mil_section = MILsection(input_dim=self.feature_size,
                                    hidden_dim=1024, 
                                    num_classes=3,
                                    num_layers=num_layers)
        # classifier output
        # we use a final simple linear layer to output the final prediction 
        self.classifier = nn.Linear(self.feature_size, 3)

    def forward(self, x):
        # x shape: (batch_size, 6, 1, 384, 384)
        batch_size, num_instances, channels, H, W = x.shape

        # Reshape to process all instances through encoder
        x = x.reshape(-1, channels, H, W)  # shape: (batch_size * 6, 1, 384, 384)
        
        # Pass through encoder
        x = self.encoder.forward_features(x)  # shape: (batch_size * 6, feature_size, h', w')
        
        # Apply pooling and flatten
        x = self.flatten(x)  # shape: (batch_size * 6, feature_size)
        
        # Reshape back to separate instances
        x = x.reshape(batch_size, num_instances, self.feature_size)  # shape: (batch_size, 6, feature_size)
        
        # Pass through MIL section
        weighted_instances = self.mil_section(x)
        # weighted_instances: (batch_size, feature_size)
        
        # classification output
        output = self.classifier(weighted_instances)  # shape: (batch_size, 3)
        
        return output
