'''
File to introduce a MIL model
Note that loads of hyperparameters could be included as arguments
It could be avg pooling size, hidden dim, etc...
Also encoder could be changed to a different model
'''

import torch
import torch.nn as nn

# import timm for models
import timm


# define a MIL model
class MILsection(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(MILsection, self).__init__()
        # here add a RNN bidirectional layer, to get contex
        # self.lstm = nn.LSTM(input_dim, input_dim//2, num_layers=2,
        # batch_first=True, dropout=0.1, bidirectional=True)
        self.aux_attention = nn.Sequential(
            nn.Tanh(),
            nn.Linear(input_dim, 1)
        )
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
        batch_size, num_instances, input_dim = bags.size()
        # bags_lstm, _ = self.lstm(bags)
        bags_lstm = bags
        attn_scores = self.attention(bags_lstm).squeeze(-1)
        aux_attn_scores = self.aux_attention(bags_lstm).squeeze(-1)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        weighted_instances = torch.bmm(attn_weights.unsqueeze(1),
                                       bags_lstm).squeeze(1)
        return weighted_instances, aux_attn_scores


# here define the whole MIL model
# uses the MILsection model and a ConvNext Small as a feature extractor
# note that loads of hyperparameters could be included as arguments
class MILmodel(nn.Module):
    def __init__(self, encoder):
        super(MILmodel, self).__init__()
        # encoder
        self.encoder = encoder
        # flattening layer, applying pooling and flattening
        # note here that we could try different pooling methods
        self.flatten = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(1))
        self.feature_size = self.encoder.feature_size
        # MIL section, loads of hyperparameters here also
        self.mil_section = MILsection(input_dim=self.feature_size,
                                      hidden_dim=1024, num_classes=3)
        # classifier output
        self.classifier = nn.Linear(self.feature_size, 3)
        # aux classifier output
        self.aux_output = nn.Linear(self.feature_size, 3)

    def forward(self, x):
        # x shape: (batch_size, 6, 1, 384, 384)
        batch_size, num_instances, channels, H, W = x.shape

        # Reshape to process all instances through encoder
        x = x.view(-1, channels, H, W)  # shape: (batch_size * 6, 1, 384, 384)

        # Pass through encoder
        x = self.encoder(x)  # shape: (batch_size * 6, feature_size, h', w')

        # Apply pooling and flatten
        x = self.flatten(x)  # shape: (batch_size * 6, feature_size)

        # Reshape back to separate instances
        x = x.view(batch_size, num_instances, -1)
        # shape: (batch_size, 6, feature_size)

        # Pass through MIL section
        x, aux_output = self.mil_section(x)
        # shape: (batch_size, feature_size), (batch_size, 6)

        # Final classifications
        output = self.classifier(x)  # shape: (batch_size, 3)
        output_aux = self.aux_output(aux_output)  # shape: (batch_size, 3)

        return output, output_aux


convnext_small = timm.create_model('convnext_small.fb_in22k_ft_in1k_384',
                                   in_chans=1, pretrained=True, num_classes=0)
