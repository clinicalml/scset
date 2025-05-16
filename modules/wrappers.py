import torch
from torch import nn
 
class HeadedEncoderWrapper(nn.Module):

    def __init__(self, encoder, head, freeze_encoder=False):
        super().__init__()
        self.encoder = encoder
        self.head = head
        self.freeze_encoder = freeze_encoder

        if self.freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        else:
            for param in self.encoder.parameters():
                param.requires_grad = True

    def forward(self, x, mask=None, cellstosort=None):
        """
        cellstosort is present in the case of a cellsorter prediction head
        """
        # embed using encoder
        if self.freeze_encoder:
            with torch.no_grad():
                x = self.encoder(x, X_mask=mask)
        else:
            x = self.encoder(x, X_mask=mask)

        # if cellsorter head, concat patient embeddings and cellstosort
        if cellstosort is not None:
            # repeat patient embedding ncells time and concatenate
            x = torch.repeat_interleave(x, cellstosort.shape[1], dim=0) #shape[1] is ncells
            # Ensure repeated tensor is back in computation graph, as repeat_interleave can break the graph
            if not x.requires_grad:
                x = x.clone().detach().requires_grad_(True)

            # reshape cellstosort from B x N x D to BN x D (where N is N cells to sort)
            cellstosort = cellstosort.reshape(-1, cellstosort.shape[-1])
            x = torch.cat((x, cellstosort), dim=1)
        
        # pass through prediction head
        return self.head(x)