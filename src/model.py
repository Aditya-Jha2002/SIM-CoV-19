import timm
import torch.nn as nn

class ImageClassifier(nn.Module):
    def __init__(self,model_arch,n_class,pretrained=False):
        super().__init__()
        self.model = timm.create_model(model_arch,pretrained=pretrained)
        n_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(n_features,n_class)

    def forward(self,x):
        x=self.model(x)
        return x