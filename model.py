from torchvision import models
from torch import nn
from parser_my import args

class Mydense121(nn.Module):
	def __init__(self, num_class, weights=True):
		super(Mydense121, self).__init__()
		self.model = models.densenet121(weights)
		self.model.classifier = nn.Linear(in_features=1024, out_features=num_class, bias=True)
		self.softmax = nn.Softmax()
		
	def forward(self,x):
		x = self.softmax(self.model(x))
		return x