import torch
from torch import nn

from base.model_managers.model_manager import Intent


class IntentClassifier(Intent):

    def __init__(self, model, label_dict):
        super().__init__()
        self.label_dict = label_dict
        self.model = model.Model(vector_size=self.vector_size,
                                 max_len=self.max_len,
                                 d_model=self.d_model,
                                 layers=self.layers,
                                 classes=len(self.label_dict))

        self.model = self.model.cuda()
        self.model.load_state_dict(torch.load(self.intent_classifier_file))
        self.softmax = nn.Softmax()

    def inference_model(self, sequence):
        output = self.model(sequence).float()
        output = self.model.classifier(output.squeeze())
        output = self.softmax(output)
        _, predict = torch.max(output, dim=0)
        self.label_dict = list(self.label_dict)
        print(output)
        return self.label_dict[predict.item()]
