import torch
from torch import nn

from base.model_managers.model_manager import Entity


class EntityRecognizer(Entity):

    def __init__(self, model, label_dict):
        super().__init__()
        self.label_dict = label_dict
        self.model = model.Model(vector_size=self.vector_size,
                                 d_model=self.d_model,
                                 layers=self.layers,
                                 classes=len(self.label_dict))

        self.model.load_state_dict(torch.load(self.intent_classifier_file))
        self.model = self.model.cuda()
        self.model.eval()  # eval 모드 (필수)

    def inference_model(self, sequence):
        output = self.model(sequence).float()
        output = output.squeeze().t()[0:input_vector_size]
        _, predict = torch.max(output, dim=0)
        output = [list(self.label_dict.keys())[i.item()] for i in predict]
        return ' '.join(output)
