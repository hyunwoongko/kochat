import torch
from torch import nn


class EntityRecognizer(Inference):

    def __init__(self, embed, model):
        self.tok = Tokenizer()
        self.load_dataset(embed)
        self.model = model.Net(len(self.data.entity_label_dict)).cuda()
        self.model.load_state_dict(torch.load(self.conf.entity_storefile))
        self.model.eval()
        self.embed = embed
        self.softmax = nn.Softmax()

    def load_dataset(self, embed):
        self.train_data, self.test_data = \
            dataset_.entity_train(embed)

    def recognize(self, text):
        tokenized = self.tok.tokenize(text)
        input_vector_size = len(tokenized)
        embedded = self.embed.embed(tokenized)
        sequence = dataset_.pad_sequencing(embedded)
        sequence = sequence.unsqueeze(0).cuda()

        output = self.model(sequence).float()
        output = output.squeeze().t()[0:input_vector_size]
        _, predict = torch.max(output, dim=1)

        output = [list(dataset_.entity_label_dict.keys())[i.item()] for i in predict]
        return ' '.join(output)
