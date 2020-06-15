import torch

from backend.base.model_manager import Entity


class EntityRecognizer(Entity):

    def __init__(self, model, label_dict):
        super().__init__()
        self.label_dict = label_dict
        self.model = model.Model(vector_size=self.vector_size,
                                 d_model=self.d_model,
                                 layers=self.layers,
                                 classes=len(self.label_dict),
                                 device=self.device)

        self.model.load_state_dict(torch.load(self.entity_recognizer_file))
        self.model = self.model.to(self.device)
        self.model.eval()  # eval 모드 (필수)

    def inference_model(self, sequence):
        length = self.get_length(sequence)
        output = self.model(sequence).float()
        output = output.squeeze().t()
        _, predict = torch.max(output, dim=1)
        output = [list(self.label_dict.keys())[i.item()] for i in predict]
        return ' '.join(output[:length])

    @staticmethod
    def get_length(sequence):
        """
        pad는 [0...0]이니까 1더해서 [1...1]로
        만들고 all로 검사해서 pad가 아닌 부분만 세기
        """
        sequence = sequence.squeeze()
        return [all(map(int, (i + 1).tolist()))
                for i in sequence].count(False)
