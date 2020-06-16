import torch
from matplotlib import pyplot as plt
from torch import nn, optim
from torch.autograd import Variable
from torch.optim import lr_scheduler

from backend.config import INTENT, BACKEND
from backend.data.dataloader import DataLoader
from backend.loss.center_loss import CenterLoss
from backend.model import FastText
from backend.model.ResNet import Model
from backend.proc.gensim_processor import GensimProcessor


def _draw_feature_space(feat, labels, label_dict):
    plt.ion()
    c = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff',
         '#ff00ff', '#990000', '#999900', '#009900', '#009999']
    plt.clf()
    for i in range(len(label_dict)):
        plt.plot(feat[labels == i, 0], feat[labels == i, 1], '.', c=c[i])
    plt.legend([i for i in range(len(label_dict))], loc='upper right')
    #   plt.xlim(xmin=-5,xmax=5)
    #   plt.ylim(ymin=-5,ymax=5)
    plt.savefig('{0}_feature_space.png'.format('test'))
    # plt.draw()
    # plt.pause(0.001)
    plt.close()


def test(test_data, model):
    correct, total = 0, 0
    test_data, test_label = test_data
    x = Variable(test_data).cuda()
    y = Variable(test_label).cuda()
    feature = model(x)

    retrieval = model.retrieval(feature)
    classification = model.classifier(retrieval)

    _, predicted = torch.max(classification.data, 1)
    total += y.size(0)
    correct += (predicted == y.data).sum()

    print("TEST ACC : {}".format((100 * correct / total)))


def train(train_data, model, criterion, opt, epoch, loss_weight, label_dict):
    vis_loader, idx_loader = [], []
    losses, acccs = 0, 0
    for i, (x, y) in enumerate(train_data):
        x = Variable(x.cuda())
        y = Variable(y.cuda())
        model = model.cuda()
        feature = model(x)

        retrieval = model.retrieval(feature).cuda()
        classification = model.classifier(retrieval).cuda()

        loss = criterion[0](classification, y)
        loss += loss_weight * criterion[1](retrieval, y)

        _, predicted = torch.max(classification.data, 1)
        accuracy = (y.data == predicted).float().mean()

        opt[0].zero_grad()
        opt[1].zero_grad()

        loss.backward()

        opt[0].step()
        opt[1].step()

        vis_loader.append(retrieval)
        idx_loader.append(y)
        losses += loss
        acccs += accuracy

    print("loss : {0}, acc : {1}".format(losses/len(train_data), acccs/len(train_data)))
    feat = torch.cat(vis_loader, 0)
    labels = torch.cat(idx_loader, 0)
    _draw_feature_space(feat.data.cpu().numpy(), labels.data.cpu().numpy(), label_dict)


def main():
    data_loader = DataLoader()
    embed = GensimProcessor(FastText)
    # embed.train(data_loader.embed_dataset())
    embed.load_model()

    dataset = data_loader.intent_dataset(embed)
    train_data, test_data = dataset

    model = Model(
        d_model=256,
        label_dict=data_loader.intent_dict,
        layers=10,
        vector_size=BACKEND['vector_size']
    )

    nll_loss = nn.CrossEntropyLoss()
    loss_weight = 1e-8
    center_loss = CenterLoss(d_model=256,
                             label_dict=data_loader.intent_dict)

    criterion = [nll_loss.cuda(), center_loss.cuda()]
    optimizer4nn = optim.Adam(model.parameters(), lr=0.01,weight_decay=0.0005)

    optimzer4center = optim.SGD(center_loss.parameters(), lr=0.5)

    for epoch in range(50):
        # print optimizer4nn.param_groups[0]['lr']
        train(train_data, model, criterion, [optimizer4nn, optimzer4center], 1500, loss_weight, data_loader.intent_dict)
        print(epoch)

if __name__ == '__main__':
    main()
