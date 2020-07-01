"""
@auther Hyunwoong
@since {6/23/2020}
@see : https://github.com/gusdnd852
"""
import itertools
import os
import re

import six
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.decomposition import IncrementalPCA
from torch import Tensor

from kochat.decorators import proc


@proc
class Visualizer:

    def __init__(self, model_dir: str, model_file: str):
        """
        학습, 검증 결과를 저장하고 시각화하는 클래스입니다.
        """

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        if not os.path.exists(model_dir + 'temp'):
            os.makedirs(model_dir + 'temp')

        self.model_dir = model_dir
        self.model_file = model_file
        self.train_loss, self.test_loss = [], []
        self.train_accuracy, self.test_accuracy = [], []
        self.train_precision, self.test_precision = [], []
        self.train_recall, self.test_recall = [], []
        self.train_f1_score, self.test_f1_score = [], []

    def save_result(self, loss: Tensor, eval_dict: dict, mode: str):
        """
        training / test 결과를 저장합니다.

        :param loss: loss 리스트
        :param eval_dict: 다양한 메트릭으로 평가한 결과가 저장된 딕셔너리
        :param mode: train or test
        """
        if mode == 'train':
            self.train_loss.append(loss.item())
            self.train_accuracy.append(eval_dict['accuracy'].item())
            self.train_precision.append(eval_dict['precision'].item())
            self.train_recall.append(eval_dict['recall'].item())
            self.train_f1_score.append(eval_dict['f1_score'].item())
            self.__save_txt(self.train_accuracy, 'train_accuracy')
            self.__save_txt(self.train_precision, 'train_precision')
            self.__save_txt(self.train_recall, 'train_recall')
            self.__save_txt(self.train_f1_score, 'train_f1_score')
            self.__save_txt(self.train_loss, 'train_loss')

        elif mode == 'test':
            self.test_loss.append(loss.item())
            self.test_accuracy.append(eval_dict['accuracy'].item())
            self.test_precision.append(eval_dict['precision'].item())
            self.test_recall.append(eval_dict['recall'].item())
            self.test_f1_score.append(eval_dict['f1_score'].item())
            self.__save_txt(self.test_accuracy, 'test_accuracy')
            self.__save_txt(self.test_precision, 'test_precision')
            self.__save_txt(self.test_recall, 'test_recall')
            self.__save_txt(self.test_f1_score, 'test_f1_score')
            self.__save_txt(self.test_loss, 'test_loss')

        else:
            raise Exception('mode는 train과 test만 가능합니다.')

    def draw_graphs(self):
        """
        다양한 메트릭 그래프를 그립니다.
        test가 True인 경우 testing 결과도 함께 그립니다.
        """

        plt.plot(self.__load_txt('train_accuracy'), 'darkgreen', label='train_accuracy')
        if len(self.test_accuracy) != 0:
            plt.plot(self.__load_txt('test_accuracy'), 'limegreen', label='test_accuracy')
        self.__draw_graph('accuracy')

        plt.plot(self.__load_txt('train_precision'), 'darkcyan', label='train_precision')
        if len(self.test_precision) != 0:
            plt.plot(self.__load_txt('test_precision'), 'cyan', label='test_precision')
        self.__draw_graph('precision')

        plt.plot(self.__load_txt('train_recall'), 'darkred', label='train_recall')
        if len(self.test_recall) != 0:
            plt.plot(self.__load_txt('test_recall'), 'red', label='test_recall')
        self.__draw_graph('recall')

        plt.plot(self.__load_txt('train_f1_score'), 'darkmagenta', label='train_f1_score')
        if len(self.test_f1_score) != 0:
            plt.plot(self.__load_txt('test_f1_score'), 'magenta', label='test_f1_score')
        self.__draw_graph('f1_score')

        plt.plot(self.__load_txt('train_loss'), 'darkgoldenrod', label='train_loss')
        if len(self.test_loss) != 0:
            plt.plot(self.__load_txt('test_loss'), 'gold', label='test_loss')
        self.__draw_graph('loss')

    def draw_matrix(self, cm: np.ndarray, target_names: list, mode: str):
        """
        metrics에서 출력된 confusion matrix을 시각화해서 그리고 저장합니다.

        :param cm: confusion matrix 객체
        :param target_names: 각 클래스들의 이름
        :param mode: train or test 모드
        """

        label_length = len(target_names)
        figure_base_size = (label_length * 1.5)
        title_font_size = (label_length * 3) + 7

        cmap = plt.get_cmap('Blues')
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        plt.figure(figsize=(figure_base_size + 7, figure_base_size + 4))
        im = plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(
            label='{} confusion matrix'.format(mode),
            fontdict={'fontsize': title_font_size},
            pad=35
        )

        tick_marks = np.arange(label_length)
        plt.xticks(tick_marks, target_names)
        plt.yticks(tick_marks, target_names)
        plt.colorbar(im, fraction=0.046, pad=0.04)
        thresh = cm.max() / 1.5

        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.savefig(self.model_file + '_confusion_matrix_{}.png'.format(mode))
        plt.close()

    def draw_report(self, report: DataFrame, mode: str):
        """
        metrics에서 출력된 classification report를 시각화해서 그리고 저장합니다.

        :param report: report 데이터 프래임
        :param mode: train or test 모드
        """
        row_colors = ['#f0f0f0', 'w']
        col_width, row_height, header_columns = 3.0, 0.625, 0

        size = (np.array(report.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')

        table = ax.table(cellText=report.values,
                         bbox=[0, 0, 1, 1],
                         colLabels=report.columns,
                         rowLabels=report.index)

        table.auto_set_font_size(False)
        table.set_fontsize(12)

        for k, cell in six.iteritems(table._cells):
            cell.set_edgecolor('white')
            if k[0] == 0 or k[1] < header_columns:
                cell.set_text_props(weight='bold', color='w')
                cell.set_facecolor('#09539d')
            else:
                cell.set_facecolor(row_colors[k[0] % len(row_colors)])

        fig = ax.get_figure()
        fig.savefig(self.model_file + '_report_{}.png'.format(mode))
        plt.close()

    def draw_feature_space(self, feats: Tensor, labels: Tensor, label_dict: dict, loss_name: str,
                           d_loss: int, epoch: int, mode: str):
        """
        모든 데이터의 샘플들의 분포를 시각화합니다.

        :param feats: 모델의 출력 features
        :param labels: 라벨 리스트
        :param label_dict: 라벨 딕셔너리
        :param loss_name: loss 함수의 이름
        :param d_loss: loss 시각화 차원
        :param epoch: 현재 진행된 epochs
        :param mode: train or test
        """

        if not isinstance(labels, np.ndarray):
            labels = labels.detach().cpu().numpy()
        if not isinstance(feats, np.ndarray):
            feats = feats.detach().cpu().numpy()

        if d_loss == 2:
            self.__2d_feature_space(feats, labels, label_dict)
        else:
            self.__3d_feature_space(feats, labels, label_dict, d_loss)

        if not os.path.exists(self.model_dir + 'feature_space'):
            os.makedirs(self.model_dir + 'feature_space')

        plt.legend(loc='upper right')
        plt.savefig(self.model_dir +
                    'feature_space/{loss_name}_{d_loss}D_{mode}_{epoch}.png'
                    .format(loss_name=loss_name, d_loss=d_loss, mode=mode, epoch=epoch))

        plt.close()

    def __draw_graph(self, mode: str):
        """
        plot된 정보들을 바탕으로 그래프를 그리고 저장합니다.

        :param mode: train or test
        """

        plt.xlabel('epochs')
        plt.ylabel(mode)
        plt.title('train test {}'.format(mode))
        plt.grid(True, which='both', axis='both')
        plt.legend()
        plt.savefig(self.model_file + '_graph_{}.png'.format(mode))
        plt.close()

    def __2d_feature_space(self, feats: np.ndarray, labels: np.ndarray, label_dict: dict):
        """
        d_loss가 2차원인 경우 2D로 시각화 합니다.

        :param feats: 모델의 출력 features
        :param labels: 라벨 리스트
        :param label_dict: 라벨 딕셔너리
        """

        data = pd.DataFrame(np.c_[feats, labels], columns=['x', 'y', 'label'])
        ax = plt.figure().add_subplot()
        for group in data.groupby('label'):
            group_index, group_table = group
            ax.scatter(group_table['x'], group_table['y'],
                       marker='o',
                       label=list(label_dict)[int(group_index)])

    def __3d_feature_space(self, feats: np.ndarray, labels: np.ndarray, label_dict: dict, d_loss: int):
        """
        d_loss가 3차원 이상인 경우 3D로 시각화 합니다.

        :param feats: 모델의 출력 features
        :param labels: 라벨 리스트
        :param label_dict: 라벨 딕셔너리
        :param d_loss: loss 시각화 차원
        """

        if d_loss != 3:
            # 3차원 이상이면 PCA 수행해서 3차원으로 만듬
            pca = IncrementalPCA(n_components=3)
            split_size = (feats.shape[0] // self.batch_size) + 1
            for batch_x in np.array_split(feats, split_size):
                pca.partial_fit(batch_x)
            feats = pca.transform(feats)

        ax = plt.figure().gca(projection='3d')
        data = pd.DataFrame(np.c_[feats, labels], columns=['x', 'y', 'z', 'label'])
        for group in data.groupby('label'):
            group_index, group_table = group
            ax.scatter(group_table['x'], group_table['y'], group_table['z'],
                       marker='o',
                       label=list(label_dict)[int(group_index)])

    def __load_txt(self, mode: str):
        """
        저장된 파일을 로드하여 배열로 반환합니다.

        :param mode: train or test
        """

        f = open(self.model_dir + 'temp{_}{mode}.txt'.format(_=self.delimeter, mode=mode), 'r')
        file = f.read()
        file = re.sub('\\[', '', file)
        file = re.sub('\\]', '', file)
        f.close()

        return [float(i) for idx, i in enumerate(file.split(','))]

    def __save_txt(self, array: list, mode: str):
        """
        배열을 입력받아서 string 변환해 txt파일로 저장합니다.

        :param array: 저장할 배열
        :param mode: train or test
        """

        f = open(self.model_dir + 'temp{_}{mode}.txt'.format(_=self.delimeter, mode=mode), 'w')
        f.write(str(array))
        f.close()
