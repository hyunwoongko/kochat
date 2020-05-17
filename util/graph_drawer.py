"""
@author : Hyunwoong
@when : 2020-03-11
@homepage : https://github.com/gusdnd852
"""
from configs import GlobalConfigs
import re
import matplotlib.pyplot as plt

conf = GlobalConfigs()
root_path = GlobalConfigs.root_path
record_step = GlobalConfigs.record_step


class GraphDrawer:

    def read_file(self, file_name):
        f = open(file_name, 'r')
        file = f.read()
        file = re.sub('\\[', '', file)
        file = re.sub('\\]', '', file)
        f.close()

        return [float(i) for idx, i in enumerate(file.split(','))]

    def draw_accuracy(self):
        train = self.read_file(root_path + 'log\\train_accuracy.txt')
        test = self.read_file(root_path + 'log\\test_accuracy.txt')
        plt.plot(train, 'b', label='train acc')
        plt.plot(test, 'r', label='test acc')
        plt.xlabel('step')
        plt.ylabel('acc')
        plt.title('training result')
        plt.legend(loc='lower left')
        plt.grid(True, which='both', axis='both')
        plt.show()

    def draw_error(self):
        train = self.read_file(root_path + 'log\\train_error.txt')
        test = self.read_file(root_path + 'log\\test_error.txt')
        plt.plot(train, 'y', label='train error')
        plt.plot(test, 'g', label='test error')
        plt.xlabel('step')
        plt.ylabel('error')
        plt.title('training result')
        plt.legend(loc='lower left')
        plt.grid(True, which='both', axis='both')
        plt.show()

    def draw_both(self):
        self.draw_error()
        self.draw_accuracy()


if __name__ == '__main__':
    drawer = GraphDrawer()
    drawer.draw_both()
