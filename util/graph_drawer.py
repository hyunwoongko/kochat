"""
@author : Hyunwoong
@when : 2020-03-11
@homepage : https://github.com/gusdnd852
"""
from config import Config
import re
import matplotlib.pyplot as plt

conf = Config()
root_path = Config.root_path


class GraphDrawer:

    def read_file(self, file_name):
        f = open(file_name, 'r')
        file = f.read()
        file = re.sub('\\[', '', file)
        file = re.sub('\\]', '', file)
        f.close()

        return [float(i) for idx, i in enumerate(file.split(','))]

    def draw(self, mode, color):
        array = self.read_file(root_path + 'log/{}.txt'.format(mode))
        plt.plot(array, color[0], label='train_{}'.format(mode))
        plt.xlabel('epochs')
        plt.ylabel(mode)
        plt.title('train ' + mode)
        plt.grid(True, which='both', axis='both')
        plt.savefig(root_path + 'log/{}.png'.format(mode))
        plt.close()


if __name__ == '__main__':
    drawer = GraphDrawer()
    drawer.draw('accuracy', 'red')
    drawer.draw('error', 'blue')
