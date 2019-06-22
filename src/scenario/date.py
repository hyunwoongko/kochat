# Author : Hyunwoong
# When : 2019-06-22
# Homepage : github.com/gusdnd852
import time


def date():
    Y = time.strftime('%Y')
    M = time.strftime('%m')
    D = time.strftime('%d')
    return "오늘은 " + Y + "년 " + M + "월 " + D + "일입니다."
