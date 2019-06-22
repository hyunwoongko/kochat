import re
import urllib.request

from bs4 import BeautifulSoup


def get_exchange(target):
    fp = urllib.request.urlopen('https://finance.naver.com/marketindex/?tabSel=exchange#tab_section')
    source = fp.read()
    fp.close()
    soup = BeautifulSoup(source, 'html.parser')
    soup = soup.find_all("option")
    flag = True
    exchange_list = []
    for i in soup:
        if '폴란드' in i:
            flag = False
        if flag:
            exchange_format = {}
            country = i.text
            country = re.sub('A', '', country)
            country = re.sub('B', '', country)
            country = re.sub('C', '', country)
            country = re.sub('D', '', country)
            country = re.sub('E', '', country)
            country = re.sub('F', '', country)
            country = re.sub('G', '', country)
            country = re.sub('H', '', country)
            country = re.sub('I', '', country)
            country = re.sub('J', '', country)
            country = re.sub('K', '', country)
            country = re.sub('L', '', country)
            country = re.sub('N', '', country)
            country = re.sub('M', '', country)
            country = re.sub('O', '', country)
            country = re.sub('P', '', country)
            country = re.sub('Q', '', country)
            country = re.sub('R', '', country)
            country = re.sub('S', '', country)
            country = re.sub('T', '', country)
            country = re.sub('U', '', country)
            country = re.sub('V', '', country)
            country = re.sub('W', '', country)
            country = re.sub('X', '', country)
            country = re.sub('Y', '', country)
            country = re.sub('Z', '', country)
            country = country.lstrip()
            country = country.rstrip()
            exchange_format['country'] = country
            exchange_format['won'] = i.get('label')
            exchange_format['value'] = i.get('value')
            exchange_list.append(exchange_format)

    for i in exchange_list:
        if target in i['country']:
            if int(i['won']) == 1:
                msg = '현재 ' + i['country'] + '의 가치는 ' + i['won'] + i['country'].split()[
                    1] + ' 당 ' + str(float(i['value'])) + ' ' + '원 입니다'
                return msg
            else:
                msg = '현재 ' + i['country'] + '의 가치는 ' + str(float(i['value'])) + i['country'].split()[
                    1] + ' 당 ' + i['won'] + ' ' + '원 입니다'
                return msg

    for i in exchange_list:
        if target not in i['country']:
            return '죄송합니다. 해당 국가의 환율은 알 수가 없습니다.'
