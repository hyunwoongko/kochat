"""
@auther Hyunwoong
@since {6/21/2020}
@see : https://github.com/gusdnd852
"""
from _app.restful_api import KochatApi

if __name__ == '__main__':
    api = KochatApi()
    api.run(ip='0.0.0.0', port=9893)
