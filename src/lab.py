'''
Author: hibana2077 hibana2077@gmaill.com
Date: 2024-04-17 15:56:39
LastEditors: hibana2077 hibana2077@gmail.com
LastEditTime: 2024-04-20 11:27:19
FilePath: /2024_president/ml/lab.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import yaml
import pprint

file_name = "./train_setting.yaml"

train_setting = yaml.safe_load(open("train_setting.yaml"))

pprint.pprint(train_setting)