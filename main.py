import pandas as pd
from utils.logger import logging
from utils.exception import MyException
from utils.config_loader import settings


file_path = settings['data']['path']

df  = pd.read_csv(file_path)

print(df.sample(5))
print(df.shape)