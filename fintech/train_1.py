import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from multiprocessing import Pool
from evaluation import evaluation
from openpyxl import Workbook
from multiprocessing import Pool
from train import train

#algorithms = ['PPO2', 'A2C', 'ACKTR-PPO', 'ACKTR-A2C']
policies = ['MlpPolicy']

algorithms = ['PPO2']
# policies = ['MlpLnLstmPolicy']

process_pool = Pool(processes=6, maxtasksperchild=1)

for algorithm in algorithms:
    for policy in policies:
            process_pool.apply_async(train, ('BIV', algorithm, policy))

process_pool.close()
process_pool.join()
