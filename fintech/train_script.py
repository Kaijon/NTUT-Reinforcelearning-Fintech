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
            process_pool.apply_async(train, ('VNQ', algorithm, policy))
            process_pool.apply_async(train, ('DBO', algorithm, policy))
            process_pool.apply_async(train, ('DIA', algorithm, policy))
            process_pool.apply_async(train, ('DON', algorithm, policy))
            process_pool.apply_async(train, ('DVY', algorithm, policy))
            process_pool.apply_async(train, ('EDV', algorithm, policy))
            process_pool.apply_async(train, ('EFAV', algorithm, policy))
            process_pool.apply_async(train, ('VV', algorithm, policy))
            process_pool.apply_async(train, ('XLB', algorithm, policy))
            process_pool.apply_async(train, ('XRT', algorithm, policy))

            process_pool.apply_async(train, ('VTI', algorithm, policy))
            process_pool.apply_async(train, ('VEU', algorithm, policy))
            process_pool.apply_async(train, ('QQQ', algorithm, policy))
            process_pool.apply_async(train, ('SPY', algorithm, policy))
            process_pool.apply_async(train, ('DBA', algorithm, policy))
            process_pool.apply_async(train, ('DBC', algorithm, policy))
            process_pool.apply_async(train, ('AGG', algorithm, policy))
            process_pool.apply_async(train, ('GLD', algorithm, policy))
            process_pool.apply_async(train, ('VNQ', algorithm, policy))
            process_pool.apply_async(train, ('VNQI', algorithm, policy))

process_pool.close()
process_pool.join()
