import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from multiprocessing import Pool
from evaluation import evaluation
from openpyxl import Workbook


#algorithms = ['PPO2', 'A2C', 'ACKTR-PPO', 'ACKTR-A2C']
algorithms = ['PPO2']
policies = ['MlpPolicy']

process_pool = Pool(os.cpu_count(), maxtasksperchild=1)
wb = Workbook()

for algorithm in algorithms:
    for policy in policies:
        ws = wb.create_sheet(algorithm + '-' + policy)
        ws.append(['', 'train', 'd1', 'd2', 'train', 'd1', 'd2', 'train', 'd1', 'd2'])

        result = [process_pool.apply_async(evaluation, ('BIV', algorithm, policy))]

        for items in result:
            ws.append(items.get())

        wb.save('fintech.xlsx')

process_pool.close()
process_pool.join()
