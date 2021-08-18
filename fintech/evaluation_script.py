import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from multiprocessing import Pool
from evaluation import evaluation
from openpyxl import Workbook


algorithms = ['PPO2', 'A2C', 'ACKTR-PPO', 'ACKTR-A2C']
policies = ['MlpPolicy']

process_pool = Pool(processes=6, maxtasksperchild=2)
wb = Workbook()

for algorithm in algorithms:
    for policy in policies:
        ws = wb.create_sheet(algorithm + '-' + policy)
        ws.append(['', 'train', 'd1', 'd2', 'train', 'd1', 'd2', 'train', 'd1', 'd2'])

        result = [process_pool.apply_async(evaluation, ('BIV', algorithm, policy)),
                    process_pool.apply_async(evaluation, ('DBO', algorithm, policy)),
                    process_pool.apply_async(evaluation, ('DIA', algorithm, policy)),
                    process_pool.apply_async(evaluation, ('DON', algorithm, policy)),
                    process_pool.apply_async(evaluation, ('DVY', algorithm, policy)),
                    process_pool.apply_async(evaluation, ('EDV', algorithm, policy)),
                    process_pool.apply_async(evaluation, ('EFAV', algorithm, policy)),
                    process_pool.apply_async(evaluation, ('VV', algorithm, policy)),
                    process_pool.apply_async(evaluation, ('XLB', algorithm, policy)),
                    process_pool.apply_async(evaluation, ('XRT', algorithm, policy)),
                    process_pool.apply_async(evaluation, ('VTI', algorithm, policy)),
                    process_pool.apply_async(evaluation, ('VEU', algorithm, policy)),
                    process_pool.apply_async(evaluation, ('QQQ', algorithm, policy)),
                    process_pool.apply_async(evaluation, ('SPY', algorithm, policy)),
                    process_pool.apply_async(evaluation, ('DBA', algorithm, policy)),
                    process_pool.apply_async(evaluation, ('DBC', algorithm, policy)),
                    process_pool.apply_async(evaluation, ('AGG', algorithm, policy)),
                    process_pool.apply_async(evaluation, ('GLD', algorithm, policy)),
                    process_pool.apply_async(evaluation, ('VNQ', algorithm, policy)),
                    process_pool.apply_async(evaluation, ('VNQI', algorithm, policy))]

        for items in result:
            ws.append(items.get())

        wb.save('fintech.xlsx')

process_pool.close()
process_pool.join()
