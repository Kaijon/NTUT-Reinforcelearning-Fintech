import gym
import os
import pandas as pd
import numpy as np

from stable_baselines.common.vec_env import  DummyVecEnv, SubprocVecEnv, VecFrameStack
from stable_baselines import PPO2, A2C, ACKTR

from env.FintechEnv import FinTechTrainEnv
from util.technical_indicators import create_indicators

def select_algorithm(algorithm, idx, model_folder, env):
    if algorithm == 'PPO2':
        model = PPO2.load(model_folder + str(idx), env)
    elif algorithm == 'A2C':
        model = A2C.load(model_folder + str(idx), env)
    elif algorithm == 'ACKTR-PPO' or algorithm == 'ACKTR-A2C':
        model = ACKTR.load(model_folder + str(idx), env)

    return model

def run_agent(env, model):
    obs = env.reset()
    done, assets, total_reword, _states = False, [], 0, None

    while not done:
        action, _states = model.predict(obs, _states)
        obs, reward, done, info = env.step(action)

        asset = info[0]['asset']
        current_price = info[0]['current_price']
        if reward[0] == -500:
            done = True

    rate_of_return = (asset - 10000) / 10000 * 100

    del model
    del env

    return rate_of_return

def run_test(df, algorithm, idx, model_folder):
    env = DummyVecEnv([lambda: FinTechTrainEnv(df, start_balance=10000, min_trading_unit=0, max_trading_count = 1000,max_change = 100, observation_length=int(3))])
    model = select_algorithm(algorithm, idx, model_folder, env)

    rate_of_return = run_agent(env, model)

    return rate_of_return

def run_evaluation(df, algorithm, model_folder):
    env = DummyVecEnv([lambda: FinTechTrainEnv(df, start_balance=10000, min_trading_unit=0, max_trading_count = 1000,max_change = 100, observation_length=int(3))])

    all_rate_of_return = []

    for idx in range(0, 100):
        if not os.path.exists(model_folder + str(idx) + '.zip'):
            break
        model = select_algorithm(algorithm, idx, model_folder, env)

        rate_of_return = run_agent(env, model)
        all_rate_of_return.append(rate_of_return)

    argmaxs = np.argsort(all_rate_of_return)[::-1][:10]

    return argmaxs

def compute_buy_and_hold(df, end_current_price=None):
    cost = 10000
    transaction_fees = 0.002
    current_price = df[0]['Close'].values[0 + 3]
    buy_amount = (1 - transaction_fees) * cost / current_price

    if end_current_price == None:
        current_price = df[0]['Close'].values[len(df[0])-2]
        sell_amount = buy_amount
        income = (1 - transaction_fees) * sell_amount * current_price
    else:
        current_price = end_current_price
        sell_amount = buy_amount
        income = (1 - transaction_fees) * sell_amount * current_price

    rate_of_return = (income - 10000) / 10000 * 100

    return rate_of_return

def evaluation(data_set, algorithm, policy):
    return_list = []
    return_list.append(data_set)
    data_folder = 'data/'
    data_set_file_name = data_set + '.csv'
    data_set_path = data_folder + data_set_file_name
    
    base_folder = '/' + algorithm + '/' + policy + '/' + data_set + '/'
    tensorboard_folder = './tensorboard' + base_folder
    model_folder = './model' + base_folder

    df_data_set = pd.read_csv(data_set_path)
    df_data_set = df_data_set.drop(['Adj Close'], axis=1)
    df_data_set['date'] = pd.to_datetime(df_data_set['Date'], format = '%Y-%m-%d')
    df_data_set['Date'] = df_data_set['date']
    df_data_set = df_data_set.drop(['date'], axis=1)
    df_data_set = df_data_set.sort_values(['Date'])

    # 切割訓練資料,評估資料,測試資料
    test_len = 252
    train_len = len(df_data_set) - test_len * 2

    train_df_data_set = df_data_set[:train_len]
    other_df_data_set = df_data_set[len(train_df_data_set):]
    d1_df_data_set = other_df_data_set[:test_len]
    d2_df_data_set = other_df_data_set[len(d1_df_data_set):]

    train_df_data_set = create_indicators(train_df_data_set.reset_index())
    d1_df_data_set = create_indicators(d1_df_data_set.reset_index())
    d2_df_data_set = create_indicators(d2_df_data_set.reset_index())

    train_df = [train_df_data_set]
    d1_df = [d1_df_data_set]
    d2_df = [d2_df_data_set]
    train_result = []
    d1_result = []
    d2_result = []

    best_model_idxs = run_evaluation(d1_df, algorithm, model_folder)
    for idx in best_model_idxs:
        for i in range(10):
            train_result.append(run_test(train_df, algorithm, idx, model_folder))
            d1_result.append(run_test(d1_df, algorithm, idx, model_folder))
            d2_result.append(run_test(d2_df, algorithm, idx, model_folder))
    return_list.append(np.mean(train_result, axis=0))
    return_list.append(np.mean(d1_result, axis=0))
    return_list.append(np.mean(d2_result, axis=0))

    train_result = []
    d1_result = []
    d2_result = []

    best_model_idxs = run_evaluation(d2_df, algorithm, model_folder)
    for idx in best_model_idxs:
        for i in range(10):
            train_result.append(run_test(train_df, algorithm, idx, model_folder))
            d1_result.append(run_test(d1_df, algorithm, idx, model_folder))
            d2_result.append(run_test(d2_df, algorithm, idx, model_folder))
    return_list.append(np.mean(train_result, axis=0))
    return_list.append(np.mean(d1_result, axis=0))
    return_list.append(np.mean(d2_result, axis=0))

    # return_list.append(compute_buy_and_hold(train_df))
    # return_list.append(compute_buy_and_hold(d1_df))
    # return_list.append(compute_buy_and_hold(d2_df))

    del train_df
    del d1_df
    del d2_df
    del train_df_data_set
    del d1_df_data_set
    del d2_df_data_set
    del df_data_set

    print('Finish: ' + algorithm + "-" + data_set)

    return return_list

if __name__ == '__main__':
    evaluation('DIA', 'PPO2', 'MlpPolicy')
