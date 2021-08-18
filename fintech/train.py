import gym
import pandas as pd
import os
import gc

from stable_baselines.common.vec_env import  DummyVecEnv, SubprocVecEnv, VecFrameStack
from stable_baselines import PPO2, A2C, ACKTR

from env.FintechEnv import FinTechTrainEnv
from util.technical_indicators import create_indicators

def train(data_set, algorithm, policy):
    gc.set_threshold(100, 5, 5)
    data_folder = 'data/'
    data_set_file_name = data_set + '.csv'
    data_set_path = data_folder + data_set_file_name
    
    base_folder = '/' + algorithm + '/' + policy + '/' + data_set + '/'
    tensorboard_folder = './tensorboard' + base_folder
    model_folder = './model' + base_folder
    debug_folder = './debug' + base_folder

    # 產生必要資料夾
    if not os.path.isdir(tensorboard_folder):
        os.makedirs(tensorboard_folder)
    if not os.path.isdir(model_folder):
        os.makedirs(model_folder)
    if not os.path.isdir(debug_folder):
        os.makedirs(debug_folder)

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

    # 添加技術指標
    train_df_data_set = create_indicators(train_df_data_set.reset_index())

    train_df = [train_df_data_set]

    # RL環境
    train_env = DummyVecEnv([lambda: FinTechTrainEnv(train_df, start_balance=10000, min_trading_unit=0, max_trading_count=1000,max_change=100, observation_length=int(3))])

    if algorithm == 'PPO2':
        model = PPO2(policy, train_env, verbose=0, nminibatches=1, tensorboard_log=tensorboard_folder, full_tensorboard_log=False)
    elif algorithm == 'A2C':
        model = A2C(policy, train_env, verbose=0, tensorboard_log=tensorboard_folder, full_tensorboard_log=False)
    elif algorithm == 'ACKTR-PPO':
        model = ACKTR(policy, train_env, verbose=0, gae_lambda=0.95, tensorboard_log=tensorboard_folder, full_tensorboard_log=False)
    elif algorithm == 'ACKTR-A2C':
        model = ACKTR(policy, train_env, verbose=0, tensorboard_log=tensorboard_folder, full_tensorboard_log=False)

    for idx in range(0, 100):
        model.learn(total_timesteps=len(train_df_data_set)*10)
        model.save(model_folder + str(idx))
        gc.collect()

    del model
    del train_env
    del train_df
    del train_df_data_set
    del df_data_set

    print('Finish: ' + algorithm + "-" + data_set)

if __name__ == '__main__':
    train('DIA', 'PPO2', 'MlpPolicy')