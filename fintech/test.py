import pandas as pd
from util.technical_indicators import create_indicators

def test(data_set):
    data_folder = 'data/'
    data_set_file_name = data_set + '.csv'
    data_set_path = data_folder + data_set_file_name

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

    print(train_df[0]['Close'].values[len(train_df[0]) - 2])

if __name__ == '__main__':
    test('DIA')
