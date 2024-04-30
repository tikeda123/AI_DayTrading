import pandas as pd
import sys, os

def setup_sys_path():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(parent_dir)
    return parent_dir

def configure_dependencies():
    from common.utils import configure_container
    from common.init_common_module import init_common_module
    configure_container()
    init_common_module()

def get_data_loader(container):
    return container.data_loader()

def create_filtered_data(df, direction):
    df_na = df.dropna(subset=['bb_profit'])
    return df_na[(df_na['bb_direction'] == direction) & (df_na['bb_profit'] != 0)]

def create_time_series_data(df, direction):
    filtered_df = create_filtered_data(df, direction)
    df['entry_volume'] = 0
    time_series_data = []

    for index in filtered_df.index:
        start_index = max(0, index - 7)
        end_index = index + 1
        df.loc[start_index:end_index, 'entry_price'] = df.iloc[end_index]['entry_price']
        df.loc[start_index:end_index, 'entry_volume'] = df.iloc[end_index]['volume']
        extracted_data = df.iloc[start_index:end_index]
        time_series_data.append(extracted_data)

    return pd.concat(time_series_data, ignore_index=True)

def generate_file_paths(config_manager, base_path, datafile):
    ml_datafile = os.path.join(base_path, datafile + "_ml.csv")
    ml_datafile_upper = os.path.join(base_path, datafile + "_upper_mlts.csv")
    ml_datafile_lower = os.path.join(base_path, datafile + "_lower_mlts.csv")
    return ml_datafile, ml_datafile_upper, ml_datafile_lower

def main():
    parent_dir = setup_sys_path()
    configure_dependencies()
    from common.container import Container

    container = Container()
    data_loader = get_data_loader(container)
    config_manager = Container.config_manager()

    data_ml_path = os.path.join(parent_dir, config_manager.get('AIML', 'DATAPATH'))
    datafile = data_loader.get_tstfile().split('.')[0]

    ml_datafile, ml_datafile_upper, ml_datafile_lower = generate_file_paths(config_manager, data_ml_path, datafile)

    data_loader.load_data_from_csv(ml_datafile)
    df = data_loader.get_raw()

    upper_data = create_time_series_data(df, 'upper')
    lower_data = create_time_series_data(df, 'lower')

    upper_data.to_csv(ml_datafile_upper)
    lower_data.to_csv(ml_datafile_lower)

    print("Upper Data:\n", upper_data)
    print("\nLower Data:\n", lower_data)

if __name__ == "__main__":
    main()
