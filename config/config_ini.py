from configparser import ConfigParser


def reset_config():
    config_ini = ConfigParser()
    config_ini.add_section('config')  # 添加table section
    config_ini.set('config', 'data_base_path', 'F:/sciwork/NN_pro/fracture-master-1220-whole-process')  # 对config添加值
    config_ini.set('config', 'model_save_folder', './checkpoint/')  # 对config添加值
    config_ini.set('config', 'model_name', 'h2y_v4_1')  # 对config添加值
    config_ini.set('config', 'stages', 'six_stages')  # 对config添加值
    config_ini.set('config', 'runs_save_folder', './runs/events/')  # 对config添加值
    config_ini.set('config', 'best_h2y_model', 'fracture_v1_best_model_h2y.pth')  # 对config添加值
    config_ini.set('config', 'best_x2y_added_model', 'fracture_v1_best_model_x2y_added.pth')  # 对config添加值
    config_ini.set('config', 'best_x2y_model', 'fracture_v1_best_model_x2y.pth')  # 对config添加值
    config_ini.set('config', 'is_model_h2y', '1')  # 对config添加值
    config_ini.add_section('train')  # 添加table section
    config_ini.set('train', 'lr', '0.02')  # 对config添加值
    config_ini.set('train', 'batch_size', '64')  # 对config添加值
    config_ini.set('train', 'num_epochs', '60')  # 对config添加值
    config_ini.set('train', 'random_seed', '1234')  # 对config添加值
    config_ini.set('train', 'save_freq', '2')  # 对config添加值
    config_ini.set('train', 'is_noise', '0')  # 对config添加值
    config_ini.set('train', 'add_physical_info', '1')  # 对config添加值
    with open('config/config.ini', 'w', encoding='utf-8') as file:
        config_ini.write(file)  # 值写入配置文件


def clarify():
    config = ConfigParser()  # 改成x2y的配置
    # 对文件修改必须先将文件读取到config
    config.read('config/config.ini', encoding='UTF-8')
    config['config']['is_model_h2y'] = '0'  # 对于存在的值则是修改
    fo = open('config/config.ini', 'w', encoding='UTF-8')  # 重新创建配置文件
    config.write(fo)  # 数据写入配置文件
    fo.close()
