import yaml
import torch
import data


def main(config):
    ### ------ Load Data ------ ###
    train_tasks = data.utils.load_tasks(config['dataset'],
                                        config['n_tasks'],
                                        train=True)

    test_tasks = data.utils.load_tasks(config['dataset'],
                                       config['n_tasks'],
                                       train=False)


def load_config(file):
    with open(file, 'r') as reader:
        config = yaml.load(reader, Loader=yaml.SafeLoader)

    return config


if __name__ == '__main__':
    config = load_config('./config.yaml')
    main(config)
