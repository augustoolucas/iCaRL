import copy
import yaml
import torch
import data
import numpy as np
from tqdm import tqdm
from torch import nn
from torchvision.models import resnet34

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def icarl_construct_exemplar_set(class_set, m, model):
    pass


def icarl_reduce_examplar_set(m, exemplars_set):
    ex_set = copy.copy(exemplars_set)
    data_shape = ex_set.data.shape[1:]
    data_shape = (0,) + data_shape
    ex_set.data = np.ndarray(data_shape)
    ex_set.targets = []
    for cls in set(exemplars_set.targets):
        idxs = np.nonzero(np.array(exemplars_set.targets) == cls)[0].tolist()
        ex_set.data = np.concatenate((ex_set.data,
                                     exemplars_set.data[idxs][:m]), axis=0)
        ex_set.targets.extend(np.array(exemplars_set.targets)[idxs][:m].tolist())

    return ex_set


def icarl_update_representation(model, dataset, exemplars_set):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    cls_loss_fn = torch.nn.CrossEntropyLoss()
    disti_loss_fn = torch.nn.CrossEntropyLoss()

    combined_set = dataset + exemplars_set if exemplars_set else dataset
    combined_loader = data.utils.get_dataloader(combined_set)

    exemplars_labels = set(exemplars_set.targets) if exemplars_set else {}

    old_model = copy.copy(model) 
    old_model = old_model.to(device)

    train_bar = tqdm(range(10))
    model.train(); old_model.eval()
    for epoch in train_bar:
        for batch, (imgs, labels) in enumerate(combined_loader):
            model.zero_grad()
            imgs, labels = imgs.to(device), labels.to(device)
            #onehot_labels = data.utils.onehot_encoder(labels, n_classes=100)
            exemplars_idxs = [idx for idx, label in enumerate(labels) if label.item() in exemplars_labels]
            with torch.no_grad():
                previous_model_output = old_model(imgs[exemplars_idxs])
            current_model_output = model(imgs)
            
            cls_loss_value = cls_loss_fn(current_model_output, labels)
            dist_loss_value = disti_loss_fn(previous_model_output,
                                            current_model_output[exemplars_idxs])
            total_loss_value = cls_loss_value + dist_loss_value
            total_loss_value.backward()
            optimizer.step()
            

def icarl_incremental_train(model, dataset, k, exemplars_set):
    model = icarl_update_representation(model, dataset, exemplars_set)
    #data_loader = data.utils.get_dataloader(dataset)
    if exemplars_set:
        m = k//len(set(exemplars_set.targets))
        exemplars_set = icarl_reduce_examplar_set(m, exemplars_set)

    return 


def icarl_classify(img, class_exemplars):
    pass


def main(config):
    ### ------ Load Data ------ ###
    train_tasks = data.utils.load_tasks(config['dataset'],
                                        config['n_tasks'],
                                        train=True)

    test_tasks = data.utils.load_tasks(config['dataset'],
                                       config['n_tasks'],
                                       train=False)
    
    model = resnet34(pretrained=True, progress=True).to(device)
    model.fc = nn.Linear(512, 100)

    exemplars_set = None
    for task in range(config['n_tasks']):
        train_set = copy.copy(train_tasks[task])
        icarl_incremental_train(model, train_set, 500, exemplars_set)


def load_config(file):
    with open(file, 'r') as reader:
        config = yaml.load(reader, Loader=yaml.SafeLoader)

    return config


if __name__ == '__main__':
    config = load_config('./config.yaml')
    main(config)
