import copy
import yaml
import torch
import data
from torch import nn
from torchvision.models import resnet34

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def icarl_construct_exemplar_set(class_set, m, model):
    pass


def icarl_reduce_examplar_set(m, exemplars_set):
    pass


def icarl_update_representation(model, dataset, exemplars_set):
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=0.01)
    classification_loss = torch.nn.CrossEntropyLoss()

    combined_set = dataset + exemplars_set if exemplars_set else dataset
    combined_loader = data.utils.get_dataloader(exemplars_set)

    exemplars_labels = set(exemplars_set.targets) if exemplars_set else {}

    old_model = copy.copy(model) 

    for epoch in range(70):
        for batch, (imgs, labels) in enumerate(combined_loader):
            imgs = imgs.to(device)
            onehot_labels = data.utils.onehot_encoder(labels, n_classes=100)
            exemplars_idxs = [idx for idx, label in enumerate(labels) if label.item() in exemplars_labels]
            previous_model_output = old_model(imgs[exemplars_idxs])
            breakpoint()
            current_model_output = model(imgs)
            
    data_loader = data.utils.get_dataloader(combined_set)
    pass


def icarl_incremental_train(model, dataset, k, exemplars_set):
    model = icarl_update_representation(model, dataset, exemplars_set)
    #data_loader = data.utils.get_dataloader(dataset)
    m = k//len(exemplars_set)
    pass


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
        icarl_incremental_train(model, train_set, 2000, train_set)


def load_config(file):
    with open(file, 'r') as reader:
        config = yaml.load(reader, Loader=yaml.SafeLoader)

    return config


if __name__ == '__main__':
    config = load_config('./config.yaml')
    main(config)
