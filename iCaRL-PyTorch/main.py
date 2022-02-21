import copy
import yaml
import torch
import data
from models import networks
import numpy as np
from tqdm import tqdm
from torch import nn
from torchvision.models import resnet34
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_dataset_features(model, dataset):
    model.eval(); model.zero_grad()
    dataloader = data.utils.get_dataloader(dataset)
    features = torch.tensor(np.ndarray((0, 100)))

    with torch.no_grad():
        for imgs, labels in dataloader:
            model_output = model(imgs.to(device))
            features = torch.cat((features, model_output.to('cpu')))

    return features

def icarl_construct_exemplar_set(model, dataset, m):
    model.eval(); model.zero_grad()

    data_shape = dataset.data.shape[1:]
    data_shape = (0,) + data_shape
    exemplars_set = copy.copy(dataset)
    exemplars_set.data = torch.tensor(np.ndarray(data_shape)).type(torch.uint8)
    exemplars_set.targets = []

    img_shape = (0, 32, 32, 3)

    for cls_idx, cls in enumerate(set(dataset.targets)):
        idxs = np.nonzero(np.array(dataset.targets) == cls)[0].tolist()
        aux_set = copy.copy(dataset)
        aux_set.data = dataset.data[idxs]
        aux_set.targets = np.array(dataset.targets)[idxs].tolist()

        features = get_dataset_features(model, aux_set)
        class_mean = torch.mean(features, dim=0)

        class_set = copy.copy(dataset)
        class_set.data = np.ndarray(img_shape, dtype=dataset.data.dtype)
        class_set.targets = []

        aux_features = torch.tensor(np.ndarray((0, 100)))
        for k in range(m):
            if k < 1:
                imgs = torch.tensor(np.ndarray((0, 3, 32, 32)))
            else:
                imgs = torch.stack([d[0] for idx, d in enumerate(class_set) if idx < k])
            with torch.no_grad():
                model_output = model(imgs.to(device))
            aux_features = torch.cat((aux_features, model_output.to('cpu')))

            arg = class_mean - (features + torch.sum(aux_features, dim=0))/k
            pk_idx = torch.argmin(torch.sum(arg, dim=1)).item()
            pk_idx = idxs[pk_idx]
            class_set.data = np.concatenate((class_set.data,
                                             np.expand_dims(dataset.data[pk_idx], 0)))

            class_set.targets.append(dataset[pk_idx][1])

        #class_set.data = class_set.data.swapaxes(1, 2).swapaxes(2, 3)
        class_set.data = (255*class_set.data).astype(np.uint8)
        exemplars_set.data = np.concatenate((exemplars_set.data,
                                             class_set.data))
        exemplars_set.targets.extend(class_set.targets)

    return exemplars_set


def icarl_reduce_examplar_set(m, exemplars_set):
    ex_set = copy.copy(exemplars_set)
    data_shape = ex_set.data.shape[1:]
    data_shape = (0,) + data_shape
    ex_set.data = np.ndarray(data_shape, dtype=exemplars_set.data.dtype)
    ex_set.targets = []
    for cls in set(exemplars_set.targets):
        idxs = np.nonzero(np.array(exemplars_set.targets) == cls)[0].tolist()
        ex_set.data = np.concatenate((ex_set.data,
                                     exemplars_set.data[idxs][:m]), axis=0)
        ex_set.targets.extend(np.array(exemplars_set.targets)[idxs][:m].tolist())

    return ex_set


def icarl_update_representation(model, dataset, exemplars_set):
    old_model = copy.deepcopy(model)

    old_model = old_model.to(device)
    model = model.to(device)
    model.train(); old_model.eval()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.00001)
    cls_loss_fn = torch.nn.CrossEntropyLoss()
    disti_loss_fn = torch.nn.CrossEntropyLoss()

    combined_set = dataset + exemplars_set if exemplars_set else dataset
    combined_loader = data.utils.get_dataloader(combined_set)

    exemplars_labels = set(exemplars_set.targets) if exemplars_set else {}

    for epoch in tqdm(range(70)):
        for batch, (imgs, labels) in enumerate(combined_loader):
            model.zero_grad()
            imgs, labels = imgs.to(device), labels.to(device)
            #onehot_labels = data.utils.onehot_encoder(labels, n_classes=100)
            exemplars_idxs = [idx for idx, label in enumerate(labels) if label.item() in exemplars_labels]
            current_idxs = list(set(range(len(labels))) - set(exemplars_idxs))

            with torch.no_grad():
                previous_model_output = old_model(imgs[exemplars_idxs])
            current_model_output = model(imgs)
            #if exemplars_set and len(exemplars_idxs) == 0 : breakpoint()
            
            cls_loss_value = cls_loss_fn(current_model_output[current_idxs], labels[current_idxs])/len(labels)
            dist_loss_value = disti_loss_fn(current_model_output[exemplars_idxs], previous_model_output)/len(labels)
            dist_loss_value = dist_loss_value if len(exemplars_idxs) > 0 else 0
            total_loss_value = cls_loss_value + dist_loss_value
            total_loss_value.backward()
            optimizer.step()

    task_acc = 0
    model.eval()
    for batch, (imgs, labels) in enumerate(combined_loader):
        imgs, labels = imgs.to(device), labels.to(device)
        current_model_output = model(imgs)
        prediction = np.argmax(current_model_output.cpu().detach().numpy(), axis=1)
        task_acc += accuracy_score(prediction, labels.tolist())

    task_acc /= len(combined_loader)
    print(f'task acc {task_acc}')

    return model


def train(model, dataset):
    model.train()
    model = model.to(device)
    dataloader = data.utils.get_dataloader(dataset)
    optimizer_classifier = torch.optim.Adam(model.parameters(),
                                            lr=0.001)
    cls_loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in tqdm(range(20)):
        for batch, (imgs, labels) in enumerate(dataloader):
            model.zero_grad()
            imgs, labels = imgs.to(device), labels.to(device)
            model_output = model(imgs)
            cls_loss_value = cls_loss_fn(model_output, labels)/len(labels)
            cls_loss_value.backward()
            optimizer_classifier.step()

    model.eval()
    task_acc = 0
    for batch, (imgs, labels) in enumerate(dataloader):
        imgs, labels = imgs.to(device), labels.to(device)
        with torch.no_grad():
            model_output = model(imgs)
        prediction = torch.argmax(model_output, dim=1).tolist()
        task_acc += accuracy_score(prediction, labels.tolist())

    task_acc /= len(dataloader)

    breakpoint()


def icarl_incremental_train(model, dataset, k, exemplars_set):
    model = icarl_update_representation(model, dataset, exemplars_set)

    if exemplars_set:
        m = k//(len(set(exemplars_set.targets)) + len(set(dataset.targets)))
        exemplars_set = icarl_reduce_examplar_set(m, exemplars_set)

        new_exemplars_set = icarl_construct_exemplar_set(model,
                                                         dataset,
                                                         m)

        new_exemplars_set.data = np.concatenate([exemplars_set.data,
                                                 new_exemplars_set.data])
        new_exemplars_set.targets = exemplars_set.targets + new_exemplars_set.targets
    else:
        m = k//len(set(dataset.targets))
        new_exemplars_set = icarl_construct_exemplar_set(model,
                                                         dataset,
                                                         m)

    return model, new_exemplars_set


def icarl_classify(test_tasks, classes_exemplars, model):
    exemplars_mean = torch.tensor(np.ndarray((0, 100)))

    for cls_idx, cls in enumerate(range(100)):
        idxs = np.nonzero(np.array(classes_exemplars.targets) == cls)[0].tolist()
        aux_set = copy.copy(classes_exemplars)
        aux_set.data = classes_exemplars.data[idxs]
        aux_set.targets = np.array(classes_exemplars.targets)[idxs].tolist()

        features = get_dataset_features(model, aux_set)
        class_mean = torch.mean(features, dim=0)
        exemplars_mean = torch.cat((exemplars_mean, class_mean.unsqueeze(0)))

    print(torch.argmax(exemplars_mean, dim=1))

    tasks_acc = []
    model.eval(); model.zero_grad()
    for test_task in test_tasks:
        test_loader = data.utils.get_dataloader(test_task)
        hits = 0
        total_samples = 0
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            total_samples += len(labels)
            with torch.no_grad():
                model_output = model(imgs)

            for feat, lab in zip(model_output, labels):
                differences = np.array([])
                for exemplar_mean in exemplars_mean:
                    differences = np.concatenate((differences,
                                                  np.array([torch.sum(feat.cpu().detach() - exemplar_mean).tolist()])))
                prediction = np.argmin(differences, axis=0)
                if prediction == lab:
                    hits += 1
                
        task_acc = hits/total_samples
        tasks_acc.append(task_acc)
    
    test_acc = sum(tasks_acc)/len(tasks_acc)
    return test_acc

def main(config):
    ### ------ Load Data ------ ###
    train_tasks = data.utils.load_tasks(config['dataset'],
                                        config['n_tasks'],
                                        train=True)

    test_tasks = data.utils.load_tasks(config['dataset'],
                                       config['n_tasks'],
                                       train=False)
    
    #model = resnet34(pretrained=False, progress=True).to(device)
    #model.fc = nn.Sequential(nn.Linear(512, 100), nn.Softmax(dim=1))

    model = networks.Classifier(n_classes=100)

    exemplars_set = None
    bar = tqdm(range(config['n_tasks']))
    for task in bar:
        train_set = copy.copy(train_tasks[task])
        model, exemplars_set = icarl_incremental_train(model=model,
                                                dataset=train_set,
                                                k=2000,
                                                exemplars_set=exemplars_set)

    test_acc = icarl_classify(test_tasks, exemplars_set, model)
    print(f'Test Accuracy: {test_acc}')


def load_config(file):
    with open(file, 'r') as reader:
        config = yaml.load(reader, Loader=yaml.SafeLoader)

    return config


if __name__ == '__main__':
    config = load_config('./config.yaml')
    main(config)
