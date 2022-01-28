import copy
import numpy as np
from torchvision import datasets, transforms

def load_data(dataset, train=True):
    if dataset == 'CIFAR100':
        data = datasets.CIFAR100(root='./Datasets',
                                 download=True,
                                 train=train,
                                 transform=transforms.ToTensor())
    else:
        print('Invalid dataset.')
        exit()

    return data
    

def create_tasks(n_tasks, data):
    n_classes = len(set(data.targets))

    assert n_classes % n_tasks == 0

    tasks_labels = []
    for idx, v in enumerate(range(n_classes//n_tasks, n_classes+1, n_classes//n_tasks)):
        tasks_labels.append([x for x in range(idx*(n_classes//n_tasks), v, 1)])

    tasks_sets = []

    for labels in tasks_labels:
        idxs = []
        for label in labels:
            all_idxs = np.nonzero(np.isin(data.targets, [label]))[0]
            idxs.extend(all_idxs)
        task_set = copy.deepcopy(data)
        task_set.targets = np.array(data.targets)[idxs].tolist()
        task_set.data = data.data[idxs]
        tasks_sets.append(task_set)

    return tasks_sets

def load_tasks(dataset, n_tasks, train=True):
    data = load_data(dataset, train)
    tasks = create_tasks(n_tasks, data)

    return tasks
