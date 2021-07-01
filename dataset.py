import torch
from torchvision import datasets
import torchvision.transforms as transforms


class DataSet:

    def __init__(self, data, labels):
        self.data = data.view(data.shape[0], -1).float() / 255.0
        self.labels = labels
        self._index_in_epoch = 0
        self._epochs_completed = 0
        self._num_examples = len(self.data)

    def next_batch(self, batch_size, stopiter=False):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            self._epochs_completed += 1
            perm = torch.randperm(len(self.data))
            self.data = self.data[perm]
            self.labels = self.labels[perm]
            if stopiter:
                self._index_in_epoch = 0
                raise StopIteration
            else:
                start = 0
                self._index_in_epoch = batch_size
        end = self._index_in_epoch
        return self.data[start:end], self.labels[start:end]


class MNISTTrainSet:

    def __init__(self, num_labeled, batch_size):
        self.batch_size = batch_size
        self.num_labeled = num_labeled
        mnist = datasets.MNIST(root='./mnist', train=True, download=True, 
                                        transform=transforms.Compose([transforms.ToTensor()]))
        data = mnist.data
        labels = mnist.targets

        self.unlabeled_ds = DataSet(data, labels)
        self.num_examples = self.unlabeled_ds._num_examples
        indices = torch.arange(self.num_examples)
        perm = torch.randperm(self.num_examples)
        data = data[perm]
        labels = labels[perm]
        n_classes = labels.max() + 1
        n_from_each_class = int(num_labeled / n_classes)
        i_labeled = []
        for c in range(n_classes):
            i = indices[labels==c][:n_from_each_class]
            i_labeled += i.tolist()
        l_data = data[i_labeled]
        l_labels = labels[i_labeled]
        self.labeled_ds = DataSet(l_data, l_labels)
    
    def next_batch(self):
        unlabeled_data, _ = self.unlabeled_ds.next_batch(self.batch_size, stopiter=True)
        if self.batch_size > self.num_labeled:
            labeled_data, labels = self.labeled_ds.next_batch(self.num_labeld)
        else:
            labeled_data, labels = self.labeled_ds.next_batch(self.batch_size)
        data = torch.vstack((labeled_data, unlabeled_data))
        return data, labels

    def __iter__(self):
        return self

    def __next__(self):
        return self.next_batch()


class MNISTTestSet:

    def __init__(self):
        mnist = datasets.MNIST(root='./mnist', train=False, download=True, 
                                        transform=transforms.Compose([transforms.ToTensor()]))
        data = mnist.data
        labels = mnist.targets
        self.ds = DataSet(data, labels)

    def next_batch(self):
        return self.ds.next_batch(self.ds._num_examples)