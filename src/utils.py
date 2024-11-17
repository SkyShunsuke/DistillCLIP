import torch

class LogMeter(object):
    """LogMeter class.
    A class to record the average value of a metric.
    
    Attributes:
        val (float): the current value of the metric.
        avg (float): the average value of the metric.
        sum (float): the sum of the metric.
        count (int): the number of the
    Usage:
        meter = LogMeter()
        meter.update(1)
        meter.update(2)
        print(meter.avg)
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self):
        fmtstr = '{name} {val} ({avg})'
        return fmtstr.format(name=self.__class__.__name__, **self.__dict__)
    
def get_onehot(label, num_classes):
    """Get one hot encoding of the label.
    
    Args:
        label (torch.Tensor): the label tensor.
        num_classes (int): the number of classes.
    Returns:
        torch.Tensor: the one hot encoding of the label.
    """
    onehot = torch.zeros(label.size(0), num_classes)
    onehot.scatter_(1, label.unsqueeze(1), 1)
    return onehot