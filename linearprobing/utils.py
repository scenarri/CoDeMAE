import os
import random
import numpy as np
import torch
import time
import matplotlib.pyplot as plt
import datetime
from collections import defaultdict, deque
import torch.distributed as dist

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.

    # Ensure that CuDNN uses deterministic algorithms
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    #torch.use_deterministic_algorithms(True, warn_only=True)

    # Disable TF32 (TensorFloat-32), which can cause nondeterministic behavior on NVIDIA A100 GPUs and newer
    if hasattr(torch.backends.cudnn, 'allow_tf32'):
        torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch, 'allow_tf32'):
        torch.allow_tf32 = False
    
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass

    #print(f"All random seeds set to {seed}")


def show(tensor: torch.Tensor,
         mode: str = 'rgb',
         index: int = 0,
         ax: plt.Axes = None) -> None:
    assert tensor.ndim == 4, "输入张量必须是4维 (N, C, W, H)"
    assert mode in ['sar', 'rgb'], "模式必须是 'sar' 或 'rgb'"
    assert 0 <= index < tensor.shape[0], "索引超出范围"
    
    # 选择指定样本并转换为numpy
    sample = tensor[index].detach().cpu().numpy()  # (C, W, H)
    
    # 通道处理
    if mode == 'sar':
        img = sample[0]  # (W, H)
        cmap = 'gray'
    elif mode == 'rgb':
        img = sample[[3, 2, 1]]  # (3, W, H)
        img = np.transpose(img, (1, 2, 0))  # (W, H, 3)
    
    # 数值转换 [0,1] -> [0,255]
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    
    # 显示图像
    # ax = ax or plt.gca()
    plt.imshow(img, cmap=cmap if mode == 'sar' else None)
    plt.show()


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)
    
class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                # if torch.cuda.is_available():
                #     print(log_msg.format(
                #         i, len(iterable), eta=eta_string,
                #         meters=str(self),
                #         time=str(iter_time), data=str(data_time),
                #         memory=torch.cuda.max_memory_allocated() / MB))
                # else:
                #     print(log_msg.format(
                #         i, len(iterable), eta=eta_string,
                #         meters=str(self),
                #         time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    
        # print('{} Total time: {} ({:.4f} s / it)'.format(
        #     header, total_time_str, total_time / len(iterable)))