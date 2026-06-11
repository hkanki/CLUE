import torch

models = {}
def register_model(name):
    def decorator(cls):
        models[name] = cls
        return cls
    return decorator
#モデル名（例：LeNet）からモデルインスタンスを作って、GPUが使えるならGPUへ移動して返す関数
def get_model(name, num_cls=10, **args):
    net = models[name](num_cls=num_cls, **args)

    # (1) タプルの場合はそれぞれを GPU に移動
    if isinstance(net, tuple):
        net = tuple(m.cuda() for m in net) if torch.cuda.is_available() else net
    # (2) 単一モデルならそのまま GPU に移動
    else:
        if torch.cuda.is_available():
            net = net.cuda()
    return net
