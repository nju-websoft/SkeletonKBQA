import qelos as q
import torch
import numpy as np
from IPython import embed
import math
from functools import partial


__all__ = ["batch_reset", "epoch_reset", "LossWrapper", "BestSaver", "no_gold", "pp_epoch_losses",
           "train_batch", "train_epoch", "test_epoch", "run_training", "CosineLRwithWarmup", "eval_loop"]


def batch_reset(module):        # performs all resetting operations on module before using it in the next batch
    for modu in module.modules():
        if hasattr(modu, "batch_reset"):
            modu.batch_reset()


def epoch_reset(module):        # performs all resetting operations on module before using it in the next epoch
    batch_reset(module)
    for modu in module.modules():
        if hasattr(modu, "epoch_reset"):
            modu.epoch_reset()


class LossWrapper(object):
    """ Wraps a normal loss with aggregating and other functionality """

    def __init__(self, loss, name=None, mode="mean", **kw):
        """
        :param loss:    actual loss class
        :param name:    name for this loss (class name by default)
        :param mode:    "mean" or "sum"
        :param kw:
        """
        super(LossWrapper, self).__init__()
        self.loss, self.aggmode = loss, mode
        self.name = name if name is not None else loss.__class__.__name__

        self.agg_history = []
        self.agg_epochs = []

        self.epoch_agg_values = []
        self.epoch_agg_sizes = []

    def get_epoch_error(self):
        """ returns the aggregated error for this epoch so far """
        if self.aggmode == "mean":
            if len(self.epoch_agg_sizes) == 0:
                ret = 0.
            else:
                total = sum(self.epoch_agg_sizes)
                fractions = [x/total for x in self.epoch_agg_sizes]
                parts = [x * y for x, y in zip(self.epoch_agg_values, fractions)]
                ret = sum(parts)
        else:
            ret = sum(self.epoch_agg_values)
        return ret

    def push_epoch_to_history(self, epoch=None):
        self.agg_history.append(self.get_epoch_error())
        if epoch is not None:
            self.agg_epochs.append(epoch)

    def __call__(self, pred, gold, _numex=None, **kw):
        l = self.loss(pred, gold, **kw)

        if _numex is None:
            _numex = pred.size(0) if not q.issequence(pred) else pred[0].size(0)
        if isinstance(l, tuple) and len(l) == 2:     # loss returns numex too
            _numex = l[1]
            l = l[0]
        if isinstance(l, torch.Tensor):
            lp = l.item()
        else:
            lp = l
        self.epoch_agg_values.append(lp)
        self.epoch_agg_sizes.append(_numex)
        return l

    def _reset(self):   # full reset
        self.reset_agg()
        self.agg_history = []
        self.agg_epochs = []

    def reset_agg(self):    # reset epoch stats
        self.epoch_agg_values = []
        self.epoch_agg_sizes = []


def no_gold(losses):
    all_linear = True
    some_linear = False
    for loss in losses:
        if isinstance(loss.loss, (q.LinearLoss, q.SelectedLinearLoss)):
            some_linear = True
        else:
            all_linear = False
    assert(all_linear == some_linear)
    return all_linear


def pp_epoch_losses(*losses:LossWrapper):
    values = [loss.get_epoch_error() for loss in losses]
    ret = " :: ".join("{:.4f}".format(value) for value in values)
    return ret


# region loops
def eval_loop(model, dataloader, device=torch.device("cpu")):
    tto = q.ticktock("testing")
    tto.tick("testing")
    tt = q.ticktock("-")
    totaltestbats = len(dataloader)
    model.eval()
    epoch_reset(model)
    outs = []
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            batch = (batch,) if not q.issequence(batch) else batch
            batch = q.recmap(batch, lambda x: x.to(device) if isinstance(x, torch.Tensor) else x)

            batch_reset(model)
            modelouts = model(*batch)

            tt.live("eval - [{}/{}]"
                .format(
                i + 1,
                totaltestbats
            )
            )
            if not q.issequence(modelouts):
                modelouts = (modelouts,)
            if len(outs) == 0:
                outs = [[] for e in modelouts]
            for out_e, mout_e in zip(outs, modelouts):
                out_e.append(mout_e)
    ttmsg = "eval done"
    tt.stoplive()
    tt.tock(ttmsg)
    tto.tock("tested")
    ret = [torch.cat(out_e, 0) for out_e in outs]
    return ret


def train_batch(batch=None, model=None, optim=None, losses=None, device=torch.device("cpu"),
                batch_number=-1, max_batches=0, current_epoch=0, max_epochs=0,
                on_start=tuple(), on_before_optim_step=tuple(), on_after_optim_step=tuple(), on_end=tuple()):
    """
    Runs a single batch of SGD on provided batch and settings.
    :param batch:  batch to run on
    :param model:   torch.nn.Module of the model
    :param optim:       torch optimizer
    :param losses:      list of losswrappers
    :param device:      device
    :param batch_number:    which batch
    :param max_batches:     total number of batches
    :param current_epoch:   current epoch
    :param max_epochs:      total number of epochs
    :param on_start:        collection of functions to call when starting training batch
    :param on_before_optim_step:    collection of functions for before optimization step is taken (gradclip)
    :param on_after_optim_step:     collection of functions for after optimization step is taken
    :param on_end:              collection of functions to call when batch is done
    :return:
    """
    [e() for e in on_start]
    optim.zero_grad()
    model.train()

    batch = (batch,) if not q.issequence(batch) else batch
    batch = q.recmap(batch, lambda x: x.to(device) if isinstance(x, torch.Tensor) else x)
    numex = batch[0].size(0)

    if q.no_gold(losses):
        batch_in = batch
        gold = None
    else:
        batch_in = batch[:-1]
        gold = batch[-1]

    q.batch_reset(model)
    modelouts = model(*batch_in)

    trainlosses = []
    for loss_obj in losses:
        loss_val = loss_obj(modelouts, gold, _numex=numex)
        loss_val = [loss_val] if not q.issequence(loss_val) else loss_val
        trainlosses.extend(loss_val)

    cost = trainlosses[0]
    # penalties
    penalties = 0
    for loss_obj, trainloss in zip(losses, trainlosses):
        if isinstance(loss_obj.loss, q.loss.PenaltyGetter):
            penalties += trainloss
    cost = cost + penalties

    if torch.isnan(cost).any():
        print("Cost is NaN!")
        embed()

    cost.backward()

    [e() for e in on_before_optim_step]
    optim.step()
    [e() for e in on_after_optim_step]

    ttmsg = "train - Epoch {}/{} - [{}/{}]: {}".format(
                current_epoch+1,
                max_epochs,
                batch_number+1,
                max_batches,
                q.pp_epoch_losses(*losses),
                )

    [e() for e in on_end]
    return ttmsg


def train_epoch(model=None, dataloader=None, optim=None, losses=None, device=torch.device("cpu"), tt=q.ticktock("-"),
                current_epoch=0, max_epochs=0, _train_batch=train_batch, on_start=tuple(), on_end=tuple(), print_every_batch=False):
    """
    Performs an epoch of training on given model, with data from given dataloader, using given optimizer,
    with loss computed based on given losses.
    :param model:
    :param dataloader:
    :param optim:
    :param losses:  list of loss wrappers
    :param device:  device to put batches on
    :param tt:
    :param current_epoch:
    :param max_epochs:
    :param _train_batch:    train batch function, default is train_batch
    :param on_start:
    :param on_end:
    :return:
    """
    for loss in losses:
        loss.push_epoch_to_history(epoch=current_epoch-1)
        loss.reset_agg()
        loss.loss.to(device)

    model.to(device)

    [e() for e in on_start]

    q.epoch_reset(model)

    for i, _batch in enumerate(dataloader):
        ttmsg = _train_batch(batch=_batch, model=model, optim=optim, losses=losses, device=device,
                             batch_number=i, max_batches=len(dataloader), current_epoch=current_epoch,
                             max_epochs=max_epochs)
        if print_every_batch:
            tt.msg(ttmsg)
        else:
            tt.live(ttmsg)

    tt.stoplive()
    [e() for e in on_end]
    ttmsg = q.pp_epoch_losses(*losses)
    return ttmsg


def test_epoch(model=None, dataloader=None, losses=None, device=torch.device("cpu"),
            current_epoch=0, max_epochs=0, print_every_batch=False,
            on_start=tuple(), on_start_batch=tuple(), on_end_batch=tuple(), on_end=tuple()):
    """
    Performs a test epoch. If run=True, runs, otherwise returns partially filled function.
    :param model:
    :param dataloader:
    :param losses:
    :param device:
    :param current_epoch:
    :param max_epochs:
    :param on_start:
    :param on_start_batch:
    :param on_end_batch:
    :param on_end:
    :return:
    """
    tt = q.ticktock("-")
    model.eval()
    q.epoch_reset(model)
    [e() for e in on_start]
    with torch.no_grad():
        for loss_obj in losses:
            loss_obj.push_epoch_to_history()
            loss_obj.reset_agg()
            loss_obj.loss.to(device)
        for i, _batch in enumerate(dataloader):
            [e() for e in on_start_batch]

            _batch = (_batch,) if not q.issequence(_batch) else _batch
            _batch = q.recmap(_batch, lambda x: x.to(device) if isinstance(x, torch.Tensor) else x)
            batch = _batch
            numex = batch[0].size(0)


            if q.no_gold(losses):
                batch_in = batch
                gold = None
            else:
                batch_in = batch[:-1]
                gold = batch[-1]

            q.batch_reset(model)
            modelouts = model(*batch_in)

            testlosses = []
            for loss_obj in losses:
                loss_val = loss_obj(modelouts, gold, _numex=numex)
                loss_val = [loss_val] if not q.issequence(loss_val) else loss_val
                testlosses.extend(loss_val)

            ttmsg = "test - Epoch {}/{} - [{}/{}]: {}".format(
                        current_epoch + 1,
                        max_epochs,
                        i + 1,
                        len(dataloader),
                        q.pp_epoch_losses(*losses)
                    )
            if print_every_batch:
                tt.msg(ttmsg)
            else:
                tt.live(ttmsg)
            [e() for e in on_end_batch]
    tt.stoplive()
    [e() for e in on_end]
    ttmsg = q.pp_epoch_losses(*losses)
    return ttmsg


def run_training(run_train_epoch=None, run_valid_epoch=None, max_epochs=1, validinter=1,
                 print_on_valid_only=False, check_stop=tuple()):
    """

    :param run_train_epoch:     function that performs an epoch of training. must accept current_epoch and max_epochs. Tip: use functools.partial
    :param run_valid_epoch:     function that performs an epoch of testing. must accept current_epoch and max_epochs. Tip: use functools.partial
    :param max_epochs:
    :param validinter:
    :param print_on_valid_only:
    :return:
    """
    tt = q.ticktock("runner")
    validinter_count = 0
    current_epoch = 0
    stop_training = current_epoch >= max_epochs
    while stop_training is not True:
        tt.tick()
        ttmsg = run_train_epoch(current_epoch=current_epoch, max_epochs=max_epochs)
        ttmsg = "Epoch {}/{} -- {}".format(current_epoch+1, max_epochs, ttmsg)
        validepoch = False
        if run_valid_epoch is not None and validinter_count % validinter == 0:
            ttmsg_v = run_valid_epoch(current_epoch=current_epoch, max_epochs=max_epochs)
            ttmsg += " -- " + ttmsg_v
            validepoch = True
        validinter_count += 1
        if not print_on_valid_only or validepoch:
            tt.tock(ttmsg)
        current_epoch += 1
        stop_training = any([e() for e in check_stop])
        stop_training = stop_training or (current_epoch >= max_epochs)


def example_usage_basic():
    # 1. define model
    model = torch.nn.Sequential(torch.nn.Linear(5, 5),
                                torch.nn.Softmax(-1))

    # 2. define data
    x = torch.rand(4, 5)
    y = torch.randint(0, 5, (4,))
    dataset = torch.utils.data.TensorDataset(x, y)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)
    y_pred = model(x)
    print(y_pred)

    # 3. define losses and wrap them
    loss = torch.nn.CrossEntropyLoss(reduction="mean")
    loss2 = torch.nn.CrossEntropyLoss(reduction="sum")
    loss = q.LossWrapper(loss)
    loss2 = q.LossWrapper(loss2)
    # print(y.size(), y_pred.size())
    l = loss(y_pred, y)
    print(l)

    # 4. define optim
    optim = torch.optim.SGD(model.parameters(), lr=1.)

    # 5. other options (device, ...)
    device = torch.device("cpu")

    # 6. define training function (using partial)
    trainepoch = partial(q.train_epoch, model=model, dataloader=dataloader, optim=optim, losses=[loss, loss2], device=device)

    # 7. run training
    run_training(run_train_epoch=trainepoch, max_epochs=5)


def example_usage_full():
    # 1. define model
    model = torch.nn.Sequential(torch.nn.Linear(5, 5),
                                torch.nn.Softmax(-1))

    # 2. define data
    x = torch.rand(100, 5)
    y = torch.randint(0, 5, (100,))
    dataset = torch.utils.data.TensorDataset(x, y)
    traindataset, validdataset, testdataset = torch.utils.data.random_split(dataset, [70, 10, 20])
    trainloader = torch.utils.data.DataLoader(traindataset, batch_size=2, shuffle=True)
    validloader = torch.utils.data.DataLoader(validdataset, batch_size=2, shuffle=False)
    testloader = torch.utils.data.DataLoader(testdataset, batch_size=2, shuffle=False)

    # 3. define losses and wrap them
    loss = torch.nn.CrossEntropyLoss(reduction="mean")
    loss2 = torch.nn.CrossEntropyLoss(reduction="sum")
    loss = q.LossWrapper(loss)
    loss2 = q.LossWrapper(loss2)

    # 4. define optim
    optim = torch.optim.SGD(model.parameters(), lr=1.0)

    # 5. other options (device, ...)
    device = torch.device("cpu")

    # 6. define training function (using partial)
    trainepoch = partial(q.train_epoch, model=model, dataloader=trainloader, optim=optim, losses=[loss, loss2], device=device)

    # 7. define validation function (using partial)
    validepoch = partial(q.test_epoch, model=model, dataloader=validloader, losses=[loss, loss2], device=device)

    # 8. run training
    run_training(run_train_epoch=trainepoch, run_valid_epoch=validepoch, max_epochs=50)

    # 9. run test function
    testresults = q.test_epoch(model=model, dataloader=testloader, losses=[loss, loss2], device=device)
    print(testresults)


def example_usage_full_with_penalty_and_hyperparam():
    # 1. define model
    class Model(torch.nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.lin = torch.nn.Linear(5, 5)
            self.sm = torch.nn.Softmax(-1)
            self._pen = 0

        def batch_reset(self):      # called before every batch
            self._pen = 0           # resets penalty

        def get_penalty(self):      # must be specified to be called by PenaltyGetter
            return self._pen

        def forward(self, _x):
            _y = self.lin(_x)
            self._pen = torch.sum(_y, dim=1)
            return self.sm(_y)

    model = Model()

    # 2. define data
    x = torch.rand(100, 5)
    y = torch.randint(0, 5, (100,))
    dataset = torch.utils.data.TensorDataset(x, y)
    traindataset, validdataset, testdataset = torch.utils.data.random_split(dataset, [70, 10, 20])
    trainloader = torch.utils.data.DataLoader(traindataset, batch_size=2, shuffle=True)
    validloader = torch.utils.data.DataLoader(validdataset, batch_size=2, shuffle=False)
    testloader = torch.utils.data.DataLoader(testdataset, batch_size=2, shuffle=False)

    # 3. define losses and penalties and wrap them
    loss = torch.nn.CrossEntropyLoss(reduction="mean")
    loss2 = torch.nn.CrossEntropyLoss(reduction="sum")
    penweight = q.hyperparam(1.)
    pen = q.PenaltyGetter(model, "get_penalty", factor=penweight)
    loss = q.LossWrapper(loss)
    loss2 = q.LossWrapper(loss2)
    pen = q.LossWrapper(pen)

    # 4. define optim
    optim = torch.optim.SGD(model.parameters(), lr=1.)

    # 5. other options (device, ...)
    device = torch.device("cpu")
    def on_start_train_epoch():
        penweight.v /= 1.2
        print(q.v(penweight))


    # 6. define training function (using partial)
    trainepoch = partial(q.train_epoch, model=model, dataloader=trainloader, optim=optim, losses=[loss, loss2, pen], device=device, on_start=[on_start_train_epoch])

    # 7. define validation function (using partial)
    validepoch = partial(q.test_epoch, model=model, dataloader=validloader, losses=[loss, loss2], device=device)

    # 8. run training
    run_training(run_train_epoch=trainepoch, run_valid_epoch=validepoch, max_epochs=50)

    # 9. run test function
    testresults = q.test_epoch(model=model, dataloader=testloader, losses=[loss, loss2], device=device)
    print(testresults)



if __name__ == '__main__':
    # example_usage_basic()
    # example_usage_full()
    example_usage_full_with_penalty_and_hyperparam()



# endregion


class BestSaver(object):
    def __init__(self, criterion, model, path, higher_is_better=True, autoload=False,
                 verbose=False, **kw):
        super(BestSaver, self).__init__(**kw)
        self.criterion = criterion
        self.model = model
        self.path = path
        self.higher_better = 1. if higher_is_better else -1.
        self.best_criterion = -np.infty if higher_is_better else np.infty
        self.verbose = verbose
        self.callbacks = {}
        self.autoload = autoload        # automatically load on END event

    # def get_hooks(self, ee):
    #     hooks = {ee.END_EPOCH: self.save_best_model}
    #     if self.autoload:
    #         hooks[ee.END] = self.autoload_best
    #     return hooks

    def save_best_model(self):
        # assert isinstance(trainer, train)
        current_criterion = self.criterion()
        decision_value = current_criterion - self.best_criterion    # positive if current is higher
        decision_value *= self.higher_better            # higher better --> positive is higher = better
        # remark: with this way, later can extend to specifying by how much it should improve --> TODO
        if decision_value > 0:
            if self.verbose:
                print("Validation criterion improved from {} to {}. Saving model..."\
                      .format(self.best_criterion, current_criterion))
            self.best_criterion = current_criterion
            torch.save(self.model.state_dict(), self.path)

    def autoload_best(self):
        if self.verbose:
            print("Reloading best weights ({})".format(self.best_criterion))
        self.model.load_state_dict(torch.load(self.path))


class CosineLRwithWarmup(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, cyc_len, lr_min=0, last_epoch=-1, warmup=100):
        self.cyc_len = cyc_len
        self.lr_min = lr_min
        self.warmup = warmup
        super(CosineLRwithWarmup, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup:
            ret = [
                self.lr_min + (base_lr - self.lr_min) * (self.last_epoch / self.warmup)
                for base_lr in self.base_lrs
            ]
        else:
            ret = [self.lr_min + (base_lr - self.lr_min) *
                   (1 + math.cos(math.pi * (((self.last_epoch - self.warmup) % self.cyc_len)
                                            / self.cyc_len))) / 2
                   for base_lr in self.base_lrs]
        return ret