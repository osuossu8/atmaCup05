from sklearn.metrics import average_precision_score


class AverageMeter(object):
    """Computes and stores the average and current value"""
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


def loss_fn(logits, targets):
    loss_fct = nn.BCEWithLogitsLoss()
    loss = loss_fct(logits, targets)
    return loss


def train_fn(data_loader, model, optimizer, device, scheduler=None):
    model.train()
    losses = AverageMeter()

    tk0 = tqdm(data_loader, total=len(data_loader))
    
    y_true = []
    y_pred = []
    for bi, d in enumerate(tk0):

        images = d["images"].to(device, dtype=torch.float32)
        targets = d["targets"].to(device, dtype=torch.float32)

        model.zero_grad()
        outputs = model(images)

        loss = loss_fn(torch.squeeze(outputs), targets)
        loss.backward()
        optimizer.step()

        outputs = torch.sigmoid(outputs).cpu().detach().numpy()
        targets = targets.float().cpu().detach().numpy()

        y_true.append(targets)
        y_pred.append(outputs)

        losses.update(loss.item(), images.size(0))
        tk0.set_postfix(loss=losses.avg)

    y_true_cat = np.concatenate(y_true, 0)
    y_pred_cat = np.concatenate(y_pred, 0)

    avg_precision = average_precision_score(y_true_cat, y_pred_cat)
    print()
    print()
    print('#####')
    print(f'train pr_score : {avg_precision}')
    print('#####')


def eval_fn(data_loader, model, device):
    model.eval()
    losses = AverageMeter()
    
    y_true = []
    y_pred = []
    with torch.no_grad():
        tk0 = tqdm(data_loader, total=len(data_loader))
        for bi, d in enumerate(tk0):
            images = d["images"].to(device, dtype=torch.float32)
            targets = d["targets"].to(device, dtype=torch.float32)

            outputs = model(images)

            loss = loss_fn(torch.squeeze(outputs), targets)

            outputs = torch.sigmoid(outputs).cpu().detach().numpy()
            targets = targets.float().cpu().detach().numpy()

            y_true.append(targets)
            y_pred.append(outputs)

            losses.update(loss.item(), images.size(0))
            tk0.set_postfix(loss=losses.avg)

        y_true_cat = np.concatenate(y_true, 0)
        y_pred_cat = np.concatenate(y_pred, 0)

        avg_precision = average_precision_score(y_true_cat, y_pred_cat)
        print()
        print()
        print('#####')
        print(f'valid pr_score : {avg_precision}')
        print('#####')
    return avg_precision, losses.avg


def test_fn(data_loader, model, device):
    model.eval()
    
    preds = []
    with torch.no_grad():
        tk0 = tqdm(data_loader, total=len(data_loader))
        for bi, d in enumerate(tk0):
            images = d["images"].to(device, dtype=torch.float32)

            outputs = model(images)
            outputs = torch.sigmoid(outputs).cpu().detach().numpy()
            preds.append(outputs)
    
    return preds
