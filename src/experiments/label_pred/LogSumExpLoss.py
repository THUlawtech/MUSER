import torch
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class LogSumExp(nn.Module):
    def __init__(self):
        super(LogSumExp, self).__init__()

    def forward(self, prediction, target, mask=None, reduction=None):
        if mask is None:
            mask = torch.ones_like(prediction).long().cuda()

        prediction_pos = prediction.masked_fill((1 - target).bool(), 1e12)

        prediction_neg = prediction.masked_fill((target | (1 - mask)).bool(), -1e12)

        zeros = torch.zeros_like(prediction[..., :1]).cuda()

        prediction_pos = torch.cat((-prediction_pos, zeros), dim=-1)
        prediction_neg = torch.cat((prediction_neg, zeros), dim=-1)

        pos_loss = torch.logsumexp(prediction_pos, dim=-1)
        neg_loss = torch.logsumexp(prediction_neg, dim=-1)
        loss = pos_loss + neg_loss

        if reduction == 'mean':
            return loss.mean()

        return loss


def log_sum_exp(prediction, target, mask=None):
    """
    Expand softmax to multi-label classification

    :param prediction:
        Torch Float Tensor with shape of [batch_size * N * sequence_length]
            don't use sigmoid or softmax

    :param target:
        Torch Long Tensor with shape of [batch_size * N * sequence_length]
            one-hot representation for the label

    :param mask:
        Torch Long Tensor with shape of [batch_size * N * sequence_length]
            attention mask, mask out the padded token.
            (padded token should not be count as negative token)

    :return:
        log sum exp loss with shape of [batch_size * 112]
            link: https://spaces.ac.cn/archives/7359
    """
    if mask is None:
        mask = torch.ones_like(prediction).long().to(device)

    prediction_pos = prediction.masked_fill((1-target).bool(), 1e12)

    prediction_neg = prediction.masked_fill((target | (1-mask)).bool(), -1e12)

    zeros = torch.zeros_like(prediction[..., :1]).to(device)

    prediction_pos = torch.cat((-prediction_pos, zeros), dim=-1)
    prediction_neg = torch.cat((prediction_neg, zeros), dim=-1)

    pos_loss = torch.logsumexp(prediction_pos, dim=-1)
    neg_loss = torch.logsumexp(prediction_neg, dim=-1)

    return pos_loss + neg_loss
