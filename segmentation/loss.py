import torch
import torch.nn as nn
from torch.autograd import Function


def major_version():
    version = torch.__version__
    return int(version.split('.')[0])


def minor_version():
    version = torch.__version__
    return int(version.split('.')[1])


class BinaryDiceLoss(Function):
    """ Dice Loss for binary segmentation
        Dice loss = 1 - Dice (minimize loss, maximize dice)

        The computation of Dice is slightly different in the forward and backward passes

        A is prediction result, B is ground-truth (binary label).

        In the FORWARD pass:
        The Dice is computed exactly as its definition. Binarized A first.
        Intersect = dot(A, B)
        Sum = sum(A) + sum(B)
        Dice = 2 * Intersect / Sum

        In the BACKWARD pass:
        The gradient is derived from the the following definition. A is softmax result, B is binary label.
        Dice = 2 * Intersect / Sum
        Intersect = dot(A, B)
        Sum = dot(A, A) + dot(B, B)    (interesting definition)

        Partially expand the derivative:
        d(Dice)/dA = 2 * [d(Intersect)/dA * Sum - Intersect * d(Sum)/dA] / (Sum)^2
        d(Intersect)/dA = B
        d(Sum)/dA = 2A

        Combine the above three definitons together
        d(Dice)/dA = 2 * [B * Sum - 2A * Intersect] / (Sum)^2

        The values Intersect and Sum are used from the forward pass.
    """

    @staticmethod
    def forward(ctx, input, target, save=True, epsilon=1e-6):

        batchsize = input.size(0)

        # convert probability to binary label using maximum probability
        _, input_label = input.max(1)

        # convert to floats
        input_label = input_label.float()
        target_label = target.float()

        if save:
            # save float version of target for backward
            ctx.save_for_backward(input, target_label)

        # convert to 1D
        input_label = input_label.view(batchsize, -1)
        target_label = target_label.view(batchsize, -1)

        # compute dice score
        ctx.intersect = torch.sum(input_label * target_label, 1)
        input_area = torch.sum(input_label, 1)
        target_area = torch.sum(target_label, 1)

        ctx.sum = input_area + target_area + 2 * epsilon

        # batch dice loss and ignore dice loss where target area = 0
        batch_loss = 1 - 2 * ctx.intersect / ctx.sum
        batch_loss[target_area == 0] = 0
        loss = batch_loss.mean()

        if minor_version() < 4:
            return torch.FloatTensor(1).fill_(loss)
        else:
            return loss

    @staticmethod
    def backward(ctx, grad_out):

        # gradient computation
        # d(Dice) / dA = [2 * target * Sum - 4 * input * Intersect] / (Sum) ^ 2
        #              = (2 * target / Sum) - (4 * input * Intersect / Sum^2)
        #              = (2 / Sum) * target - (4 * Intersect / Sum^2) * Input
        #              = a * target - b * input
        #
        # DiceLoss = 1 - Dice
        # d(DiceLoss) / dA = -a * target + b * input
        input, target = ctx.saved_tensors
        intersect, sum = ctx.intersect, ctx.sum

        a = 2 / sum
        b = 4 * intersect / sum / sum

        batchsize = a.size(0)
        a = a.view(batchsize, 1, 1, 1)
        b = b.view(batchsize, 1, 1, 1)

        grad_diceloss = -a * target + b * input[:, 1:2]

        # TODO for target=0 (background), if whether b is close 0, add epsilon to b

        # sum gradient of foreground and background probabilities should be zero
        if minor_version() < 4:
            grad_input = torch.cat((grad_diceloss * -grad_out[0],
                                    grad_diceloss * grad_out[0]), 1)
        else:
            grad_input = torch.cat((grad_diceloss * -grad_out.item(),
                                grad_diceloss *  grad_out.item()), 1)

        # 1) gradient w.r.t. input, 2) gradient w.r.t. target
        return grad_input, None
    

class MultiDiceLoss(nn.Module):
    """
    Dice Loss for egmentation(include binary segmentation and multi label segmentation)
    This class is generalization of BinaryDiceLoss
    """
    def __init__(self, weights, num_class, device):
        """
        :param weights: weight for each class dice loss
        :param num_class: the number of class
        """
        super(MultiDiceLoss, self).__init__()
        self.num_class = num_class

        assert len(weights) == self.num_class, "the length of weight must equal to num_class"
        self.weights = torch.FloatTensor(weights)
        self.weights = self.weights/self.weights.sum()
        self.weights = self.weights.to(device)

    def forward(self, input_tensor, target):
        """
        :param input_tensor: network output tensor
        :param target: ground truth
        :return: weighted dice loss and a list for all class dice loss, expect background
        """
        dice_losses = []
        weight_dice_loss = 0
        all_slice = torch.split(input_tensor, [1] * self.num_class, dim=1)

        for i in range(self.num_class):
            # prepare for calculate label i dice loss
            slice_i = torch.cat([1 - all_slice[i], all_slice[i]], dim=1)
            target_i = (target == i) * 1

            # BinaryDiceLoss save forward information for backward
            # so we can't use one BinaryDiceLoss for all classes
            dice_function = BinaryDiceLoss()
            dice_i_loss = dice_function.apply(slice_i, target_i)

            # save all classes dice loss and calculate weighted dice
            dice_losses.append(dice_i_loss)
            weight_dice_loss += dice_i_loss * self.weights[i]

        return weight_dice_loss, [dice_loss.item() for dice_loss in dice_losses]


def cal_dsc_loss(input, target):

    batchsize = input.size(0)

    _, input_label = input.data.max(1)
    input_label = input_label.float().view(batchsize, -1)
    target_label = target.data.view(batchsize, -1)

    intersect = torch.sum(input_label * target_label, 1)

    epsilon = 1e-6
    input_area = torch.sum(input_label, 1)
    target_area = torch.sum(target_label, 1)
    sum = input_area + target_area + 2 * epsilon

    batch_loss = 1.0 - 2.0 * intersect / sum
    batch_loss[target_area == 0] = 0

    return batch_loss

