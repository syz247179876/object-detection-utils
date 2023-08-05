import torch
import typing as t


def iou_loss(
        box1: torch.Tensor,
        box2: torch.Tensor,
        return_loss: bool = True
) -> t.Union[torch.Tensor, t.Tuple[..., torch.Tensor]]:
    """
    loss based on IOU
    Input:
        box1: dimension -> [n, 4], 4 -> [x1, y1, x2, y2]
        box2: dimension -> [m, 4], 4 -> [x1, y1, x2, y2]

    Output:
        output: dimension -> [n, m]
    """

    a_size, b_size = box1.size(0), box2.size(1)

    # [n, 0:2] -> [n, m, 0:2]
    # [m, 0:2] -> [n, m, 0:2]
    inter_t_l = torch.max(box1[:, :2].unsqueeze(1).expand(a_size, b_size, 2),
                          box2[:, :2].unsqueeze(0).expand(a_size, b_size, 2))
    inter_t_r = torch.max(box1[:, 2:].unsqueeze(1).expand(a_size, b_size, 2),
                          box2[:, 2:].unsqueeze(0).expand(a_size, b_size, 2))
    inter = torch.clamp(inter_t_r - inter_t_l, min=0)
    inter = inter[..., 0] * inter[..., 1]

    # [n,] -> [n, 1] -> [n, m]
    # [m,] -> [1, m] -> [n, m]
    area_a = ((box1[..., 2] - box1[..., 0]) * (box1[..., 3] - box1[..., 1])).unsqueeze(1).expand_as(inter)
    area_b = ((box2[..., 2] - box2[..., 0]) * (box2[..., 3] - box2[..., 1])).unsqueeze(0).expand_as(inter)
    union = (area_a + area_b - inter + 1e-20)
    iou_plural = inter / union
    return -torch.log(iou_plural) if return_loss else (inter, union, iou_plural)


def giou_loss(
        box1: torch.Tensor,
        box2: torch.Tensor,
        return_loss: bool = True,
) -> t.Union[torch.Tensor, t.Tuple[..., torch.Tensor]]:
    """
    loss based on GIOU, add calculation method for intersection scale.

    Advantages:
        1.GIOU based on ratio, so it is not sensitive to scale compared to MSE loss
        2.GIOU not only focuses on overlapping areas, but also other non overlapping areas, so it make up for IOU
        which only care about overlapping areas.
    Input:
        box1: dimension -> [n, 4], 4 -> [x1, y1, x2, y2]
        box2: dimension -> [m, 4], 4 -> [x1, y1, x2, y2]

    Output:
        output: dimension -> [n, m]
    """

    box1_num, box2_num = box1.size(0), box2.size(0)
    _, union, iou_plural = iou_loss(box1, box2, False)

    # dim trans => [n, m, 4], box2 dim => [n, m, 4]
    box1 = box1.unsqueeze(1).expand(box1_num, box2_num, 4)
    box2 = box2.unsqueeze(0).expand(box1_num, box2_num, 4)

    x_c1 = torch.min(box1[..., 0], box2[..., 0])
    x_c2 = torch.max(box1[..., 2], box2[..., 2])
    y_c1 = torch.min(box1[..., 1], box2[..., 1])
    y_c2 = torch.max(box1[..., 3], box2[..., 3])
    external_area = (x_c2 - x_c1) * (y_c2 - y_c1)
    g_iou = iou_plural - (external_area - union) / external_area

    return 1 - g_iou if return_loss else (iou_plural, g_iou, x_c2 - x_c1, y_c2 - y_c1)


def diou_loss(
        box1: torch.Tensor,
        box2: torch.Tensor,
        return_loss: bool = True,
) -> torch.Tensor:
    """
    loss based on DIOU
    Input:
        box1: dimension -> [n, 4], 4 -> [x1, y1, x2, y2]
        box2: dimension -> [m, 4], 4 -> [x1, y1, x2, y2]

    Output:
        output: dimension -> [n, m]
    """

    box1_num, box2_num = box1.size(0), box2.size(0)
    iou_plural, _, dia_x, dia_y = giou_loss(box1, box2, False)
    x_mid1 = (box1[:, 0] + box1[:, 2]) / 2.
    y_mid1 = (box1[:, 1] + box1[:, 3]) / 2.

    x_mid2 = (box2[:, 0] + box2[:, 2]) / 2.
    y_mid2 = (box2[:, 1] + box2[:, 3]) / 2.

    # dim trans -> [n, m, 2]
    mid_box1 = torch.stack((x_mid1, y_mid1), dim=-1).unsqueeze(1).expand(box1_num, box2_num, 2)
    mid_box2 = torch.stack((x_mid2, y_mid2), dim=-1).unsqueeze(0).expand(box1_num, box2_num, 2)

    # calculate euclidean distance
    eu_dis = torch.pow(mid_box1 - mid_box2, 2).sum(dim=-1, keepdim=True)
    # calculate diagonal distance
    dia_dis = torch.pow(dia_x, 2) + torch.pow(dia_y, 2)
    d_iou = iou_plural - eu_dis / dia_dis
    return 1 - d_iou if return_loss else d_iou


def ciou_loss(self, box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    """
    loss based on CIOU
    Input:
        box1: dimension -> [n, 4], 4 -> [x1, y1, x2, y2]
        box2: dimension -> [m, 4], 4 -> [x1, y1, x2, y2]

    Output:
        output: dimension -> [n, m]
    """
    pass
