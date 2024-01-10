import logging
import numpy as np
import torch
from submodules.Inpaint_Anything.sam_segment import predict_masks_with_sam
from submodules.Inpaint_Anything.lama_inpaint import inpaint_img_with_lama
from submodules.Inpaint_Anything.utils import dilate_mask

logger = logging.getLogger(__name__)


class Inpainter:
    def __init__(self, opt):
        self.opt = opt
        self.hand_type = self.opt.hand_type
        self.object_type = self.opt.object_type
        return

    def to(self, device):
        self.device = device

    def __call__(self, images, handarm_segs, object_segs):
        batch_size = images.shape[0]

        # handarm mask
        handarm_mask = torch.zeros_like(handarm_segs)
        handarm_mask[handarm_segs > 0] = 255
        if self.hand_type == "mask":
            handarm_masks = handarm_mask
        elif self.hand_type == "point":
            handarm_masks = []
            for b in range(batch_size):
                x, y = Inpainter.barycenter(handarm_mask[b])
                point = torch.cat([x[None], y[None]], dim=0)
                handarm_masks.append(self.sam(images[b], point))
            handarm_masks = torch.stack(handarm_masks, dim=0)
        else:
            logger.error(f"Unknown inpaint hand_type: {self.hand_type}")

        # object mask
        object_mask = torch.zeros_like(object_segs)
        object_mask[object_segs > 0] = 255
        if self.object_type == "mask":
            object_masks = object_mask
        elif self.object_type == "point":
            object_masks = []
            for b in range(batch_size):
                x, y = Inpainter.barycenter(object_mask[b])
                point = torch.cat([x[None], y[None]], dim=0)
                object_masks.append(self.sam(images[b], point))
            object_masks = torch.stack(object_masks, dim=0)
        else:
            logger.error(f"Unknown inpaint object_type: {self.object_type}")

        # handarm + object mask
        inpaint_masks = torch.zeros_like(handarm_segs)
        inpaint_masks[handarm_masks > 0] = 255
        inpaint_masks[object_masks > 0] = 255
        inpaint_masks = dilate_mask(inpaint_masks.cpu().numpy(), self.opt.mask_dilate_size)

        # inpaint
        inpainted_images = []
        for b in range(batch_size):
            inpainted_images.append(self.lama(images[b], inpaint_masks[b]))
        inpainted_images = np.stack(inpainted_images, axis=0)

        return inpainted_images

    def sam(self, image, point):
        masks, _, _ = predict_masks_with_sam(
            image,
            point,
            torch.tensor([1], dtype=torch.int, device=self.device),
            model_type=self.opt.sam.model_type,
            ckpt_p=self.opt.sam.ckpt,
            device=self.device,
        )
        masks = masks.to(torch.uint8) * 255
        mask = masks[0, self.opt.sam.mask_idx]  # does not support batch size > 1
        return mask

    def lama(self, image, mask):
        img_inpainted = inpaint_img_with_lama(
            image,
            mask,
            self.opt.lama.config_file,
            self.opt.lama.ckpt,
            device=self.device,
        )
        return img_inpainted

    @staticmethod
    def barycenter(mask):
        y_indices, x_indices = torch.where(mask)
        x_center = torch.mean(x_indices.float())
        y_center = torch.mean(y_indices.float())

        return x_center, y_center
