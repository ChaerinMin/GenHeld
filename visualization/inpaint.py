import logging
import os 

import numpy as np
import torch
from PIL import Image

from submodules.Inpaint_Anything.sam_segment import predict_masks_with_sam
from submodules.Inpaint_Anything.lama_inpaint import inpaint_img_with_lama
from submodules.Inpaint_Anything.utils import dilate_mask
from submodules.EgoHOS.mmsegmentation.mmseg.apis import inference_segmentor, init_segmentor
from submodules.EgoHOS.mmsegmentation.visualize import visualize_twohands, visualize_twohands_obj1

logger = logging.getLogger(__name__)


class Inpainter:
    def __init__(self, opt, device):
        self.opt = opt
        self.device = device
        self.hand_type = self.opt.hand_type
        self.object_type = self.opt.object_type

        # egohos
        self.hand_segmentor = init_segmentor(
            self.opt.egohos.hand_config,
            self.opt.egohos.hand_ckpt,
            device=self.device,
        )
        self.cb_segmentor = init_segmentor(
            self.opt.egohos.cb_config,
            self.opt.egohos.cb_ckpt,
            device=self.device,
        )
        self.object_segmentor = init_segmentor(
            self.opt.egohos.object_config,
            self.opt.egohos.object_ckpt,
            device=self.device,
        )
        return

    def __call__(self, images, fidxs):
        batch_size = images.shape[0]

        # handarm, object EgoHOS
        if os.path.exists(self.opt.cached.handarm_seg % fidxs[-1]):
            # load
            handarm_segs = []
            object_segs = []
            for fidx in fidxs:
                handarm_segs.append(np.array(Image.open(self.opt.cached.handarm_seg % fidx)))
                object_segs.append(np.array(Image.open(self.opt.cached.object_seg % fidx)))
        else:
            # compute
            handarm_segs, object_segs = self.egohos(images)
            handarm_vis, object_vis = self.visualize_egohos(images, handarm_segs, object_segs)
            # save 
            handarm_dir = os.path.dirname(self.opt.cached.handarm_seg % fidxs[0])
            object_dir = os.path.dirname(self.opt.cached.object_seg % fidxs[0])
            handarm_vis_dir = handarm_dir + "_vis"
            object_vis_dir = object_dir + "_vis"
            if not os.path.exists(handarm_dir):
                os.makedirs(handarm_dir)
            if not os.path.exists(object_dir):
                os.makedirs(object_dir)
            if not os.path.exists(handarm_vis_dir):
                os.makedirs(handarm_vis_dir)
            if not os.path.exists(object_vis_dir):
                os.makedirs(object_vis_dir)
            for b, fidx in enumerate(fidxs):
                handarm_seg_path = self.opt.cached.handarm_seg % fidx
                object_seg_path = self.opt.cached.object_seg % fidx
                handarm_vis_path = os.path.join(handarm_vis_dir, os.path.basename(handarm_seg_path))
                object_vis_path = os.path.join(object_vis_dir, os.path.basename(object_seg_path))
                Image.fromarray(handarm_segs[b].astype(np.uint8)).save(handarm_seg_path)
                Image.fromarray(object_segs[b].astype(np.uint8)).save(object_seg_path)
                Image.fromarray(handarm_vis[b].astype(np.uint8)).save(handarm_vis_path)
                Image.fromarray(object_vis[b].astype(np.uint8)).save(object_vis_path)
        handarm_segs = torch.tensor(handarm_segs).to(self.device)
        object_segs = torch.tensor(object_segs).to(self.device)

        # handarm SAM
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
            raise ValueError

        # object SAM
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
            raise ValueError

        # handarm + object
        inpaint_masks = torch.zeros_like(handarm_segs)
        inpaint_masks[handarm_masks > 0] = 255
        inpaint_masks[object_masks > 0] = 255
        inpaint_masks = dilate_mask(inpaint_masks.cpu().numpy(), self.opt.mask_dilate_size)

        # LAMA
        inpainted_images = []
        for b in range(batch_size):
            inpainted_images.append(self.lama(images[b], inpaint_masks[b]))
        inpainted_images = np.stack(inpainted_images, axis=0)

        return inpainted_images

    def egohos(self, images): 
        batch_size = images.shape[0]
        hand_segs = []
        cbs = []
        object_segs = []
        for b in range(batch_size):
            hand_seg = inference_segmentor(self.hand_segmentor, images[b])[0]
            cb = inference_segmentor(self.cb_segmentor, images[b], twohands=hand_seg)[0]
            object_seg = inference_segmentor(self.object_segmentor, images[b], twohands=hand_seg, cb=cb)[0]
            hand_segs.append(hand_seg)
            cbs.append(cb)
            object_segs.append(object_seg)
        hand_seg = np.stack(hand_segs, axis=0)
        cb = np.stack(cbs, axis=0)
        object_seg = np.stack(object_segs, axis=0)
        return hand_seg, object_seg
    
    def visualize_egohos(self, images, handarm_segs, object_segs):
        handarm_vis = visualize_twohands(images, handarm_segs)
        object_vis = handarm_segs.copy()
        object_vis[object_segs == 1] = 3
        object_vis[object_segs == 2] = 4
        object_vis[object_segs == 3] = 5
        object_vis = visualize_twohands_obj1(images, object_vis)
        return handarm_vis, object_vis
    
    def sam(self, image, point):
        masks, _, _ = predict_masks_with_sam(
            image,
            point,
            torch.tensor([1], dtype=torch.int),
            model_type=self.opt.sam.model_type,
            ckpt_p=self.opt.sam.ckpt,
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
        )
        return img_inpainted

    @staticmethod
    def barycenter(mask):
        y_indices, x_indices = torch.where(mask)
        x_center = torch.mean(x_indices.float())
        y_center = torch.mean(y_indices.float())

        return x_center, y_center
