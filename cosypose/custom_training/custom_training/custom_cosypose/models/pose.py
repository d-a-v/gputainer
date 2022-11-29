import torch

from cosypose.lib3d.camera_geometry import get_K_crop_resize, boxes_from_uv

from cosypose.lib3d.cropping import deepim_crops_robust as deepim_crops
from cosypose.lib3d.camera_geometry import project_points_robust as project_points

from cosypose.models.pose import PosePredictor

from custom_training.custom_cosypose.custom_config import CUSTOM_CONFIG

from cosypose.utils.logging import get_logger
logger = get_logger(__name__)


class CustomPosePredictor(PosePredictor):
    def crop_inputs(self, images, K, TCO, labels):
        bsz, nchannels, h, w = images.shape
        assert K.shape == (bsz, 3, 3)
        assert TCO.shape == (bsz, 4, 4)
        assert len(labels) == bsz
        meshes = self.mesh_db.select(labels)
        points = meshes.sample_points(
            min(2000, CUSTOM_CONFIG['max_sampled_npoints']), deterministic=True)
        uv = project_points(points, K, TCO)
        boxes_rend = boxes_from_uv(uv)
        boxes_crop, images_cropped = deepim_crops(
            images=images, obs_boxes=boxes_rend, K=K,
            TCO_pred=TCO, O_vertices=points, output_size=self.render_size, lamb=1.4
        )
        K_crop = get_K_crop_resize(K=K.clone(), boxes=boxes_crop,
                                   orig_size=images.shape[-2:], crop_resize=self.render_size)
        if self.debug:
            self.tmp_debug.update(
                boxes_rend=boxes_rend,
                rend_center_uv=project_points(
                    torch.zeros(bsz, 1, 3).to(K.device), K, TCO),
                uv=uv,
                boxes_crop=boxes_crop,
            )
        return images_cropped, K_crop.detach(), boxes_rend, boxes_crop
