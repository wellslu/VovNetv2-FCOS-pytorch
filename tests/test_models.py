import unittest
importnumpy as np
import torch

import src
from src.models import VoVNetV2_FCOS


class TestModels(unittest.TestCase):

    def test_vgg(self):
        m = VoVNetV2_FCOS()
        bbox_util = src.utils.DecodeBox([8, 16, 32, 64, 128])
        img_shape = (512, 512)
        x = torch.randn(1, 3, 512, 512)
        with torch.no_grad():
            y = m(x)
        outputs = bbox_util.decode_box(y, (512, 512))
        results = bbox_util.non_max_suppression(outputs, (512, 512), img_shape, True, conf_thres = 0.5, nms_thres = 0.3)[0]
        if results is not None:
            locs, confs, clss = results[:, :4], results[:, 4], results[:, 5]

        self.assertListEqual(list(y.size()), [1, 10])


if __name__ == '__main__':
    unittest.main()
