"""自适应局部阈值的回归测试。"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image, ImageDraw  # noqa: E402

from server.algorithms import color, segmentation  # noqa: E402


def high_contrast_image():
    img = Image.new("RGB", (120, 120), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([30, 30, 80, 80], fill=(0, 0, 0))
    return img


class AdaptiveThresholdTest(unittest.TestCase):
    def test_adaptive_threshold_contains_black_and_white(self):
        result = color.threshold(high_contrast_image(), {"mode": "adaptive", "block": 15})

        histogram = result.histogram()
        self.assertGreater(histogram[0], 0)
        self.assertGreater(histogram[255], 0)
        self.assertEqual(histogram[0] + histogram[255], 120 * 120)

    def test_region_segmentation_does_not_cover_entire_image(self):
        result = segmentation.segment(high_contrast_image(), {"method": "region", "block": 15})

        self.assertGreaterEqual(result["region_count"], 1)
        self.assertLess(result["coverage"], 0.99)


if __name__ == "__main__":
    unittest.main()
