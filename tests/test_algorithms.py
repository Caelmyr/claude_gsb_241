import os
import sys
import unittest

from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.algorithms import color, segmentation


def square_image(fg=20, bg=240, size=120):
    img = Image.new("RGB", (size, size), (bg, bg, bg))
    draw = ImageDraw.Draw(img)
    margin = size // 4
    draw.rectangle([margin, margin, size - margin - 1, size - margin - 1],
                   fill=(fg, fg, fg))
    return img


class ThresholdTests(unittest.TestCase):
    def test_adaptive_keeps_dark_shape_on_light_background(self):
        img = square_image(fg=20, bg=240)
        result = color.threshold(img, {"mode": "adaptive", "value": 127, "block": 15})

        self.assertEqual(result.getpixel((10, 10)), 255)
        self.assertEqual(result.getpixel((60, 60)), 0)
        self.assertIn(0, set(result.getdata()))
        self.assertIn(255, set(result.getdata()))

    def test_adaptive_keeps_light_shape_on_dark_background(self):
        img = square_image(fg=240, bg=20)
        result = color.threshold(img, {"mode": "adaptive", "value": 127, "block": 15})

        self.assertEqual(result.getpixel((10, 10)), 0)
        self.assertEqual(result.getpixel((60, 60)), 255)
        self.assertIn(0, set(result.getdata()))
        self.assertIn(255, set(result.getdata()))


class RegionGrowingTests(unittest.TestCase):
    def test_high_contrast_shape_is_not_one_region(self):
        img = square_image(fg=20, bg=240)
        gray = img.convert("L")
        _, components = segmentation._region_grow(gray, 15)
        result = segmentation.segment(img, {"method": "region", "block": 15, "alpha": 0.45})
        overlay = result["image"].convert("RGB")

        self.assertGreaterEqual(len(components), 2)
        self.assertGreaterEqual(result["region_count"], 2)
        self.assertNotEqual(overlay.getpixel((10, 10)), overlay.getpixel((60, 60)))
        self.assertEqual(result["coverage"], 1.0)


if __name__ == "__main__":
    unittest.main()
