import unittest
import numpy as np
from onnx_detector import decode_predictions


class DecodeTests(unittest.TestCase):
    def test_agnostic_nms_and_letterbox_coordinates(self):
        output = np.zeros((1, 7, 3), np.float32)
        # Two overlapping faces with competing labels, plus a separate face.
        output[0, :, 0] = [100, 140, 80, 80, .9, .01, .01]
        output[0, :, 1] = [101, 140, 80, 80, .01, .8, .01]
        output[0, :, 2] = [240, 140, 60, 60, .01, .01, .7]
        names = {0: "with_mask", 1: "without_mask", 2: "mask_weared_incorrect"}
        result = decode_predictions(output, names, (160, 320, 3), 1, (0, 80), .35)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["box"], [60, 20, 140, 100])
        self.assertEqual(result[0]["label"], "with_mask")
        self.assertEqual(result[1]["label"], "mask_weared_incorrect")

    def test_empty_low_confidence_and_invalid_predictions(self):
        names = {0: "mask"}
        output = np.zeros((1, 5, 2), np.float32)
        output[0, :, 0] = [20, 20, 10, 10, .1]
        output[0, :, 1] = [float("nan"), 20, 10, 10, .9]
        self.assertEqual(decode_predictions(output, names, (320, 320, 3), 1, (0, 0), .35), [])
        with self.assertRaises(ValueError):
            decode_predictions(np.zeros((1, 6, 10)), names, (320, 320, 3), 1, (0, 0), .35)

    def test_scaled_boxes_clipped_to_source(self):
        output = np.array([[[10], [10], [100], [100], [.9]]], dtype=np.float32)
        result = decode_predictions(output, {0: "mask"}, (100, 100, 3), 2, (0, 0), .35)
        self.assertEqual(result[0]["box"], [0, 0, 30, 30])


if __name__ == "__main__":
    unittest.main()
