import unittest
import zlib

from spa_control.bluetooth_camera import FrameParser, HEADER, MAGIC, MAX_JPEG


def packet(jpeg=b"\xff\xd8hello\xff\xd9", sequence=5):
    return HEADER.pack(MAGIC, 42, sequence, 1234567, len(jpeg), zlib.crc32(jpeg)) + jpeg


class CameraParserTests(unittest.TestCase):
    def test_partial_header_and_payload_with_boot_noise(self):
        parser = FrameParser()
        data = b"rst:0x1\nSPA_CAM camera=OK\n" + packet()
        result = []
        for byte in data:
            result.extend(parser.feed(bytes([byte])))
        self.assertEqual(len(result), 1)
        self.assertEqual((result[0].boot_id, result[0].sequence, result[0].capture_us),
                         (42, 5, 1234567))
        self.assertEqual(result[0].jpeg, b"\xff\xd8hello\xff\xd9")

    def test_corrupt_crc_resynchronizes(self):
        bad = bytearray(packet())
        bad[-3] ^= 1
        parser = FrameParser()
        result = parser.feed(bad + packet(sequence=6))
        self.assertEqual([frame.sequence for frame in result], [6])
        self.assertEqual(parser.rejected, 1)

    def test_oversized_length_does_not_allocate_payload(self):
        parser = FrameParser()
        result = parser.feed(HEADER.pack(MAGIC, 0, 0, 0, MAX_JPEG + 1, 0) + packet())
        self.assertEqual(len(result), 1)
        self.assertLess(len(parser.buffer), HEADER.size)

    def test_noise_is_bounded(self):
        parser = FrameParser()
        parser.feed(b"a" * 200000)
        self.assertLess(len(parser.buffer), len(MAGIC))

    def test_non_jpeg_and_multiple_frames(self):
        parser = FrameParser()
        result = parser.feed(packet(b"not-a-jpeg") + packet(sequence=7) + packet(sequence=8))
        self.assertEqual([frame.sequence for frame in result], [7, 8])
        self.assertEqual(parser.rejected, 1)


if __name__ == "__main__":
    unittest.main()
