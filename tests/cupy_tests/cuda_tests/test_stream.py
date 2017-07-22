import unittest

from cupy import cuda
from cupy import testing
from cupy.testing import attr


class TestStream(unittest.TestCase):

    def test_init(self):
        with self.assertRaises(ValueError):
            cuda.Stream(null=True)

        self.assertEqual(None, cuda.Stream.null.device)
        with cuda.Device():
            stream = cuda.Stream()
            self.assertEqual(0, stream.device)

    @attr.gpu
    def test_get_and_add_callback(self):
        N = 100
        cupy_arrays = [testing.shaped_random((2, 3)) for _ in range(N)]

        stream = cuda.Stream.null
        out = []
        for i in range(N):
            numpy_array = cupy_arrays[i].get(stream=stream)
            stream.add_callback(
                lambda _, __, t: out.append(t[0]),
                (i, numpy_array))

        stream.synchronize()
        self.assertEqual(out, list(range(N)))

    @attr.gpu
    def test_with_statement(self):
        stream1 = cuda.Stream()
        stream2 = cuda.Stream()
        self.assertEqual(cuda.Stream.null, cuda.get_current_stream())
        with stream1:
            self.assertEqual(stream1, cuda.get_current_stream())
            with stream2:
                self.assertEqual(stream2, cuda.get_current_stream())
            self.assertEqual(stream1, cuda.get_current_stream())
        self.assertEqual(cuda.Stream.null, cuda.get_current_stream())

    @attr.gpu
    def test_use(self):
        stream1 = cuda.Stream().use()
        self.assertEqual(stream1, cuda.get_current_stream())
        cuda.Stream.null.use()
        self.assertEqual(cuda.Stream.null, cuda.get_current_stream())

    @testing.multi_gpu(2)
    def test_get_current_stream_on_another_device(self):
        with cuda.Device(0):
            stream0 = cuda.Stream()
            stream0.use()
        with self.assertRaises(ValueError):
            with cuda.Device(1):
                cuda.get_current_stream()
        # Stream.null is available on all devices
        cuda.Stream.null.use()
        with cuda.Device(1):
            cuda.get_current_stream()
