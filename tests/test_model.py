import unittest

import torch

from src.model import AttentionHead, MultiheadAttention


class TestAttentionHead(unittest.TestCase):

    def _assert_attn_head_shape(self, out):
        self.assertEqual(out.shape[0], self.d_b)
        self.assertEqual(out.shape[1], self.seq_len)
        self.assertEqual(out.shape[2], self.d_v)

    def test_shape(self):
        self.d_model = 512
        self.seq_len = 8192
        self.d_b = 4
        self.d_k = 128
        self.d_v = 64

        x = torch.zeros(self.d_b, self.seq_len, self.d_model)
        attn = AttentionHead(
            d_model=self.d_model, d_k=self.d_k, d_v=self.d_v, seq_len=self.seq_len
        )

        out = attn(x)

        self._assert_attn_head_shape(out)

    def test_simple(self):
        self.d_model = 4
        self.seq_len = 4
        self.d_b = 1
        self.d_k = 2
        self.d_v = 3

        x = torch.tensor(
            [[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]],
            dtype=torch.float32,  # forcibly assign dtype for matmul compatibility
        )
        attn = AttentionHead(
            d_model=self.d_model, d_k=self.d_k, d_v=self.d_v, seq_len=self.seq_len
        )

        # manually instantiate weights
        with torch.no_grad():
            # Looks like Linear takes a transposed matrix?
            W_Q = torch.tensor([[1, 1, 1, 1], [0, 0, 0, 0]])
            W_K = torch.tensor([[1, 1, 1, 1], [0, 0, 0, 0]])
            W_V = torch.tensor([[1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]])

            attn.W_Q.weight.copy_(W_Q)
            attn.W_K.weight.copy_(W_K)
            attn.W_V.weight.copy_(W_V)

        out = attn(x)

        self._assert_attn_head_shape(out)
        expected = torch.tensor([[[1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]]])
        self.assertTrue(torch.equal(out, expected))


class TestMultiheadAttention(unittest.TestCase):

    def test_shape(self):
        self.d_model = 512
        self.seq_len = 8192
        self.d_b = 4
        self.d_k = 128
        self.d_v = 64
        self.n_heads = 8

        x = torch.zeros(self.d_b, self.seq_len, self.d_model)
        attn = MultiheadAttention(
            d_model=self.d_model,
            d_k=self.d_k,
            d_v=self.d_v,
            n_heads=self.n_heads,
            seq_len=self.seq_len,
        )

        out = attn(x)

        self.assertEqual(out.shape[0], self.d_b)
        self.assertEqual(out.shape[1], self.seq_len)
        self.assertEqual(out.shape[2], self.d_model)

    def test_simple(self):
        self.d_model = 4
        self.seq_len = 4
        self.d_b = 1
        self.d_k = 2
        self.d_v = 3
        self.n_heads = 2

        x = torch.tensor(
            [[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]],
            dtype=torch.float32,  # forcibly assign dtype for matmul compatibility
        )
        attn = MultiheadAttention(
            d_model=self.d_model,
            d_k=self.d_k,
            d_v=self.d_v,
            n_heads=self.n_heads,
            seq_len=self.seq_len,
        )

        with torch.no_grad():
            for i in range(self.n_heads):
                W_Q = torch.tensor([[1, 1, 1, 1], [0, 0, 0, 0]])
                W_K = torch.tensor([[1, 1, 1, 1], [0, 0, 0, 0]])
                W_V = torch.tensor([[1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]])

                attn.heads[i].W_Q.weight.copy_(W_Q)
                attn.heads[i].W_K.weight.copy_(W_K)
                attn.heads[i].W_V.weight.copy_(W_V)

            W_O = torch.tensor(
                [
                    [1, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0],
                ]
            )
            attn.W_O.weight.copy_(W_O)

        out = attn(x)

        self.assertEqual(out.shape[0], self.d_b)
        self.assertEqual(out.shape[1], self.seq_len)
        self.assertEqual(out.shape[2], self.d_model)

        expected = torch.tensor(
            [[[1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]]]
        )
        self.assertTrue(torch.equal(out, expected))


if __name__ == "__main__":
    unittest.main()
