from networks.embedder import Embedder


class NerfLib:
    def __init__(self, conf):
        self.x_embedder = Embedder(conf['x_enc_count'])
        self.d_embedder = Embedder(conf['d_enc_count'])

    def embed_x(self, x):
        return self.x_embedder(x)

    def embed_d(self, d):
        return self.d_embedder(d)
