class BasicFormatter:
    def __init__(self, PLM_vocab, query_len, cand_len, mode):
        self.PLM_vocab = PLM_vocab
        self.query_len = query_len
        self.cand_len = cand_len
        self.mode = mode

    def process(self, data, mode):
        return data

