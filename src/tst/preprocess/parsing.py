def find_anker_words(ref_vocab, target_vocab, tol=.1, min_freq=5e-5):
    out = {}
    for k, v in target_vocab.items():
        if ref_vocab.get(k) and ref_vocab[k] > min_freq and target_vocab[k] > min_freq:
            diff = abs((ref_vocab[k] - target_vocab[k]) / ref_vocab[k])
            if diff < tol:
                out[k] = diff
    return out

