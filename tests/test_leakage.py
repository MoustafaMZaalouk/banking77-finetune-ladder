"""Leakage-focused tests: nothing about the test split may reach training.

BANKING77's documented 6 duplicate texts across train/test are pinned in
tests/test_data.py. These tests cover the leakage the code could
introduce on top of that: a vocabulary or a fit that saw test data.
"""


from src.rung_tfidf import train_tfidf_lr


def test_tfidf_vocabulary_comes_from_train_only():
    train_texts = ["card lost again", "balance please", "card balance"]
    test_texts = ["zzunique token never trained"]
    vectorizer, classifier = train_tfidf_lr(train_texts, [0, 1, 0])
    for token in ("zzunique", "never"):
        assert token not in vectorizer.vocabulary_
    # and the pipeline still predicts on test text (OOV handled)
    assert len(classifier.predict(vectorizer.transform(test_texts))) == 1


def test_tfidf_refits_from_scratch_each_call():
    # No shared state between runs: a second fit on other data must not
    # retain the first vocabulary.
    first = train_tfidf_lr(["apple pear", "apple"], [0, 1])[0]
    second = train_tfidf_lr(["plum cherry", "plum"], [0, 1])[0]
    assert "apple" not in second.vocabulary_
    assert set(first.vocabulary_).isdisjoint(second.vocabulary_)


def test_full_finetune_rung_trains_every_weight():
    # The frozen-vs-trained distinction is load-bearing in this repo:
    # rung 5 must not quietly freeze anything.
    import inspect

    from src import rung_fullft
    source = inspect.getsource(rung_fullft)
    assert "requires_grad" not in source
    assert "body.parameters()" in source
    assert "head.parameters()" in source


def test_frozen_rung_has_no_fit_path():
    # The frozen encoder wrapper only exposes encode/transform behavior;
    # there is deliberately no API that updates its weights.
    from src import rung_frozen

    assert not hasattr(rung_frozen, "fit_encoder")
    import inspect
    source = inspect.getsource(rung_frozen)
    assert "backward" not in source
    assert ".train(" not in source
