from implementation.plumbing import sequence_to_clean_tokens

from numpy.testing import assert_equal # NOTE Numpy Testing is weird an adheres to ACTUAL, EXPECTED pattern

class TestSequenceToCleanWordTokens():

    def test_empty_string(self):
        assert [] == sequence_to_clean_tokens("")

    def test_clean_sequence(self):
        seq = "This sentence has no puncuation"
        tokens = sequence_to_clean_tokens(seq)

        assert_equal(tokens, ['this', 'sentence', 'has', 'no', 'puncuation'])

    def test_remove_tabs(self):
        seq = "This sentence has    a tab in it."
        tokens = sequence_to_clean_tokens(seq)

        assert '\t' not in tokens
        assert all(x not in ['', ' ', '\t'] for x in tokens)

    def test_sequences(self):
        seq = "It was raining a lot on Macy's birthday"
        tokens = sequence_to_clean_tokens(seq)

        assert_equal(tokens, ['it', 'was', 'raining', 'a', 'lot', 'on', 'macy', "'", 's', 'birthday'])
