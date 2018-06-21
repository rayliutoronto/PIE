from __future__ import absolute_import

from PIE.tokenizer import Tokenizer

t = Tokenizer()
print("localtion::: {}", __file__)

def test_email():
    tokens = [token.text for doc in t.split('abc@toronto.ca') for token in doc]
    assert tokens == ['abc', '@', 'toronto.ca']


def test_phone():
    tokens = [token.text for doc in t.split('416-000-1234') for token in doc]
    assert tokens == ['416', '-', '000', '-', '1234']
