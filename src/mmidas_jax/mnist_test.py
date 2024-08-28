from expecttest import assert_expected_inline

def inc(x):
    return x + 1


def test_answer():
    assert_expected_inline(str(inc(4)), """5""")

def test_split():
    s = 'hello world'
    assert_expected_inline(str(s.split()), """['hello', 'world']""")