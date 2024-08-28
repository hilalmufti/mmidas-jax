from expecttest import assert_expected_inline

import jax.numpy as jnp

def inc(x):
    return x + 1

def test_answer():
    assert_expected_inline(str(inc(4)), """5""")

def test_split():
    s = 'hello world'
    assert_expected_inline(str(s.split()), """['hello', 'world']""")

def test_jnp():
    assert_expected_inline(str(jnp.array([1, 2, 3])), """[1 2 3]""")