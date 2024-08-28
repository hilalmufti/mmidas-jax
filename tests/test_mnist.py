import os

from expecttest import assert_expected_inline
import jax
from jax import random
import jax.numpy as jnp

from mmidas_jax.mnist import mk_linear, mk_fc, relu

def inc(x):
    return x + 1

def test_jax_devices():
    assert_expected_inline(str(jax.devices()), """[METAL(id=0)]""")

def test_pjrt():
    assert_expected_inline(str(os.environ["ENABLE_PJRT_COMPATIBILITY"]), """1""")

def test_answer():
    assert_expected_inline(str(inc(4)), """5""")

def test_split():
    s = 'hello world'
    assert_expected_inline(str(s.split()), """['hello', 'world']""")

def test_jnp():
    assert_expected_inline(str(jnp.array([1, 2, 3])), """[1 2 3]""")

def test_mk_linear0():
    key = random.key(0)
    assert_expected_inline(str(mk_linear(3, 3, key)), """\
(Array([[-0.02610558,  0.00033853,  0.01086333],
       [-0.01480299,  0.01540325,  0.01062516],
       [ 0.00541748,  0.00017023,  0.00272268]], dtype=float32), Array([ 0.01137878, -0.01220955, -0.00591536], dtype=float32))""")
    assert_expected_inline(str(mk_linear(1, 3, key)), """\
(Array([[ 0.00138932],
       [ 0.00509335],
       [-0.00531161]], dtype=float32), Array([ 0.01137878, -0.01220955, -0.00591536], dtype=float32))""")
    assert_expected_inline(str(mk_linear(3, 1, key)), """(Array([[ 0.00138932,  0.00509335, -0.00531161]], dtype=float32), Array([-0.01251539], dtype=float32))""")

def test_mk_linear546():
    key = random.key(546)
    assert_expected_inline(str(mk_linear(3, 3, key)), """\
(Array([[ 0.00455107, -0.00145467, -0.01521307],
       [ 0.00368185, -0.00640634, -0.01229068],
       [ 0.00905397, -0.00399138,  0.00254749]], dtype=float32), Array([0.00789843, 0.00347356, 0.0058376 ], dtype=float32))""")
    assert_expected_inline(str(mk_linear(1, 3, key)), """\
(Array([[ 0.00018403],
       [-0.00152481],
       [-0.00812268]], dtype=float32), Array([0.00789843, 0.00347356, 0.0058376 ], dtype=float32))""")
    assert_expected_inline(str(mk_linear(3, 1, key)), """(Array([[ 0.00018403, -0.00152481, -0.00812268]], dtype=float32), Array([-0.00778186], dtype=float32))""")

def test_mk_fc():
    key = random.key(546)
    assert_expected_inline(str(mk_fc([4, 2, 2, 1], key)), """\
[(Array([[ 0.00938669,  0.02961616,  0.00263324, -0.00594663],
       [ 0.00518566,  0.00423364, -0.00554846, -0.01303717]],      dtype=float32), Array([0.00235619, 0.01016501], dtype=float32)), (Array([[ 0.01997421,  0.00094804],
       [ 0.00542988, -0.0088178 ]], dtype=float32), Array([0.03150228, 0.0044116 ], dtype=float32)), (Array([[ 0.00888148, -0.00981103]], dtype=float32), Array([0.0124264], dtype=float32))]""")

def test_relu():
    assert_expected_inline(str(relu(jnp.array([1, -2, 3]))), """[1 0 3]""")

def test_random_img():
    assert_expected_inline(str(random.normal(random.key(1), (2 * 2,))), """[ 0.17269018 -0.46084192  1.2229712  -0.07101909]""")