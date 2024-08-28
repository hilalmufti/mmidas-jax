# %%
import os
os.environ["ENABLE_PJRT_COMPATIBILITY"] = str(1)

import math

from jax import grad, jit, vmap, random
import jax.numpy as jnp
from jax.scipy.special import logsumexp

SEED = 546
key = random.key(SEED)

# %%
def mnist_fun(x):
    print("mnist fun called")
    return x + 2

# %%
def mk_linear(m, n, key, scale=1e-2):
    w_key, b_key = random.split(key)
    return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))

mk_linear(4, 6, key)

# %%
def mk_fc(sizes, key):
    keys = random.split(key, len(sizes))
    return [mk_linear(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]

mk_fc([2, 2, 3, 4, 5], key)

# %%
def relu(x):
    return jnp.maximum(0, x)

relu(jnp.array([1, -2, 3]))

# %%
def predict(params, image):
    activations = image

logsumexp(jnp.array([1, 2, 3]))

# %%

sizes = [784, 512, 512, 10]
lr = 0.01
epochs = 8
batch_size = 128
n_targets = 10
params = mk_fc(sizes, key)

random_flattened_image = random.normal(random.key(1), (2 * 2,))

jnp.dot(jnp.array([1, 2, 1, 1]),  random_flattened_image)

