# %%
import os
os.environ["ENABLE_PJRT_COMPATIBILITY"] = str(1)

import math
from functools import reduce

from jax import grad, jit, vmap, random
import jax.numpy as jnp
from jax.scipy.special import logsumexp

SEED = 546
key = random.key(SEED)

# %%
def starreduce(fun, seq, init):
    for x in iter(seq):
        init = fun(init, *x)
    return init

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
def linear(x, W, b):
    return jnp.dot(W, x) + b

# %%
def relu(x):
    return jnp.maximum(0, x)

relu(jnp.array([1, -2, 3]))

# %%
def log_softmax(x):
    return x - logsumexp(x)

# %%
def predict(params, img):
    activations = starreduce(lambda acc, W, b: relu(jnp.dot(W, acc) + b), params[:-1], img)
    W, b = params[-1]
    return log_softmax(jnp.dot(W, activations) + b)

# %%
def one_hot(x, k: int, dtype=jnp.float32):
    def mapten(x):
        return x[:, None]
    assert len(x.shape) == 1
    return jnp.array(mapten(x) == jnp.arange(k), dtype)

# %%
def accuracy(params, imgs, targets):
    target_class = jnp.argmax(targets, axis=-1)
    predicted_class = jnp.argmax(batched_predict(params, imgs), axis=1)
    return jnp.mean(predicted_class == target_class)

# %%
def loss(params, imgs, targets):
    preds = batched_predict(params, imgs)
    return -jnp.mean(preds * targets)

def update(params, x, y):
    grads = grad(loss)(params, x, y)
    return [(w - step_size * dw, b - step_size * db)
            for (w, b), (dw, db) in zip(params, grads)]

# update(params, rand_img, predict(params, rand_img))

# %%

sizes = [784, 512, 512, 10]
lr = 0.01
epochs = 8
batch_size = 128
n_targets = 10
params = mk_fc(sizes, key)

rand_img = random.normal(random.key(SEED + 1), (28 * 28,))
rand_imgs = random.normal(random.key(SEED + 1), (10, 28 * 28))

batched_predict = vmap(predict, in_axes=(None, 0))

batched_predict(params, rand_imgs)