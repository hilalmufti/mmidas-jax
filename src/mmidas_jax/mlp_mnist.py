from jax import grad, jit, vmap, random
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from jax.tree_util import tree_map
import numpy as np
from torch.utils import data 
from torchvision.datasets import MNIST
from pyrsistent import pmap, pvector

def random_layer_params(m, n, k, scale=1e-2):
  kw, kb = random.split(k)
  return scale * random.normal(kw, (n, m)), scale * random.normal(kb, (n,))

def init_network_params(szs, k):
  ks = random.split(k, len(szs))
  return [random_layer_params(m, n, k) for m, n, k in zip(szs[:-1], szs[1:], ks)]

def relu(x):
  return jnp.maximum(0, x)

def predict(params, image):
  activations = image
  for w, b in params[:-1]:
    outputs = jnp.dot(w, activations) + b
    activations = relu(outputs)
  final_w, final_b = params[-1]
  logits = jnp.dot(final_w, activations) + final_b
  return logits - logsumexp(logits)

def one_hot(x, k, dtype=jnp.float32):
  return jnp.array(x[:, None] == jnp.arange(k), dtype)

def accuracy(params, images, targets):
  target_class = jnp.argmax(targets, axis=1)
  predicted_class = jnp.argmax(batched_predict(params, images), axis=1)
  return jnp.mean(predicted_class == target_class)

# TODO: what is this loss function
def loss(params, images, targets):
  preds = batched_predict(params, images)
  return -jnp.mean(preds * targets)

@jit
def update(params, x, y, step_size=1e-3): 
  grads = grad(loss)(params, x, y)
  return [(w - step_size * dw, b - step_size * db)
          for (w, b), (dw, db) in zip(params, grads)]

batched_predict = vmap(predict, in_axes=(None, 0))

SEED = 546
k = random.key(546)
szs = [784, 512, 512, 10]
params = init_network_params(szs, k)

rand_img = random.normal(random.key(547), (28 * 28,))
rand_imgs = random.normal(random.key(547), (10, 28 * 28,))

def numpy_collate(batch):
  return tree_map(np.asarray, data.default_collate(batch))