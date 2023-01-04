import jax

print('JAX version:', jax.__version__)
print('Devices:', jax.devices())
print('Default backend:', jax.default_backend())
