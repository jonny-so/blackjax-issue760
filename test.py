
import blackjax
import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy.stats as stats

loc, scale = 10, 20
observed = np.random.normal(loc, scale, size=1_000)

def logdensity_fn(loc, log_scale, observed=observed):
    """Univariate Normal"""
    scale = jnp.exp(log_scale)
    logjac = log_scale
    logpdf = stats.norm.logpdf(observed, loc, scale)
    return logjac + jnp.sum(logpdf)

def inference_loop(rng_key, kernel, initial_state, num_samples):
    @jax.jit
    def one_step(state, rng_key):
        state, _ = kernel(rng_key, state)
        return state, state

    keys = jax.random.split(rng_key, num_samples)
    _, states = jax.lax.scan(one_step, initial_state, keys)

    return states

rng_key = jax.random.PRNGKey(0)

logdensity = lambda x: logdensity_fn(**x)

initial_position = {"loc": 1.0, "log_scale": 1.0}

def f(rng_key):
    jax.debug.print('warming up')
    warmup = blackjax.window_adaptation(blackjax.nuts, logdensity, is_mass_matrix_diagonal=False)
    rng_key, warmup_key, sample_key = jax.random.split(rng_key, 3)
    (state, parameters), _ = warmup.run(warmup_key, initial_position, num_steps=200)

    kernel = blackjax.nuts(logdensity, **parameters).step
    jax.debug.print('warmed up, pos={}', state.position)
    states = inference_loop(sample_key, kernel, state, 100)

    mcmc_samples = states.position
    return mcmc_samples

_f = jax.jit(f).lower(rng_key).compile()
mcmc_samples = _f(rng_key)
