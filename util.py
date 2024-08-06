from web3 import AsyncWeb3
from web3.middleware.async_cache import async_construct_simple_cache_middleware
import os

from asyncio import gather
from typing import Callable, Iterable

# SOMEDAY zkevm. Infura doesn't even support it. (I don't need zkevm web3 now though)
WEB3_URLS = {
    "ethereum": "https://mainnet.infura.io/v3/{infura_apikey}",
    "arbitrum": "https://arbitrum-mainnet.infura.io/v3/{infura_apikey}",
}


async def get_web3(chain) -> AsyncWeb3:
    apikeys = {
        "infura_apikey": os.environ["INFURA_APIKEY"],
    }

    provider = AsyncWeb3.AsyncHTTPProvider(WEB3_URLS[chain].format(**apikeys))
    w3 = AsyncWeb3(provider)

    # SOMEDAY we could add a load limiter / automatic retry here. Doesn't seem needed for now.

    # Prevent web3.py from making a chainid request together with every eth_call request
    # (so we'd have *twice* the requests).
    # See https://ethereum.stackexchange.com/questions/131768/how-to-reduce-the-number-of-eth-chainid-calls-when-using-web3-python
    cache_chain_id_middleware = await async_construct_simple_cache_middleware(
        rpc_whitelist=[
            "web3_clientVersion",
            "net_version",
            "eth_chainId",
        ]  # type: ignore
    )
    w3.middleware_onion.add(cache_chain_id_middleware, name="Cache chain_id")

    return w3


async def gather_dict(cors: dict) -> dict:
    """
    Gather a dict where values are coroutines into their results.
    """
    return dict(zip(cors.keys(), await gather(*cors.values())))


async def gather_sync(*cors) -> tuple:
    """
    Hack / debugging tool that can be used in place of gather().
    """
    return tuple([await c for c in cors])


async def gather_dict_sync(cors: dict) -> dict:
    """
    Debugging utility
    """
    return dict(zip(cors.keys(), await gather_sync(*cors.values())))


async def async_map_dict(af: Callable, inputs: Iterable) -> dict:
    """
    Dict of mapping the async function `af` over `inputs`.

    Duplicate values in `inputs` are lost.
    """
    return await gather_dict({i: af(i) for i in inputs})


async def apure(x):
    """Wrap a pure value into an async coroutine."""
    return x


def plain_dict(d, lvl=float("inf")):
    """
    Transform a nested dict into a 1-level dict with keys of form (key1, key2, ...). If lvl is given, only do this that many levels.
    """
    if not isinstance(d, dict):
        return d
    if lvl < 1:
        return d

    ret = {}
    for k, v in d.items():
        v_plain = plain_dict(v, lvl - 1)
        if not isinstance(v_plain, dict):
            ret[k] = v_plain
            continue
        for k1, v_ret in v_plain.items():
            k_ret = (k, *k1) if isinstance(k1, tuple) else (k, k1)
            ret[k_ret] = v_ret
    return ret
