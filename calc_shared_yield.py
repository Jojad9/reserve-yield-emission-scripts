import asyncio
from asyncio import gather
import aiohttp
import argparse
import json
from dotenv import load_dotenv
import os
import sys
import logging

from gql import Client as GqlClient, gql
from gql.transport.aiohttp import AIOHTTPTransport as GqlAIOHTTPTransport

from collections import Counter, defaultdict

from dune_client.client_async import AsyncDuneClient
from dune_client.models import ExecutionState as DuneExecutionState
from dune_client.query import QueryBase as DuneQueryBase

from web3 import AsyncWeb3

from jmespath import search as jsearch

import jsno

import pandas as pd
import numpy as np

from icecream import ic

from collections import defaultdict

from typing import TypeVar, cast

import yaml

from util import get_web3, async_map_dict, gather_dict, apure, gather_sync, plain_dict

import logging

# NB ic() logs at the DEBUG level here. Separate INFO level calls are used where it makes sense.
# Set this to DEBUG to get additional output.
loglevel = logging.INFO
logging.basicConfig(level=loglevel)
logger = logging.getLogger(__name__)
logger.setLevel(loglevel)
ic.configureOutput(outputFunction=logger.debug)


def log_info_dict(message: str, info: dict):
    structured_message = f"{message} | {json.dumps(jsno.jsonify(info))}"
    logger.info(structured_message)


# Minimum amount below which an emission is simply discarded.
# This is because, e.g., sGYD doesn't allow lower amounts, and also, it's just a bit ridiculous.
MIN_EMISSION_AMOUNT = 1.0

# See: https://docs.balancer.fi/reference/subgraph/
# Production subgraph addresses. `thegraph_apikey` will be replaced from env.
BALANCER_SUBGRAPHS = {
    "ethereum": "https://gateway-arbitrum.network.thegraph.com/api/{thegraph_apikey}/subgraphs/id/C4ayEZP2yTXRAB8vSaTrgN4m9anTe9Mdm2ViyiAuV9TV",
    "arbitrum": "https://gateway-arbitrum.network.thegraph.com/api/{thegraph_apikey}/subgraphs/id/98cQDy6tufTJtshDCuhh9z2kWXsQWBHVh2bqnLHsGAeS",
    "polygon-zkevm": "https://api.studio.thegraph.com/query/24660/balancer-polygon-zk-v2/version/latest",
}

# Also see the above link.
# For some reason, gauge information is on a separate subgraph.
BALANCER_GAUGE_SUBGRAPHS = {
    "ethereum": "https://gateway-arbitrum.network.thegraph.com/api/{thegraph_apikey}/subgraphs/id/4sESujoqmztX6pbichs4wZ1XXyYrkooMuHA8sKkYxpTn",
    "arbitrum": "https://gateway-arbitrum.network.thegraph.com/api/{thegraph_apikey}/subgraphs/id/Bb1hVjJZ52kL23chZyyGWJKrGEg3S6euuNa1YA6XRU4J",
    "polygon-zkevm": "https://api.studio.thegraph.com/query/24660/balancer-gauges-polygon-zk/version/latest",
}

# Max rows returned by the graph
# THEGRAPH_LIMIT = 100

DUNE_QUERY_ID_EXCESS_RESERVE = 3904505

# Same on all chains
BALANCER_VAULT = "0xBA12222222228d8Ba445958a75a0704d566BF2C8"

T = TypeVar("T")


def unwrap(x: T | None) -> T:
    """Typing stuff"""
    assert x is not None
    return x


def unique_row(df: pd.DataFrame) -> pd.Series:
    assert len(df) > 0
    assert len(df) <= 1
    return df.iloc[0]


def deannualize(yield_: float, period: pd.Timedelta) -> float:
    return (1 + yield_) ** (period.days / 365) - 1


def annualize(yield_: float, period: pd.Timedelta) -> float:
    return (1 + yield_) ** (365 / period.days) - 1


async def gql_execute_paginated(session, query, variable_values=None, limit=1) -> dict:
    """
    Pull a graphql query with automatic skip-based pagination.

    We are not being "smart" about this at all and this only works when the query looks like this:

    1. It pulls a single list, and this is the thing you wanna paginate.
    2. It uses a stable ordering where new records are only appended at the end.
    3. It has a parameter $skip: Int! that skips records.

    Parameters:

    - session: graphql session from `async with GqlClient(...)`
    - query: graphql query from gql()
    - variable_values: variables values for your query, except $skip.
    - limit > 0: min number of records returned if there is more data. The default means "unknown" and does one more query in total than otherwise necessary.
    """
    if variable_values is None:
        variable_values = {}

    skip = 0
    result_key = None
    inners = []

    while True:
        ic(skip)
        result = await session.execute(
            query, variable_values=variable_values | {"skip": skip}
        )
        ((result_key, inner),) = list(result.items())
        ic(inner)

        inners.extend(inner)

        if len(inner) < limit:
            return {result_key: inners}

        skip += len(inner)


async def get_pool_balance_events(
    chain: str, pool: str, config: dict
) -> tuple[float, pd.DataFrame]:
    ic("get_pool_balance_events", chain, pool)

    url = BALANCER_SUBGRAPHS[chain].format(**config["apikeys"])

    # timeout: it's not the fastest thing in the world and we're not being careful not to saturate either.
    # (this turns out to be way faster in practice, so probably not necessary)
    transport = GqlAIOHTTPTransport(url=url, timeout=60 * 3)
    async with GqlClient(transport=transport, execute_timeout=30) as session:
        # First, get some pool metadata
        query = gql(
            """
            query getPoolMetadata ($pool_address: Bytes!) {
              pools (where:{address: $pool_address}) {
                id
                createTime
                tokens {
                    symbol
                }
              }
            }
            """
        )
        result = await session.execute(
            query,
            variable_values={
                "pool_address": pool,
            },
        )
        (pool_result,) = result["pools"]

        pool_id = pool_result["id"]
        (ix_token_gyd,) = [
            i for i, tkn in enumerate(pool_result["tokens"]) if tkn["symbol"] == "GYD"
        ]
        pool_create_time = pd.to_datetime(pool_result["createTime"], unit="s", utc=True)

        # Now collect the actual data: joins/exits, swaps, and initial balance.
        # We build up `cor_*` coroutine variables and gather them below.

        variable_values = {
            "pool_id": pool_id,
            "start_block": config["start_blocks"][chain],
            "end_block_plus_1": config["end_blocks"][chain] + 1,
        }

        query_joinExits = gql(
            """
            query getJoinsExits ($pool_id: String!, $start_block: BigInt!, $end_block_plus_1: BigInt!, $skip: Int!) {
                joinExits(where: {pool: $pool_id, block_gt: $start_block, block_lt: $end_block_plus_1}, orderBy: block, skip: $skip) {
                    type
                    block
                    timestamp
                    amounts
                }
            }  
            """
        )
        cor_joinExits = gql_execute_paginated(session, query_joinExits, variable_values)

        query_swaps = gql(
            """
            query getSwaps ($pool_id: String!, $start_block: BigInt!, $end_block_plus_1: BigInt!, $skip: Int!) {
                swaps(where: {poolId: $pool_id, block_gt: $start_block, block_lt: $end_block_plus_1}, orderBy: block, skip: $skip) {
                  tokenInSym
                  tokenOutSym
                  tokenAmountIn
                  tokenAmountOut
                  block
                  timestamp
                }
            }  
            """
        )
        cor_swaps = gql_execute_paginated(session, query_swaps, variable_values)

        # For new pools, there may be no data at the start_block. In that case, we actually also don't need to query any data: the starting balance is 0.
        # NOTE there's some imprecision here if the pool is launched & liquidity is immediately added in the same or a very close block. But we don't do that.
        if pool_create_time < config["start_time"]:
            query_init_gyd = gql(
                """
                query PoolBalanceAtBlock($pool_id: ID!, $start_block: Int!) {
                  pool(id: $pool_id, block: {number: $start_block}) {
                    tokens {
                      balance
                    }
                  }
                }
                """
            )
            cor_init_gyd = session.execute(query_init_gyd, variable_values)
        else:
            # Dummy result
            cor_init_gyd = apure(
                {"pool": {"tokens": [{"balance": 0.0}] * len(pool_result["tokens"])}}
            )

        # SOMEDAY this is slow, but when I use gather(), I overload the endpoint & time out. We should
        # have some strategy here instead.
        result_joinExits, result_swaps, result_init_gyd = await gather_sync(
            cor_joinExits, cor_swaps, cor_init_gyd
        )

        init_gyd_amount = float(
            result_init_gyd["pool"]["tokens"][ix_token_gyd]["balance"]
        )

    # Process events into a useful form
    events = []
    for e in result_joinExits["joinExits"]:
        sign = {"Join": 1, "Exit": -1}[e["type"]]
        # All amounts are decimal-unscaled (decimal numbers)! Some are strings though.
        # (maybe we could use Decimal instead of float at some point but I don't think it matters here)
        gyd_amount = float(e["amounts"][ix_token_gyd])
        events.append(
            {
                "block": int(e["block"]),
                "timestamp": int(e["timestamp"]),
                "delta_gyd": sign * gyd_amount,
                "type": e["type"],
            }
        )
    for e in result_swaps["swaps"]:
        assert e["tokenInSym"] == "GYD" or e["tokenOutSym"] == "GYD"
        is_gyd_in = e["tokenInSym"] == "GYD"
        sign = 1 if is_gyd_in else -1
        gyd_amount = float(e["tokenAmountIn"] if is_gyd_in else e["tokenAmountOut"])
        events.append(
            {
                "block": int(e["block"]),
                "timestamp": int(e["timestamp"]),
                "delta_gyd": sign * gyd_amount,
                "type": "Swap",
            }
        )

    log_info_dict(
        "pool",
        {
            "chain": chain,
            "pool": pool,
            "pool_create_time": pool_create_time,
            "init_gyd_amount": init_gyd_amount,
            "n joinExits": len(result_joinExits["joinExits"]),
            "n swaps": len(result_swaps["swaps"]),
        },
    )

    if not events:
        return init_gyd_amount, pd.DataFrame(
            [], columns=["block", "timestamp", "delta_gyd", "type"]  # type: ignore
        )

    events_df = pd.DataFrame(events).sort_values("block").reset_index(drop=True)
    events_df["timestamp"] = pd.to_datetime(events_df["timestamp"], unit="s", utc=True)

    # NOTE We do *not* separately account for protocol fees right now. That's probably fine, though.

    return init_gyd_amount, events_df


async def get_erc4626_balance_events(
    chain: str, addr: str, deployment_block: int, config: dict
) -> tuple[float, float, pd.DataFrame]:
    """
    Works for any ERC4626 vault that holds GYD as its underlying.

    deployment_block: required b/c there seems to be no way to figure this out reliably across chains OR catch the exception if it isn't deployed.
    """
    # ic(chain, addr)

    w3: AsyncWeb3 = config["w3"][chain]

    start_block = config["start_blocks"][chain]
    end_block = config["end_blocks"][chain]

    with open("abi/IERC4626.json") as f:
        abi = json.load(f)
    con = w3.eth.contract(addr, abi=abi)  # type: ignore

    async def get_init_final_gyd_amount() -> tuple[float, float]:
        init_gyd_amount, final_gyd_amount = await gather(
            (
                con.functions.totalAssets().call(block_identifier=start_block)
                if start_block >= deployment_block
                else apure(0.0)
            ),
            (
                con.functions.totalAssets().call(block_identifier=end_block)
                if end_block >= deployment_block
                else apure(0.0)
            ),
        )
        return init_gyd_amount / 1e18, final_gyd_amount / 1e18

    evs_deposit, evs_withdraw, (init_gyd_amount, final_gyd_amount) = await gather(
        *(
            ee.get_logs(
                fromBlock=start_block + 1,
                toBlock=end_block,
            )
            for ee in (con.events.Deposit(), con.events.Withdraw())
        ),
        get_init_final_gyd_amount(),
    )
    evs_all: list = evs_deposit + evs_withdraw  # type: ignore

    # SOMEDAY we should make sure we're not hitting the rate limit, but fine for now.
    async def go(b: int) -> pd.Timestamp:
        block = await w3.eth.get_block(b)
        return pd.to_datetime(block.timestamp, unit="s", utc=True)  # type: ignore

    block_to_timestamp = await async_map_dict(go, {ev.blockNumber for ev in evs_all})
    # ic(len(block_to_timestamp))

    # NOTE this doesn't take compounding across the period into account! See README. We return the error as `extra_gyd_amount` below.
    events = []
    # ic(len(evs_deposit), len(evs_withdraw))
    for ev in evs_all:
        sign = {"Deposit": 1, "Withdraw": -1}[ev.event]
        events.append(
            {
                "block": ev.blockNumber,
                "timestamp": block_to_timestamp[ev.blockNumber],
                "delta_gyd": sign * ev.args.assets / 1e18,
                "type": ev.event,
            }
        )

    extra_gyd_amount = final_gyd_amount - (
        init_gyd_amount + sum(ev["delta_gyd"] for ev in events)
    )

    log_info_dict(
        "sgyd",
        {
            "chain": chain,
            "addr": addr,
            "init_gyd_amount": init_gyd_amount,
            "final_gyd_amount": final_gyd_amount,
            "extra_gyd_amount": extra_gyd_amount,
            "n deposit": len(evs_deposit),
            "n withdraw": len(evs_withdraw),
        },
    )

    if not events:
        return (
            init_gyd_amount,
            extra_gyd_amount,
            pd.DataFrame([], columns=["block", "timestamp", "delta_gyd", "type"]),  # type: ignore
        )

    events_df = pd.DataFrame(events).sort_values("block").reset_index(drop=True)
    events_df["timestamp"] = pd.to_datetime(events_df["timestamp"], unit="s", utc=True)

    return init_gyd_amount, extra_gyd_amount, events_df


async def get_pool_gauges(chain: str, pool: str, config: dict) -> list[str]:
    """
    Get the gauges for the given pool. We support multiple because in principle, there can be multiple.
    But the first one is always the preferred one, so the other ones can *probably* be ignored.
    """
    ic("get_pool_gauges", chain, pool)

    url = BALANCER_GAUGE_SUBGRAPHS[chain].format(**config["apikeys"])

    # timeout: it's not the fastest thing in the world and we're not being careful not to saturate either.
    # (in practice, this probably doesn't matter)
    transport = GqlAIOHTTPTransport(url=url, timeout=60 * 3)
    async with GqlClient(transport=transport, execute_timeout=30) as session:
        query = gql(
            """
            query LiquidityGauges($pool_address: Bytes!) {
              liquidityGauges(where: {poolAddress: $pool_address, isKilled: false}) {
                id
                isPreferentialGauge
                tokens {
                    symbol
                }
              }
            }
            """
        )
        result = await session.execute(query, variable_values={"pool_address": pool})
    ic(chain, pool, result)
    liquidityGauges = result["liquidityGauges"]

    def sortkey(lg: dict):
        return -1 if lg["isPreferentialGauge"] else 1

    w3: AsyncWeb3 = config["w3"]["ethereum"]  # only for address helper
    ret = [
        str(w3.to_checksum_address(g["id"]))
        for g in sorted(liquidityGauges, key=sortkey)
    ]

    log_info_dict("gauges", {"chain": chain, "pool": pool, "gauges": ret})
    return ret


async def get_excess_reserves(config: dict) -> tuple[pd.Series, pd.DataFrame]:
    """
    Returns: (current row, historical data frame)
    """
    async with AsyncDuneClient(config["apikeys"]["dune_apikey"]) as dune:
        # This is a bit inefficient but fast & saves credits & and our dataset is small.

        query = DuneQueryBase(DUNE_QUERY_ID_EXCESS_RESERVE)

        result = await dune.get_latest_result(query)

        # SOMEDAY has to be updated if we want to pull higher-time-granularity data in the future.
        start_of_day = config["end_time"].floor("D")
        if (
            result.state != DuneExecutionState.COMPLETED
            or pd.Timestamp(unwrap(result.times.execution_ended_at)) < start_of_day
        ):
            result = await dune.refresh(query, ping_frequency=10)
            assert result.state == DuneExecutionState.COMPLETED

    data = pd.DataFrame(unwrap(result.result).rows)
    data["day"] = pd.to_datetime(data["day"], utc=True)
    data["excess_reserve"] = pd.to_numeric(data["excess_reserve_tt"])
    data["gyd_circulating"] = pd.to_numeric(data["gyd_emitted_tt"])

    # Fix: dates are attached to the values at the *end* of the day. We want the value at the actual time (i.e., beginning of day) here.
    # The last date (originally for today) is the current value.
    data = data.sort_values("day")
    data["day"] = data["day"].shift(-1)
    current = data.iloc[-1].copy()
    data = data.drop(data.index[-1])  # type: ignore

    assert data.isna().sum().sum() == 0
    return current, data


async def get_pool_data(chain: str, pool: str, pool_config: dict, config: dict) -> dict:
    if gauge := pool_config.get("gauge"):
        co_gauges = apure([gauge])
    else:
        co_gauges = get_pool_gauges(chain, pool, config)

    (init_gyd_amount, events_df), gauges = await gather(
        get_pool_balance_events(chain, pool, config),
        co_gauges,
    )
    return {
        "init_gyd_amount": init_gyd_amount,
        "balance_events_df": events_df,
        "gauges": gauges,
    }


async def get_chainpool_data(
    config: dict,
) -> dict[tuple[str, str], dict]:
    return await gather_dict(
        {
            (chain, pool["address"]): get_pool_data(
                chain, pool["address"], pool, config
            )
            for chain, venues in config["venues"].items()
            for pool in venues.get("pools", [])
        }
    )


async def get_chainsgyd_balance_events(
    config: dict,
) -> dict[tuple[str, str], tuple[float, float, pd.DataFrame]]:
    """
    This supports multiple sGYDs for symmetry with the pools infra. But of course there's really just one per chain.
    """
    return await gather_dict(
        {
            (chain, v["address"]): get_erc4626_balance_events(
                chain, v["address"], v["deployment_block"], config
            )
            for chain, venues in config["venues"].items()
            for v in venues.get("sgyd", [])
        }
    )


def process_balance_events(
    init_gyd: float, edf: pd.DataFrame, config: dict
) -> tuple[dict, pd.DataFrame]:
    """
    For a venue holding GYD, process the dataframe of events and initial amounts to get amounts per block, average amounts, etc.
    """
    assert init_gyd >= 0

    # SOMEDAY this shouldn't be needed; my queries are clean enough. (famous last words though)
    edf = cast(
        pd.DataFrame,
        edf[edf["timestamp"].between(config["start_time"], config["end_time"])],
    )

    total_duration = config["end_time"] - config["start_time"]
    if edf.empty:
        return {"avg_gyd": init_gyd}, pd.DataFrame()

    # Group by timestamp (i.e., by block) and process changes to get GYD held.
    delta_gyds = cast(
        pd.DataFrame,
        (edf.groupby("timestamp", as_index=False)["delta_gyd"].sum()),
    )
    delta_gyds = delta_gyds.sort_values("timestamp")
    delta_gyds["gyd"] = init_gyd + delta_gyds["delta_gyd"].cumsum()
    assert (delta_gyds["gyd"] >= 0).all()

    delta_gyds["duration"] = delta_gyds["timestamp"].diff().shift(-1)

    # Handle time segment before the first / after the last event.
    # This is asymmetric b/c of... the direction of time.
    init_duration = delta_gyds["timestamp"].iloc[0] - config["start_time"]
    final_duration = config["end_time"] - delta_gyds["timestamp"].iloc[-1]

    delta_gyds.loc[delta_gyds.index[-1], "duration"] = final_duration

    delta_gyds["duration_secs"] = delta_gyds["duration"].dt.total_seconds()
    assert (delta_gyds["duration_secs"] > 0).all()

    gyd_secs = (
        (delta_gyds["gyd"] * delta_gyds["duration_secs"]).sum()
    ) + init_gyd * init_duration.total_seconds()
    avg_gyd = gyd_secs / total_duration.total_seconds()

    return {"avg_gyd": avg_gyd}, delta_gyds


def balance_events_to_venues(
    chainpool_data, chainsgyd_balance_events, config: dict
) -> dict:
    venues = defaultdict(lambda: defaultdict(dict))
    for (chain, pool), data in chainpool_data.items():
        res, _ = process_balance_events(
            data["init_gyd_amount"], data["balance_events_df"], config
        )
        res["gauges"] = data["gauges"]
        venues[chain]["pools"][pool] = res

    for (chain, addr), (init_gyd, extra_gyd, edf) in chainsgyd_balance_events.items():
        res, _ = process_balance_events(init_gyd, edf, config)

        # SOMEDAY this could be enabled to approximate compounding for sGYD.
        # Patch it to include extra_gyd. We spread these evenly across the time period
        # (this is an approximation!)
        # res["avg_gyd"] += extra_gyd

        venues[chain]["sgyd"][addr] = res

    return venues


def process_excess_reserves(
    excess_reserves_vals: tuple[pd.Series, pd.DataFrame], config: dict
) -> dict:
    """
    excess_reserves: Coming from Dune
    """
    (current_row, excess_reserves) = excess_reserves_vals
    start_row = unique_row(
        cast(
            pd.DataFrame,
            excess_reserves[excess_reserves.day == config["start_time"].floor("D")],
        )
    )
    end_row = unique_row(
        cast(
            pd.DataFrame,
            excess_reserves[excess_reserves.day == config["end_time"].floor("D")],
        )
    )
    profit = end_row["excess_reserve"] - start_row["excess_reserve"]
    ic(start_row, end_row, current_row)

    ret = {
        "end_excess_reserve": end_row["excess_reserve"],
        "end_gyd_circulating": end_row["gyd_circulating"],
        "start_excess_reserve": start_row["excess_reserve"],
        "start_gyd_circulating": start_row["gyd_circulating"],
        "current_excess_reserve": current_row["excess_reserve"],
        "current_gyd_circulating": current_row["gyd_circulating"],
        "profit": profit,
    }

    # Sanity check.
    # This can *in principle* happen, so we don't crash the script, but it shouldn't. If this happens, proceed with caution. (emissions would also be 0)
    profit_vars = ret
    if any(v < 0 for v in profit_vars.values()):
        logging.error(
            "Negative reserve profit. %s",
            repr(jsno.jsonify(profit_vars)),
        )

    ic(ret)
    return ret


def check_config(config: dict):
    """Static checks for config, to catch issues early."""
    assert config["start_time"] < config["end_time"]

    for chain_venues in config["venues"].values():
        counts = Counter(
            vc["address"] for venues in chain_venues.values() for vc in venues
        )
        assert not any(c > 1 for c in counts.values())

    assert 0 <= config.get("sgyd_max_apr", np.inf)
    assert 0 <= config.get("pools_max_apr", np.inf)


def calc_emissions(venues: dict, config: dict, excess_reserves_info: dict) -> dict:
    """Main business logic. Updates venues and returns a dict with summary data."""

    # Compute total emission amount
    G = excess_reserves_info["current_gyd_circulating"]
    V = G + excess_reserves_info["current_excess_reserve"]
    emission_total_gyd = max(
        0.0,
        min(
            excess_reserves_info["profit"] * config.get("profit_share", 1.0),
            V / config.get("min_collateralization_ratio", 1.0) - G,
        ),
    )
    if emission_total_gyd <= 0:
        logger.error(
            "No emission. This is likely due to no profit or the min collateralization ratio."
        )
        emission_total_gyd = 0.0

    kind_total_avg_gyd = defaultdict(lambda: 0.0)
    for chain_venues in venues.values():
        for kind, vs in chain_venues.items():
            for v in vs.values():
                kind_total_avg_gyd[kind] += v["avg_gyd"]
    avg_gyd_total = sum(jsearch("*[].*[].*[].avg_gyd", venues))
    assert np.isclose(avg_gyd_total, sum(kind_total_avg_gyd.values()))

    period = config["end_time"] - config["start_time"]
    max_yield_period_kind = {
        "sgyd": deannualize(config.get("sgyd_max_apr", np.inf), period),
        "pools": deannualize(config.get("pools_max_apr", np.inf), period),
    }
    kind_max_emissions = {
        kind: avg_gyd_kind * max_yield_period_kind[kind]
        for kind, avg_gyd_kind in kind_total_avg_gyd.items()
    }

    kind_emissions = {
        kind: avg_gyd_kind / avg_gyd_total * emission_total_gyd
        for kind, avg_gyd_kind in kind_total_avg_gyd.items()
    }

    # Adjust for max APR. This algorithm only works b/c there are exactly two kinds.
    # O/w we have to redistribute overflow proportionally.
    def fix_max_apr_violation(kind: str, other_kind: str | None):
        overflow = kind_emissions[kind] - kind_max_emissions[kind]
        if overflow > 0:
            kind_emissions[kind] = kind_max_emissions[kind]
            if other_kind is not None:
                kind_emissions[other_kind] += overflow

    fix_max_apr_violation("sgyd", "pools")
    fix_max_apr_violation("pools", "sgyd")
    fix_max_apr_violation("sgyd", None)

    log_info_dict(
        "emissions",
        {
            "kind_total_avg_gyd": kind_total_avg_gyd,
            "kind_emissions": kind_emissions,
            "avg_gyd_total": avg_gyd_total,
            "emission_total_gyd pre  APR cap": emission_total_gyd,
            "emission_total_gyd post APR cap": sum(kind_emissions.values()),
        },
    )

    # Sanity check
    assert (
        sum(kind_emissions.values()) <= emission_total_gyd
    ), "Capping APRs cannot increase emissions"
    emission_total_gyd = sum(kind_emissions.values())

    for chain_venues in venues.values():
        for kind, vs in chain_venues.items():
            for v in vs.values():
                v["weight_in_kind"] = v["avg_gyd"] / kind_total_avg_gyd[kind]
                v["emission"] = v["weight_in_kind"] * kind_emissions[kind]

                # For reporting only:
                v["avg_yield_period"] = v["emission"] / v["avg_gyd"]
                v["avg_apr"] = annualize(v["avg_yield_period"], period)
                v["weight"] = v["emission"] / emission_total_gyd

    # Get rid of small emission amounts. See above.
    # We also re-calculate weights to keep things consistent.
    # NB weight_in_kind will be slightly distorted but doesn't seem important (we're not re-using that and better to keep track of history)
    for chain, chain_venues in venues.items():
        for kind, vs in chain_venues.items():
            for addr in list(vs.keys()):
                if vs[addr]["emission"] < MIN_EMISSION_AMOUNT:
                    logger.warning(
                        f"Ignoring venue {chain}:{kind}:{addr}: Emission too small ({vs[addr]['emission']} GYD)"
                    )
                    del vs[addr]
    emission_total_gyd = sum(jsearch("*[].*[].*[].emission", venues))
    for v in jsearch("*[].*[].*[]", venues):
        v["weight"] = v["emission"] / emission_total_gyd
    log_info_dict(
        "adjusted emissions",
        {
            "emission_total_gyd": emission_total_gyd,
        },
    )

    # SOMEDAY refactor: use plain_dict() a bit more elsewhere to simplify code.

    # We check for missing gauges *after* removing small amounts not to over-complain
    for (chain, vkind, addr), v in plain_dict(venues, lvl=2).items():
        if vkind == "pools":
            if not v["gauges"]:
                logger.error(
                    f"No gauges defined for pool {chain}:{addr}. Emission will fail."
                )
                continue
            # SOMEDAY We could also select for only gauges that have GYD registered above, or filter here.
            # This could make sense if we need a new gauge for GYD and it may not be the preferred one. I hope not, though.
            # DEACTIVATED this check because it was a bit flaky in the past. If GYD is not registered, emission will revert.
            # gauge = v["gauges"][0]
            # if not "GYD" in gauge["tokens"]:
            #     logger.error(
            #         f"GYD is not registered as a rewards token for preferred gauge {gauge['address']} of pool {chain}:{addr}. Emission will likely fail."
            #     )

    return {
        "emission_total": emission_total_gyd,
    }


async def run(config: dict):
    logging.basicConfig(level=logging.WARNING)

    # Avoid excessive messages from gql (comment the following lines out for debugging)
    gql_logger = logging.getLogger("gql.transport.aiohttp")
    gql_logger.setLevel(logging.WARNING)

    chainpool_data, chainsgyd_balance_events, excess_reserves = await gather(
        get_chainpool_data(config),
        get_chainsgyd_balance_events(config),
        get_excess_reserves(config),
    )

    venues = balance_events_to_venues(chainpool_data, chainsgyd_balance_events, config)
    excess_reserves_info = process_excess_reserves(excess_reserves, config)

    emission_info = calc_emissions(venues, config, excess_reserves_info)

    config_out = config.copy()
    del config_out["apikeys"]
    del config_out["w3"]
    return jsno.jsonify(
        {
            "venues": venues,
            "config": config_out,
            "excess_reserves": excess_reserves_info,
            "emissions": emission_info,
        }
    )


async def get_block_at_timestamp(chain: str, timestamp: pd.Timestamp, config: dict):
    """
    Get block at time string (like "2024-01-01 12:33 UTC").

    Fails if we use a timestamp so far back that there was actually no block.
    """
    timestamp_unix = int(timestamp.timestamp())

    if chain == "polygon-zkevm":
        # Not supported by the defillama api
        url = f"https://api-zkevm.polygonscan.com/api"
        params = {
            "module": "block",
            "action": "getblocknobytime",
            "timestamp": timestamp_unix,
            "closest": "before",
            "apikey": config["apikeys"]["polygonscan_zkevm_apikey"],
        }
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                assert response.status == 200
                content = await response.json()
                return int(content["result"])
    else:
        url = f"https://coins.llama.fi/block/{chain}/{timestamp_unix}"
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                assert response.status == 200
                content = await response.json()
                return int(content["height"])


async def parse_args_get_config(argv) -> dict:
    epilog = """
        Config file format: see config.example.yaml
    """
    parser = argparse.ArgumentParser(description="Calc yield share", epilog=epilog)
    parser.add_argument("config_file", metavar="CONFIG_FILE.yaml")
    args = parser.parse_args(argv)

    with open(args.config_file) as f:
        config = yaml.safe_load(f)

    load_dotenv()
    config["apikeys"] = {
        "thegraph_apikey": os.environ["THEGRAPH_APIKEY"],
        "polygonscan_zkevm_apikey": os.environ["POLYGONSCAN_ZKEVM_APIKEY"],
        "dune_apikey": os.environ["DUNE_APIKEY"],
        "infura_apikey": os.environ["INFURA_APIKEY"],
    }

    config["start_time"] = pd.to_datetime(config["start_time"], utc=True)
    config["end_time"] = pd.to_datetime(config["end_time"], utc=True)

    check_config(config)

    chains = sorted(set(config["venues"].keys()) | {"ethereum"})
    config["chains"] = chains

    start_blocks = await async_map_dict(
        lambda chain: get_block_at_timestamp(chain, config["start_time"], config),
        chains,
    )
    config = config | {"start_blocks": start_blocks}

    end_blocks = await async_map_dict(
        lambda chain: get_block_at_timestamp(chain, config["end_time"], config),
        chains,
    )
    config = config | {"end_blocks": end_blocks}

    # Only need this for ethereum and arbitrum right now..
    config["w3"] = {
        "ethereum": await get_web3("ethereum"),
        "arbitrum": await get_web3("arbitrum"),
    }

    # gql endpoints
    # chains_with_pools = sorted(
    #     {chain for chain, venues in VENUES.items() if venues.get("pools")}
    # )
    # config["chains_with_pools"] = chains_with_pools
    return config


async def main():
    config = await parse_args_get_config(sys.argv[1:])
    result = await run(config)
    print(yaml.safe_dump(result, indent=2))
    # del config["apikeys"]
    # del config["w3"]
    # ic(config)
    # print(yaml.safe_dump(jsno.jsonify(config), indent=2))


if __name__ == "__main__":
    asyncio.run(main())
