import yaml
from jmespath import search as jsearch
import numpy as np
import json

from dotenv import load_dotenv

from dataclasses import dataclass

import toolz

from web3 import AsyncWeb3
import eth_abi

from typing import Any, cast

import argparse

import logging

from util import get_web3, gather_dict

import asyncio
from asyncio import gather
import sys
import os

from icecream import ic
from enum import Enum

import pandas as pd
import tzlocal

import pickle

loglevel = logging.INFO
logging.basicConfig(level=loglevel)
logger = logging.getLogger(__name__)
logger.setLevel(loglevel)
# NB ic() logs at INFO level here, not DEBUG b/c we don't have that many calls & it's fine.
ic.configureOutput(outputFunction=logger.info)

# These are hard-baked into the Stream library and cannot be pulled from chain.
SGYD_MINIMUM_STREAM_AMOUNT = 1.0
SGYD_MINIMUM_DURATION_SECONDS = 60 * 60  # 1 hour
SGYD_MAXIMUM_DURATION_SECONDS = 5 * 365 * 24 * 60 * 60  # 5 years


DISTRIBUTORS = {
    "ethereum": {
        "address": "0xc1024e475E357375E16c7F86fE46cbc6fFB0561D",
    },
    "arbitrum": {
        "address": "0x4Dfdad359bd4c154dD307350582C4bff636Ba299",
    },
}

# Ethereum only
DISTRIBUTION_MANAGER_ADDRESS = "0x4D9C765D7431fF354d208CF9A37e7186180e6586"

# EOA that this script will use if --submit-transact is passed
DISTRIBUTION_SUBMITTER_ADDRESS = "0x2A1ef5110063D3F078b66A93C8e925df1E01aB80"

GYD_ADDRESSES = {
    "ethereum": "0xe07F9D810a48ab5c3c914BA3cA53AF14E4491e8A",
    "arbitrum": "0xCA5d8F8a8d49439357d3CF46Ca2e720702F132b8",
    "polygon-zkevm": "0xCA5d8F8a8d49439357d3CF46Ca2e720702F132b8",
}

# Address that is allowed to perform the distribution (call into the L1 Distributor).
# Only for simulation.
# This is the DistributionManager.
DISTRIBUTION_CALLER_ADDRESS = DISTRIBUTION_MANAGER_ADDRESS

# source: https://github.com/smartcontractkit/chain-selectors/blob/main/selectors.yml
# (go by chain ID in that file to find the right selector)
CCIP_CHAIN_SELECTORS = {
    "arbitrum": 4949039107694359620,
    "polygon-zkevm": 4348158687435793198,
}

LAST_SUBMITTED_DISTRIBUTIONS_FILENAME = "last_submitted_distributions.pkl"


class DestinationType(Enum):
    SGyd = 0
    Gauge = 1
    L2 = 2


@dataclass
class Distribution:
    destinationType: DestinationType
    recipient: str
    amount: int
    data: bytes

    def abi_encode(self):
        return eth_abi.encode(  # pyright: ignore[reportPrivateImportUsage]
            ("uint8", "address", "uint256", "bytes"), self.to_tuple()
        )

    def to_tuple(self):
        """
        Convert to a representation that can be passed into a function.
        """
        return (
            self.destinationType.value,
            self.recipient,
            int(self.amount * 1e18),
            self.data,
        )

    def repr_calldata(self, format="python"):
        if format == "python":
            return repr(self.to_tuple())

        tpl_raw = (
            repr(self.destinationType.value),
            "0x" + self.recipient,
            repr(int(self.amount * 1e18)),
            "0x" + self.data.hex(),
        )
        if format == "raw":
            return "(" + ", ".join(tpl_raw) + ")"
        elif format == "cast":
            return " ".join(tpl_raw)
        else:
            raise ValueError("format??")

    def amount_s(self):
        return int(self.amount * 1e18)


@dataclass
class DistributionToL2:
    l2_distributor: str
    ccip_chain_selector: int
    l2_distribution: Distribution

    def data(self):
        # Note: Inside the string you can't have spaces!
        return eth_abi.encode(  # pyright: ignore[reportPrivateImportUsage]
            ("uint256", "(uint8,address,uint256,bytes)"),
            (self.ccip_chain_selector, self.l2_distribution.to_tuple()),
        )

    def to_full(self):
        return Distribution(
            DestinationType.L2,
            self.l2_distributor,
            self.l2_distribution.amount,
            self.data(),
        )

    def to_tuple(self):
        return self.to_full().to_tuple()

    def repr_calldata(self, format="python"):
        return self.to_full().repr_calldata(format=format)

    def amount_s(self):
        return self.to_full().amount_s()


# Mostly a debugging / testing tool by now.
def repr_calldata(x, format="python"):
    if isinstance(x, (Distribution, DistributionToL2)):
        return x.repr_calldata(format)
    elif isinstance(x, list):
        if format == "python":
            return repr([d.to_tuple() for d in x])
        elif format == "raw":
            return "[" + ", ".join([d.repr_calldata(format) for d in x]) + "]"
        elif format == "cast":
            raise ValueError(
                "cast format is (probably) not supported with batch calls."
            )
        else:
            raise ValueError("format??")
    else:
        raise TypeError("Unsupported type: " + str(type(x)))


def save_distributions(distributions: list[Distribution | DistributionToL2]) -> str:
    filename = LAST_SUBMITTED_DISTRIBUTIONS_FILENAME
    with open(filename, "wb") as f:
        pickle.dump(distributions, f)
    return filename


def load_last_distributions() -> list[Distribution | DistributionToL2]:
    with open(LAST_SUBMITTED_DISTRIBUTIONS_FILENAME, "rb") as f:
        return pickle.load(f)


@dataclass
class Config:
    """Internal config."""

    w3_ethereum: AsyncWeb3

    min_distribution_interval_seconds: int
    max_emission: float

    # L1 distributor
    distributor: Any
    distribution_manager: Any

    # Property of DistributionManager
    min_execution_delay_seconds: int

    force: bool
    simulate: bool
    calldata_format: str

    sgyd_start_time: pd.Timestamp
    sgyd_end_time: pd.Timestamp

    submit_transact: bool = False
    test_transact: bool = False


def check_config_output(config: Config, out: dict):
    """Some basic safety checks. These should never fail."""

    assert out["emissions"]["emission_total"] > 0
    assert np.isclose(sum(jsearch("venues.*.*[].*.weight[]", out)), 1.0)
    assert np.isclose(
        sum(jsearch("venues.*.*[].*.emission[]", out)),
        out["emissions"]["emission_total"],
    )

    assert all(
        e >= SGYD_MINIMUM_STREAM_AMOUNT
        for e in jsearch("venues.*.sgyd[].*[].emission", out)
    )

    assert out["emissions"]["emission_total"] <= config.max_emission

    # SOMEDAY check distribution time using min interval and current time. (or maybe below)
    # Right know this feels like too much pain for the buck.
    # This is a function of the distribution (type and recipient; for L2, )
    # To be sure, we can check the last time *overall* was before the interval (pull from or use out.config.start_time assuming we have no overlaps. Warn o/w).
    # Currently not checking anything here, emission will revert if it's not right.


async def get_chainstuff() -> dict:
    w3 = await get_web3("ethereum")

    with open("abi/GYDDistributor.json") as f:
        abi = json.load(f)
    distributor = w3.eth.contract(DISTRIBUTORS["ethereum"]["address"], abi=abi)  # type: ignore

    with open("abi/ERC20.json") as f:
        abi = json.load(f)
    gyd = w3.eth.contract(GYD_ADDRESSES["ethereum"], abi=abi)  # type: ignore

    with open("abi/DistributionManager.json") as f:
        abi = json.load(f)
    distribution_manager = w3.eth.contract(DISTRIBUTION_MANAGER_ADDRESS, abi=abi)  # type: ignore

    (
        max_rate_s,
        min_distribution_interval_seconds,
        gyd_total_supply_s,
        min_execution_delay_seconds,
    ) = await gather(
        distributor.functions.maxRate().call(),
        distributor.functions.minimumDistributionInterval().call(),
        gyd.functions.totalSupply().call(),
        distribution_manager.functions.minExecutionDelay().call(),
    )
    max_rate = max_rate_s / 1e18
    gyd_total_supply = gyd_total_supply_s / 1e18

    max_emission = max_rate * gyd_total_supply

    ic(
        max_rate,
        gyd_total_supply,
        max_emission,
        min_distribution_interval_seconds,
    )

    return {
        "w3_ethereum": w3,
        "distributor": distributor,
        "distribution_manager": distribution_manager,
        "max_emission": max_emission,
        # Currently unused b/c it's too painful to actually check this.
        # But at least you know what could be the problem if there is a failure.
        "min_distribution_interval_seconds": min_distribution_interval_seconds,
        "min_execution_delay_seconds": min_execution_delay_seconds,
    }


def read_isotime_or_local(s: str) -> pd.Timestamp:
    "Like pd.datetime() but if the result has not time zone, we assume the local one."
    ret = pd.to_datetime(s)
    if ret.tz is None:
        local_timezone = tzlocal.get_localzone()
        ret = ret.tz_localize(local_timezone)
    return ret


async def parse_args_get_config_output(argv) -> tuple[Config, dict]:
    """
    Returns (our config, parsed output from calc_shared_yield.py)
    """

    epilog = """
        The input to this is the output file generated by `calc_shared_yield.py` 
    """
    parser = argparse.ArgumentParser(
        description="Make calldata for emitting GYD to venues", epilog=epilog
    )
    parser.add_argument("output_file", metavar="OUTPUT_FILE.yaml")
    parser.add_argument(
        "--force",
        help="Do not run any pre-checks for consistency.",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--no-simulate",
        help="Don't simulate the distributions using eth_call",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--calldata-format",
        choices=["python", "raw", "cast"],
        help="Format that calldata should be printed in: python or raw (for etherscan and safe; default), cast (for foundry's `cast` command). If you use --submit-transact, you don't really need to see the calldata, though.",
        default="raw",
    )
    parser.add_argument(
        "--sgyd-start-time",
        help="Start time for sGYD emission. ISO formatted / default local timezone. Must be far enough in the future to prevent a discontinuous jump in sGYD rate. Default = now + 12h",
    )
    parser.add_argument(
        "--sgyd-end-time",
        help="End time for sGYD emission. ISO formatted / default local timezone. Must be > start time. Default = start time + evaluation time period",
    )
    parser.add_argument(
        "--submit-transact",
        help="Submit the distribution for execution to the DistributionManager. Execution has to be done separately. You need DISTRIBUTION_SUBMITTER_PRIVATE_KEY and you *can* set GAS_PRICE_GWEI in the environment. Submission happens on Ethereum.",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--test-transact",
        help="TESTING only. Send the transactions for distribution. You need DISTRIBUTION_PRIVATE_KEY and you *can* set GAS_PRICE_GWEI in the environment. On ethereum. This won't work for prod deployments b/c no EOA is authorized to execute the distribution.",
        action="store_true",
        default=False,
    )
    args = parser.parse_args(argv)

    with open(args.output_file) as f:
        out = yaml.safe_load(f)

    if s := args.sgyd_start_time:
        sgyd_start_time = read_isotime_or_local(s)
    else:
        sgyd_start_time = pd.Timestamp.now() + pd.Timedelta(hours=12)

    if s := args.sgyd_end_time:
        sgyd_end_time = read_isotime_or_local(s)
    else:
        default_sgyd_period = pd.to_datetime(
            out["config"]["end_time"]
        ) - pd.to_datetime(out["config"]["start_time"])
        sgyd_end_time = sgyd_start_time + default_sgyd_period

    load_dotenv()
    chainstuff = await get_chainstuff()

    config = Config(
        **chainstuff,
        force=args.force,
        simulate=not args.no_simulate,
        calldata_format=args.calldata_format,
        sgyd_start_time=cast(pd.Timestamp, sgyd_start_time),
        sgyd_end_time=sgyd_end_time,
        test_transact=args.test_transact,
        submit_transact=args.submit_transact,
    )
    ic(config)

    if not config.force:
        check_config_output(config, out)

    return config, out


def mk_pool_distribution(config: Config, addr: str, v: dict) -> Distribution:
    gauges = v["gauges"]
    if not gauges:
        logger.error(
            f"Distribution of {v['emission']} GYD desired to pool {addr} but no gauge was found."
        )
        raise ValueError("No gauge")
    elif len(gauges) > 1:
        logger.warning(
            f"Multiple gauges found for pool {addr}. Using the preferred one (if any)."
        )

    gauge = gauges[0]
    return Distribution(
        destinationType=DestinationType.Gauge,
        recipient=gauge,
        amount=v["emission"],
        data=b"",
    )


def mk_sgyd_distribution(config: Config, addr: str, v: dict) -> Distribution:
    startend_enc = eth_abi.encode(  # pyright: ignore[reportPrivateImportUsage]
        ("uint256", "uint256"),
        (
            int(config.sgyd_start_time.timestamp()),
            int(config.sgyd_end_time.timestamp()),
        ),
    )
    return Distribution(
        destinationType=DestinationType.SGyd,
        recipient=addr,
        amount=v["emission"],
        data=startend_enc,
    )


def mk_distribution(
    config: Config, chain: str, vtype: str, addr: str, v: dict
) -> Distribution | DistributionToL2:
    if vtype == "sgyd":
        final_distribution = mk_sgyd_distribution(config, addr, v)
    elif vtype == "pools":
        final_distribution = mk_pool_distribution(config, addr, v)
    else:
        raise ValueError(f"Unknown venue type: {vtype}")

    if chain == "ethereum":
        return final_distribution
    else:
        return DistributionToL2(
            l2_distributor=DISTRIBUTORS[chain]["address"],
            ccip_chain_selector=CCIP_CHAIN_SELECTORS[chain],
            l2_distribution=final_distribution,
        )


async def simulate_distribution(
    distributor, distribution: Distribution | DistributionToL2
) -> dict:
    """
    Simulate a single distribution. Note that this won't be how distributions are actually executed
    (we use the batch feature), but it gives better error messages in case anything fails.
    """
    dtpl = distribution.to_tuple()
    if isinstance(distribution, DistributionToL2):
        try:
            fee = await distributor.functions.getL2DistributionFee(dtpl).call()
        except Exception as e:
            return {
                "success": False,
                "function": "getL2DistributionFee",
                "exception": e,
            }
        try:
            # Doesn't return anything.
            await distributor.functions.distributeGYD(dtpl).call(
                {"from": DISTRIBUTION_CALLER_ADDRESS, "value": fee}
            )
        except Exception as e:
            return {
                "success": False,
                "function": "distributeGYD",
                "fee": fee / 1e18,
                "exception": e,
            }
        return {"success": True, "fee": fee / 1e18}
    else:
        try:
            await distributor.functions.distributeGYD(dtpl).call(
                {"from": DISTRIBUTION_CALLER_ADDRESS}
            )
        except Exception as e:
            return {
                "success": False,
                "function": "distributeGYD",
                "exception": e,
            }
        return {"success": True}


async def simulate_batch_distribution(
    distributor, distributions: list[Distribution | DistributionToL2]
) -> dict:
    dtpls = [distribution.to_tuple() for distribution in distributions]
    has_l2_distribution = any(
        isinstance(distribution, DistributionToL2) for distribution in distributions
    )
    try:
        fee = await distributor.functions.getBatchDistributionFee(dtpls).call()
    except Exception as e:
        return {
            "success": False,
            "function": "getBatchDistributionFee",
            "exception": e,
        }
    try:
        # Doesn't return anything.
        await distributor.functions.batchDistributeGYD(dtpls).call(
            {"from": DISTRIBUTION_CALLER_ADDRESS, "value": fee}
        )
    except Exception as e:
        return {
            "success": False,
            "function": "batchDistributeGYD",
            "fee": fee / 1e18,
            "exception": e,
        }
    return {
        "success": True,
        "fee": fee / 1e18,
        "has_l2_distribution": has_l2_distribution,
    }


async def simulate_submission(
    distributor,
    distribution_manager,
    distributions: list[Distribution | DistributionToL2],
) -> dict:
    dtpls = [distribution.to_tuple() for distribution in distributions]
    try:
        fee = await distributor.functions.getBatchDistributionFee(dtpls).call()
    except Exception as e:
        return {
            "success": False,
            "function": "getBatchDistributionFee",
            "exception": e,
        }
    try:
        # Doesn't return anything.
        await distribution_manager.functions.enqueueDistribution(dtpls).call(
            {"from": DISTRIBUTION_SUBMITTER_ADDRESS, "value": fee}
        )
    except Exception as e:
        return {
            "success": False,
            "function": "enqueueDistribution",
            "fee": fee / 1e18,
            "exception": e,
        }
    return {
        "success": True,
        "fee": fee / 1e18,
    }


# Currently unused, debugging tool.
async def mk_distribution_calldata_str(
    distributor, distribution: Distribution | DistributionToL2, format: str
) -> dict:
    if isinstance(distribution, DistributionToL2):
        fee = await distributor.functions.getL2DistributionFee(
            distribution.to_tuple()
        ).call()
    else:
        fee = 0
    return {"calldata": distribution.repr_calldata(format=format), "value": fee}


# Currently barely used, mostly a debugging / testing / checking tool. See below.
async def mk_batch_distribution_calldata_str(
    distributor, distributions: list[Distribution | DistributionToL2], format: str
) -> dict:
    dtpls = [distribution.to_tuple() for distribution in distributions]
    fee = await distributor.functions.getBatchDistributionFee(dtpls).call()
    return {"calldata": repr_calldata(distributions, format=format), "value": fee}


async def transact(w3: AsyncWeb3, fn, account, tx_params: dict):
    """
    Why t.f. do I have to write this?

    fn = AsyncContractFunction with arguments applied.
    """
    nonce = await w3.eth.get_transaction_count(account.address)

    tx_params = {"from": account.address, "nonce": nonce, **tx_params}

    tx = await fn.build_transaction(tx_params)
    tx = w3.eth.account.sign_transaction(tx, private_key=account.key)

    tx_hash = await w3.eth.send_raw_transaction(tx.rawTransaction)
    tx_receipt = await w3.eth.wait_for_transaction_receipt(tx_hash)

    if not tx_receipt.status:  # type: ignore
        raise RuntimeError(f"tx failed: {tx_receipt}")

    return tx_receipt


def get_test_distributor_caller(w3):
    """
    Get the account (with private key) of the DISTRIBUTION_CALLER_ADDRESS. Testing only; this is not possible for the prod deployment.
    """
    private_key = os.environ["DISTRIBUTION_PRIVATE_KEY"]
    account = w3.eth.account.from_key(private_key)
    assert account.address == DISTRIBUTION_CALLER_ADDRESS
    return account


def get_distribution_submitter(w3):
    """
    Get the account (with private key) of the DISTRIBUTION_SUBMITTER_ADDRESS. For prod.
    """
    private_key = os.environ["DISTRIBUTION_SUBMITTER_PRIVATE_KEY"]
    account = w3.eth.account.from_key(private_key)
    assert account.address == DISTRIBUTION_SUBMITTER_ADDRESS
    return account


# Unused.
async def transact_distribution(
    distributor,
    distribution: Distribution | DistributionToL2,
):
    w3 = distributor.w3

    if gas_price_gwei_str := os.environ.get("GAS_PRICE_GWEI"):
        gas_price_wei = int(float(gas_price_gwei_str) * 1e9)
    else:
        gas_price_wei = int((await w3.eth.gas_price) * 1.1)

    account = get_test_distributor_caller(w3)

    tx_params = {
        "gasPrice": gas_price_wei,
    }
    ic(tx_params)

    # Approval

    amount_s = distribution.amount_s()

    with open("abi/ERC20.json") as f:
        abi = json.load(f)
    gyd = w3.eth.contract(GYD_ADDRESSES["ethereum"], abi=abi)

    allowance = await gyd.functions.allowance(
        account.address, distributor.address
    ).call()

    if allowance < amount_s:
        fn = gyd.functions.approve(distributor.address, amount_s)
        tx_receipt = await transact(distributor.w3, fn, account, tx_params)
        logger.info(f"Approval done. {tx_receipt.transactionHash.hex()}")  # type: ignore

    # Distribution

    if isinstance(distribution, DistributionToL2):
        fee = await distributor.functions.getL2DistributionFee(
            distribution.to_tuple()
        ).call()
    else:
        fee = 0

    fn = distributor.functions.distributeGYD(distribution.to_tuple())
    tx_receipt = await transact(
        distributor.w3, fn, account, {"value": fee, **tx_params}
    )

    logger.info(f"Transaction done. {tx_receipt.transactionHash.hex()}")  # type: ignore


async def transact_batch_distribution(
    distributor,
    distributions: list[Distribution | DistributionToL2],
):
    """
    TESTING deployments only, not possible in prod deployment.
    """
    w3 = distributor.w3

    if gas_price_gwei_str := os.environ.get("GAS_PRICE_GWEI"):
        gas_price_wei = int(float(gas_price_gwei_str) * 1e9)
    else:
        gas_price_wei = int((await w3.eth.gas_price) * 1.1)

    account = get_test_distributor_caller(w3)

    tx_params = {
        "gasPrice": gas_price_wei,
    }
    ic(tx_params)

    # Approval

    amount_s = sum(distribution.amount_s() for distribution in distributions)

    with open("abi/ERC20.json") as f:
        abi = json.load(f)
    gyd = w3.eth.contract(GYD_ADDRESSES["ethereum"], abi=abi)

    allowance = await gyd.functions.allowance(
        account.address, distributor.address
    ).call()

    if allowance < amount_s:
        fn = gyd.functions.approve(distributor.address, amount_s)
        tx_receipt = await transact(distributor.w3, fn, account, tx_params)
        logger.info(f"Approval done. {tx_receipt.transactionHash.hex()}")  # type: ignore

    # Distribution

    dtpls = [distribution.to_tuple() for distribution in distributions]
    fee = await distributor.functions.getBatchDistributionFee(dtpls).call()

    fn = distributor.functions.batchDistributeGYD(dtpls)
    tx_receipt = await transact(
        distributor.w3, fn, account, {"value": fee, **tx_params}
    )

    logger.info(f"Transaction done. {tx_receipt.transactionHash.hex()}")  # type: ignore


# SOMEDAY refactor: use this function in way more places or, even better, use it further up the call stack and cache the value.
async def get_batch_distribution_fee_s(
    distributor,
    distributions: list[Distribution | DistributionToL2],
) -> int:
    """
    Returns a scaled (18-decimal int) value.
    """
    dtpls = [distribution.to_tuple() for distribution in distributions]
    return await distributor.functions.getBatchDistributionFee(dtpls).call()


async def transact_submission(
    distributor,
    distribution_manager,
    distributions: list[Distribution | DistributionToL2],
    fee_s: int | None = None,
):
    if fee_s is None:
        fee_s = await get_batch_distribution_fee_s(distributor, distributions)

    w3 = distributor.w3

    if gas_price_gwei_str := os.environ.get("GAS_PRICE_GWEI"):
        gas_price_wei = int(float(gas_price_gwei_str) * 1e9)
    else:
        gas_price_wei = int((await w3.eth.gas_price) * 1.1)

    account = get_distribution_submitter(w3)
    tx_params = {
        "gasPrice": gas_price_wei,
    }
    ic(tx_params)

    dtpls = [distribution.to_tuple() for distribution in distributions]
    fn = distribution_manager.functions.enqueueDistribution(dtpls)
    tx_receipt = await transact(
        distributor.w3, fn, account, {"value": fee_s, **tx_params}
    )
    logger.info(f"Submission transaction done. {tx_receipt.transactionHash.hex()}")  # type: ignore


async def run(config: Config, out: dict):
    venues_flat = {
        (chain, vtype, addr): v
        for chain, chain_venues in out["venues"].items()
        for vtype, venues in chain_venues.items()
        for addr, v in venues.items()
    }

    distributions = {
        cta: mk_distribution(config, chain, vtype, addr, v)
        for cta, v in venues_flat.items()
        for chain, vtype, addr in [cta]
    }
    distributions_list = list(distributions.values())
    ic(distributions)

    distributions_list_enc = [d.to_tuple() for d in distributions_list]
    ic(distributions_list_enc)

    if config.simulate:
        logger.info("Simulating individual calls...")

        sim_results = await gather_dict(
            {
                cta: simulate_distribution(config.distributor, d)
                for cta, d in distributions.items()
            }
        )
        failures = toolz.valfilter(lambda res: not res["success"], sim_results)
        nsuccess = len(sim_results) - len(failures)

        ic(failures)

        logger.info(f"{nsuccess} simulations successful.")
        successes = toolz.valfilter(lambda res: res["success"], sim_results)
        for cta, res in successes.items():
            logger.info(f"Success: {cta} - {res}")

        if failures:
            logger.error(f"{len(failures)} simulations failed.")
            for cta, res in failures.items():
                logger.error(f"Failed: {cta} with exception: {res['exception']}")
            raise ValueError("Distribution simulations failed.")

        logger.info("Simulating batch call...")
        sim_result = await simulate_batch_distribution(
            config.distributor, distributions_list
        )
        if sim_result["success"]:
            logger.info("Batch simulation successful.")
        else:
            logger.error(
                f"Batch simulation failed with exception: "
                + str(sim_result["exception"])
            )

        logger.info("Simulating submission...")
        sim_result = await simulate_submission(
            config.distributor, config.distribution_manager, distributions_list
        )
        if sim_result["success"]:
            logger.info("Submission simulation successful.")
        else:
            logger.error(
                f"Submission simulation failed with exception: "
                + str(sim_result["exception"])
            )

    # NB this isn't super relevant anymore b/c we have the DistributionManager in between now. Mostly info/debug output.
    cd = await mk_batch_distribution_calldata_str(
        config.distributor, distributions_list, format=config.calldata_format
    )
    logger.info(
        f"Upon execution, the following call will be performed to the batchDistributeGYD() function of the distributor {config.distributor.address} from {DISTRIBUTION_CALLER_ADDRESS}:"
    )
    logger.info(f"    args:  {cd['calldata']}")
    logger.info(f"    value: {cd['value']}")

    if config.submit_transact:
        print(f"--- Running submission tx ---")
        fee_s = await get_batch_distribution_fee_s(
            config.distributor, distributions_list
        )
        await transact_submission(
            config.distributor,
            config.distribution_manager,
            distributions_list,
            fee_s=fee_s,
        )
        min_execution_time = pd.Timestamp.now() + pd.Timedelta(
            seconds=config.min_execution_delay_seconds
        )
        print(
            f"After {min_execution_time}, call executeDistribution() of the DistributionManager {config.distribution_manager.address} on ethereum from the multisig with the following parameters:"
        )
        print(f"    value: {fee_s} = {fee_s/1e18}")

        filename = save_distributions(distributions_list)
        print(f"Distributions saved at: {filename}")

    if config.test_transact:
        print(f"--- Running tx ---")
        await transact_batch_distribution(config.distributor, distributions_list)

    # Unused since we have the batch feature:
    # calldatas = await gather_dict(
    #     {
    #         cta: mk_distribution_calldata_str(
    #             config.distributor, d, format=config.calldata_format
    #         )
    #         for cta, d in distributions.items()
    #     }
    # )
    # print(
    #     f"Perform the following calls to the distributeGYD() function of the distributor {config.distributor.address} from {DISTRIBUTION_CALLER_ADDRESS}:"
    # )
    # for cta, cd in calldatas.items():
    #     print(f"  for {cta}:")
    #     print(f"    args:  {cd['calldata']}")
    #     print(f"    value: {cd['value']}")

    # if config.test_transact:
    #     for cta, d in distributions.items():
    #         print(f"--- Running tx for {cta} ---")
    #         await transact_distribution(config.distributor, d)

    return None


async def main(argv):
    config, out = await parse_args_get_config_output(argv)
    await run(config, out)


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:]))
