
# Gyroscope Reserve Yield Emission Scripts

These scripts handle the calculation of reserve yield to be emitted to GYD holders in different venues, and they also help with submitting these emissions on-chain. See: (TODO LINK TO DOCS)

## Code & Environment

This code uses poetry to manage dependencies and venv. Usage:

```
$ pipx install poetry  # or curl or pip, but pipx is probably best.
$ poetry install
$ poetry run python calc_shared_yield.py  # or `poetry shell` to enter an environment, then use python as normal
```

In `.env`, you need the following variables:

```
THEGRAPH_APIKEY
POLYGONSCAN_ZKEVM_APIKEY
DUNE_APIKEY
INFURA_APIKEY
```

## Scripts

1. `calc_shared_yield.py` - Calculate yield to be emitted. Takes a (hand-writen) yaml file as input and produces a yaml file as output. See `--help` and below for options.
2. `submit_distribution.py` - Produce the calldata to emit that yield via the Distributor, simulate, and optionally submit the distribution to the DistributionManager. Takes the output yaml file from step 1 as input. See `--help` for options. This also saves the distribution data locally for checking.
3. `check_pending_distribution.py` - Compare the locally saved distribution data to the pending distribution in DistributionManager. Optional helper for double checking.

## Configuration / Input for `calc_shared_yield.py`

Input is a single YAML file. See `config.example.yaml`.

**The time frame should probably be one week. See Emission below.**

`start_time` and `end_time` are UTC unless something else is indicated. When in doubt, indicate a time zone using the ISO format.

Optional global keys in the config file (toplevel):

- `sgyd_max_apr` - Max annualized yield for each sGYD venue.
- `pools_max_apr` - Max annualized yield for each pool.
- `min_collateralization_ratio` - Minimum collateralization ratio of the system. We emit less GYD to satisfy this, if needed.
- `profit_share` - Share of reserve profits to emit to all venues combined, subject to `min_collateralization_ratio`. Default 1.

Optional per-venues keys in the config file:

- Pools: `gauge`: to set the gauge to use manually. Otherwise, gauges are pulled from the balancer gauges subgraph.

Very small emissions (< 1 GYD) are ignored. This is required for sGYD and probably a good idea for gauges, too.

The script should give reasonable warnings / errors whenever something doesn't look right. Also gives some info messages.

## Output of `calc_shared_yield.py`

YAML to stdout. Among other stats, each venue has an 'emission' value, which is the amount of GYD that should be minted. This file also contains the input `config` for reference.

## Approach for yield calculation

We do the following:

1. Calculate GYD averaged over time (GYD x time / total_time_frame) across the time frame for each "venue" that should receive a share of the reserve yield (different GYD pools, sGYD). Calculate that in relative terms.
2. Calculate profit to the reserve (in absolute USD) across the time frame. This is the difference in reserve surplus across the time frame, where reserve surplus := (reserve value) - (GYD in circulation)

**Note:** We only emit (part of) the *profit* across the time period to venues, not the total surplus. This means that any surplus that is already there will be retained (subject to asset prices).

Then the profits are distributed to venues based on average GYD holdings but limited by the optional parameters described above.

## Emission via `submit_distribution.py`

GYD amounts are emitted to the different venues (sGYD deployments, pools) via the Distributor and DistributionManager contracts. The distribution is submitted using `submit_distribution.py` (you need to specify `--submit-transact` to actually do this on-chain). The distribution then needs to be confirmed/executed separately. Emission uses the batch call function of the distributor, so only one call is emitted. `submit_distribution.py` takes in the output file generated from `calc_shared_yield.py`, plus a few command line arguments (see `--help`). The script simulates emissions via `eth_call`. (both individual call and the batch call are simulated to detect problems)

For the *testing* deployment, you can also pass `--test-transact` to *run* the distributions. But this won't work in prod because no EOA is authorized.

### Emission Time Frame

For sGYD, one can specify an emission time frame. The default is from now + 12h across a time frame equal to the evaluation time frame that was passed to `calc_shared_yield.py`. This is the right thing to do to keep yields somewhat stable over time (subject to the market and limiting parameters). The default is to start some time out from now so that the transaction is actually signed when emission starts, to avoid discontinuous emissions.

For pools, the time frame is always **one week** from the time when the reward tokens are deposited into the gauge. We cannot change this. Because of this, the evaluation time frame should ideally also be one week to reflect the yield on reserves.

### Error codes

When emission simulation fails, consider the following error codes from Distributor. (you can generate them via `forge selectors list` in the sgyd repo):

BaseDistributor:

+----------+---------------------------------------------------+--------------------------------------------------------------------+
| Type     | Signature                                         | Selector                                                           |
+===================================================================================================================================+
| Error    | AccessControlBadConfirmation()                    | 0x6697b232                                                         |
|----------+---------------------------------------------------+--------------------------------------------------------------------|
| Error    | AccessControlEnforcedDefaultAdminDelay(uint48)    | 0x19ca5ebb                                                         |
|----------+---------------------------------------------------+--------------------------------------------------------------------|
| Error    | AccessControlEnforcedDefaultAdminRules()          | 0x3fc3c27a                                                         |
|----------+---------------------------------------------------+--------------------------------------------------------------------|
| Error    | AccessControlInvalidDefaultAdmin(address)         | 0xc22c8022                                                         |
|----------+---------------------------------------------------+--------------------------------------------------------------------|
| Error    | AccessControlUnauthorizedAccount(address,bytes32) | 0xe2517d3f                                                         |
|----------+---------------------------------------------------+--------------------------------------------------------------------|
| Error    | InvalidDestinationType()                          | 0xb6d2cdef                                                         |
|----------+---------------------------------------------------+--------------------------------------------------------------------|
| Error    | NonZeroValue()                                    | 0xe320176b                                                         |
|----------+---------------------------------------------------+--------------------------------------------------------------------|
| Error    | SafeCastOverflowedUintDowncast(uint8,uint256)     | 0x6dfcc650                                                         |
+----------+---------------------------------------------------+--------------------------------------------------------------------+

DistributionManager:

+----------+--------------------------------------------------------+---------------------------------------------------------------+
| Type     | Signature                                              | Selector                                                      |
+===================================================================================================================================+
| Error    | AccessControlBadConfirmation()                         | 0x6697b232                                                    |
|----------+--------------------------------------------------------+---------------------------------------------------------------|
| Error    | AccessControlUnauthorizedAccount(address,bytes32)      | 0xe2517d3f                                                    |
|----------+--------------------------------------------------------+---------------------------------------------------------------|
| Error    | AddressInsufficientBalance(address)                    | 0xcd786059                                                    |
|----------+--------------------------------------------------------+---------------------------------------------------------------|
| Error    | CannotRekoveAdminRole()                                | 0x98298733                                                    |
|----------+--------------------------------------------------------+---------------------------------------------------------------|
| Error    | DistributionNotExecutable(uint256,uint256)             | 0x861bd402                                                    |
|----------+--------------------------------------------------------+---------------------------------------------------------------|
| Error    | EmptyDistribution()                                    | 0x3b182f55                                                    |
|----------+--------------------------------------------------------+---------------------------------------------------------------|
| Error    | FailedInnerCall()                                      | 0x1425ea42                                                    |
|----------+--------------------------------------------------------+---------------------------------------------------------------|
| Error    | NoPendingDistribution()                                | 0xbcc384ec                                                    |
|----------+--------------------------------------------------------+---------------------------------------------------------------|
| Error    | RoleAlreadyGranted(bytes32,address)                    | 0x6dd4f06c                                                    |
+----------+--------------------------------------------------------+---------------------------------------------------------------+

GydDistributor:

+----------+-----------------------------------------------------+--------------------------------------------------------------------+
| Type     | Signature                                           | Selector                                                           |
+=====================================================================================================================================+
| Error    | AccessControlBadConfirmation()                      | 0x6697b232                                                         |
|----------+-----------------------------------------------------+--------------------------------------------------------------------|
| Error    | AccessControlEnforcedDefaultAdminDelay(uint48)      | 0x19ca5ebb                                                         |
|----------+-----------------------------------------------------+--------------------------------------------------------------------|
| Error    | AccessControlEnforcedDefaultAdminRules()            | 0x3fc3c27a                                                         |
|----------+-----------------------------------------------------+--------------------------------------------------------------------|
| Error    | AccessControlInvalidDefaultAdmin(address)           | 0xc22c8022                                                         |
|----------+-----------------------------------------------------+--------------------------------------------------------------------|
| Error    | AccessControlUnauthorizedAccount(address,bytes32)   | 0xe2517d3f                                                         |
|----------+-----------------------------------------------------+--------------------------------------------------------------------|
| Error    | AddressInsufficientBalance(address)                 | 0xcd786059                                                         |
|----------+-----------------------------------------------------+--------------------------------------------------------------------|
| Error    | DistributionTooSoon(bytes32)                        | 0x9b37fd72                                                         |
|----------+-----------------------------------------------------+--------------------------------------------------------------------|
| Error    | FailedInnerCall()                                   | 0x1425ea42                                                         |
|----------+-----------------------------------------------------+--------------------------------------------------------------------|
| Error    | FeeNotCovered(uint256,uint256)                      | 0x11497925                                                         |
|----------+-----------------------------------------------------+--------------------------------------------------------------------|
| Error    | InvalidDestinationType()                            | 0xb6d2cdef                                                         |
|----------+-----------------------------------------------------+--------------------------------------------------------------------|
| Error    | MaxRateExceeded()                                   | 0x30041da6                                                         |
|----------+-----------------------------------------------------+--------------------------------------------------------------------|
| Error    | MismatchingAmounts(uint256,uint256)                 | 0x1131b1e7                                                         |
|----------+-----------------------------------------------------+--------------------------------------------------------------------|
| Error    | NonZeroValue()                                      | 0xe320176b                                                         |
|----------+-----------------------------------------------------+--------------------------------------------------------------------|
| Error    | NotWhitelistedKey(bytes32)                          | 0xbd3deaf9                                                         |
|----------+-----------------------------------------------------+--------------------------------------------------------------------|
| Error    | SafeCastOverflowedUintDowncast(uint8,uint256)       | 0x6dfcc650                                                         |
+----------+-----------------------------------------------------+--------------------------------------------------------------------+

L2GydDistributor:

+----------+---------------------------------------------------+--------------------------------------------------------------------+
| Type     | Signature                                         | Selector                                                           |
+===================================================================================================================================+
| Error    | AccessControlBadConfirmation()                    | 0x6697b232                                                         |
|----------+---------------------------------------------------+--------------------------------------------------------------------|
| Error    | AccessControlEnforcedDefaultAdminDelay(uint48)    | 0x19ca5ebb                                                         |
|----------+---------------------------------------------------+--------------------------------------------------------------------|
| Error    | AccessControlEnforcedDefaultAdminRules()          | 0x3fc3c27a                                                         |
|----------+---------------------------------------------------+--------------------------------------------------------------------|
| Error    | AccessControlInvalidDefaultAdmin(address)         | 0xc22c8022                                                         |
|----------+---------------------------------------------------+--------------------------------------------------------------------|
| Error    | AccessControlUnauthorizedAccount(address,bytes32) | 0xe2517d3f                                                         |
|----------+---------------------------------------------------+--------------------------------------------------------------------|
| Error    | InvalidDestinationType()                          | 0xb6d2cdef                                                         |
|----------+---------------------------------------------------+--------------------------------------------------------------------|
| Error    | NonZeroValue()                                    | 0xe320176b                                                         |
|----------+---------------------------------------------------+--------------------------------------------------------------------|
| Error    | SafeCastOverflowedUintDowncast(uint8,uint256)     | 0x6dfcc650                                                         |
+----------+---------------------------------------------------+--------------------------------------------------------------------+

sGYD:

+----------+-----------------------------------------------------+--------------------------------------------------------------------+
| Type     | Signature                                           | Selector                                                           |
+=====================================================================================================================================+
| Error    | AccessControlBadConfirmation()                      | 0x6697b232                                                         |
|----------+-----------------------------------------------------+--------------------------------------------------------------------|
| Error    | AccessControlEnforcedDefaultAdminDelay(uint48)      | 0x19ca5ebb                                                         |
|----------+-----------------------------------------------------+--------------------------------------------------------------------|
| Error    | AccessControlEnforcedDefaultAdminRules()            | 0x3fc3c27a                                                         |
|----------+-----------------------------------------------------+--------------------------------------------------------------------|
| Error    | AccessControlInvalidDefaultAdmin(address)           | 0xc22c8022                                                         |
|----------+-----------------------------------------------------+--------------------------------------------------------------------|
| Error    | AccessControlUnauthorizedAccount(address,bytes32)   | 0xe2517d3f                                                         |
|----------+-----------------------------------------------------+--------------------------------------------------------------------|
| Error    | AddressEmptyCode(address)                           | 0x9996b315                                                         |
|----------+-----------------------------------------------------+--------------------------------------------------------------------|
| Error    | AddressInsufficientBalance(address)                 | 0xcd786059                                                         |
|----------+-----------------------------------------------------+--------------------------------------------------------------------|
| Error    | ERC1967InvalidImplementation(address)               | 0x4c9c8ce3                                                         |
|----------+-----------------------------------------------------+--------------------------------------------------------------------|
| Error    | ERC1967NonPayable()                                 | 0xb398979f                                                         |
|----------+-----------------------------------------------------+--------------------------------------------------------------------|
| Error    | ERC20InsufficientAllowance(address,uint256,uint256) | 0xfb8f41b2                                                         |
|----------+-----------------------------------------------------+--------------------------------------------------------------------|
| Error    | ERC20InsufficientBalance(address,uint256,uint256)   | 0xe450d38c                                                         |
|----------+-----------------------------------------------------+--------------------------------------------------------------------|
| Error    | ERC20InvalidApprover(address)                       | 0xe602df05                                                         |
|----------+-----------------------------------------------------+--------------------------------------------------------------------|
| Error    | ERC20InvalidReceiver(address)                       | 0xec442f05                                                         |
|----------+-----------------------------------------------------+--------------------------------------------------------------------|
| Error    | ERC20InvalidSender(address)                         | 0x96c6fd1e                                                         |
|----------+-----------------------------------------------------+--------------------------------------------------------------------|
| Error    | ERC20InvalidSpender(address)                        | 0x94280d62                                                         |
|----------+-----------------------------------------------------+--------------------------------------------------------------------|
| Error    | ERC4626ExceededMaxDeposit(address,uint256,uint256)  | 0x79012fb2                                                         |
|----------+-----------------------------------------------------+--------------------------------------------------------------------|
| Error    | ERC4626ExceededMaxMint(address,uint256,uint256)     | 0x284ff667                                                         |
|----------+-----------------------------------------------------+--------------------------------------------------------------------|
| Error    | ERC4626ExceededMaxRedeem(address,uint256,uint256)   | 0xb94abeec                                                         |
|----------+-----------------------------------------------------+--------------------------------------------------------------------|
| Error    | ERC4626ExceededMaxWithdraw(address,uint256,uint256) | 0xfe9cceec                                                         |
|----------+-----------------------------------------------------+--------------------------------------------------------------------|
| Error    | FailedInnerCall()                                   | 0x1425ea42                                                         |
|----------+-----------------------------------------------------+--------------------------------------------------------------------|
| Error    | InvalidInitialization()                             | 0xf92ee8a9                                                         |
|----------+-----------------------------------------------------+--------------------------------------------------------------------|
| Error    | InvalidStream()                                     | 0xa0f87d33                                                         |
|----------+-----------------------------------------------------+--------------------------------------------------------------------|
| Error    | MathOverflowedMulDiv()                              | 0x227bc153                                                         |
|----------+-----------------------------------------------------+--------------------------------------------------------------------|
| Error    | NotInitializing()                                   | 0xd7e6bcf8                                                         |
|----------+-----------------------------------------------------+--------------------------------------------------------------------|
| Error    | SafeCastOverflowedUintDowncast(uint8,uint256)       | 0x6dfcc650                                                         |
|----------+-----------------------------------------------------+--------------------------------------------------------------------|
| Error    | SafeERC20FailedOperation(address)                   | 0x5274afe7                                                         |
|----------+-----------------------------------------------------+--------------------------------------------------------------------|
| Error    | TooManyStreams()                                    | 0xbd56d753                                                         |
|----------+-----------------------------------------------------+--------------------------------------------------------------------|
| Error    | UUPSUnauthorizedCallContext()                       | 0xe07c8dba                                                         |
|----------+-----------------------------------------------------+--------------------------------------------------------------------|
| Error    | UUPSUnsupportedProxiableUUID(bytes32)               | 0xaa1d49a4                                                         |
+----------+-----------------------------------------------------+--------------------------------------------------------------------+


## Limitations of the yield calculation approach

The yield calculation approach comes with some limitations. See also the sGYD docs (TODO LINK TO DOCS PAGE).

### Changes to yield within period

The current approach does *not* take into account when changes in yield within the time frame correlate with changes to GYD holdings across venues. For example, consider the following situation:

- Starting state: some small-ish amount of GYD exists and is sitting mostly in some venue V1, and the GYD reserve has some high-ish yield rate from high-yield collateral.
- In the middle of the period, some agent mints a large amount of GYD from a low-yield collateral and deposits the GYD into some venue V2. The low-yield collateral depresses the yield on the reserve overall.
- Some time passes, then end of the period.

The script would compute the total profit that the reserve made on its collateral overall across the time period. This profit could be medium-large and it would be received mostly by venue V2 (if the minted amount and the differences in yield are large enough). One might argue, though, that V2 should receive less of the profit because, at the time where it was large, yield was actually low (and V1 should receive more because the reserve made most of its yield while most of the GYD were in V1)

In the future, if one would like to accurately account for effects like these, one could use something like a "virtual portfolio" approach for each venue, like this:

- Fix a given venue. We track a number T.
- Initialize T := starting GYD in venue. Then proceed through events where GYD is added/removed from the venue.
- If GYD is added/removed, adjust T accordingly.
- Between these events, split those time frames into sub-time-frames where the reserve composition did not change (i.e., no minting/redeeming of GYD)
  - Within each sub-time-frame, let R be the factor by which the reserve value changed across the time frame.
  - Multiply T by R.
- The reward (in USD or GYD) to the venue is T - (ending GYD in venue).

This approach is not implemented right now because it requires very precise (block-level) reserve values for GYD that would need to be queried quite often. (maybe rounding to daily/hourly frequency is fine actually, but I haven't checked this)

### No compounding within period

Currently, GYD holdings in the different venues do *not* compound within the evaluation period when they already receive emissiones from reserve yield.

Specifically:

- For pools, GYD holdings within these pools do not compound in anticipation of a share of the yield. This seems about right b/c reserve yield emissions are operationally not transparently added to the pool (need to be claimed).
- For sGYD, GYD holdings also do not compound. Only deposit/withdraw actions are considered. This means that the time-weighted GYD holdings would be higher if everyone withdrew, then added again.
  - This is currently a limitation of the approach.
  - We could be smarter here (e.g., query sGYD more frequently within the period for its current assets), but we cannot completely solve this without modeling out the streamer logic in this script, which is a bit too much effort. (e.g., it could be that some streamer runs out during the period)
  - The difference is likely small, though, when periods are not too long.
  - Note that sGYD *will* auto-accumulate *across* periods b/c we take the `totalAssets()` at the beginning.
  - SOMEDAY We could add a compensation where we assume that any extra GYD are spread equally across the period. Disabled for now.

### No block-level precision

Precision is *not* to the block level. This goes along a couple dimensions:

1. Reserve profit / excess reserve: This is pulled from a Dune query that's currently daily. Could be expanded to some higher granularity (e.g. hourly) but likely won't be at the block level. If we ever want to change this (thought it's likely not worth it):
  - We *could* re-implement the dune query to work at the block level, but this would likely create infeasibly large data. (especially on fast chains)
  - We could also re-implement the value calculation in this script, using only data from Infura/Alchemy.

2. Blocks across chains are fundamentally not synchronized. There's nothing we can do here.

