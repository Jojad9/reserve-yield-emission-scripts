# Optional helper script, NOT tested yet.

from dotenv import load_dotenv
import submit_distribution

import asyncio
import sys
import argparse

from pprint import pprint


async def main(argv):
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Validate submitted distribution pending in DistributionManager. Safety check.",
    )
    parser.parse_args(argv)

    try:
        pending_distributions_here = submit_distribution.load_last_distributions()
        pending_dtpls_here = [d.to_tuple() for d in pending_distributions_here]
        print("Expected pending distributions:")
        pprint(pending_distributions_here)
        print()
    except FileNotFoundError:
        # So you can view the pending distributions even without the file, for testing / debugging.
        # The check below is gonna fail but the output is gonna be there.
        print("No pending distributions file.")
        print()
        pending_dtpls_here = None

    print("Expected pending distributions, encoded:")
    pprint(pending_dtpls_here)
    print()

    # We only need the distribution_manager but it's easiest to just get everything.
    chainstuff = await submit_distribution.get_chainstuff()
    distribution_manager = chainstuff["distribution_manager"]

    pending_dtpls_there = (
        await distribution_manager.functions.getPendingDistributionsBatch().call()
    )

    print("On-chain pending distributions, encoded:")
    pprint(pending_dtpls_there)
    print()

    # TODO check if this check actually passes on equality (i.e., the types are equal enough and everything)
    assert pending_dtpls_here == pending_dtpls_there

    print("Equal.")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:]))
