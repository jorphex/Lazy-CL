# Lazy-CL

Python script for the lazy, automating managing your Aerodrome Concentrated Liquidity position so you can stop clicking so much on Aerodrome.

- Rebalances when price hits trigger thresholds of range.
  - Also has force rebalance button. Besides built in retries, this can retry or recover from failed transactions.
- Claims pending AERO rewards.
  - Also has manual claim and sell button.
- Sells claimed AERO (to USDC via KyberSwap).
- Manages that USDC profit (keep in wallet or compound into the LP).
- Lets you withdraw USDC profit or entire USDC balance.
- Includes an emergency exit button to exit and dump everything to USDC.
- User control through a Telegram bot.

## Note
- It's set up for WBLT/USDC on Base right now. Change the contract addresses for other pairs/tokens/chains.
- Range width, rebalance triggers, swap amounts and time thresholds are adjustable.
- The bot starts PAUSED. You need to tell it to start operations via Telegram.
- Can probably be mass produced for multiple pools using Ape's Silverback
- Probably works on Velo and other forks and chains.

### To Do
- Automatically fund gas: sell 10-20 USDC or AERO for ETH when ETH balance drops below threshold.
- Display AERO APR in bot status messages (I have no idea where to get this data).
