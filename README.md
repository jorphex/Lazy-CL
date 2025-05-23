# Lazy-CL
Python script for the lazy that automatically manages Aerodrome CL deposits so that you can stop clicking on Aerodrome. Instadumps AERO. Keeps profit in USDC or compounds into LP. Can send USDC profit to defined EOA. Emergency exit withdraws and dumps everything to USDC.

Uses Telegram for user communication. Replace contract addresses to whatever pool and tokens you want. Replace settings to whatever you want. Can probably be mass produced for multiple pools using Ape's Silverback. Probably works on Velo and other forks and chains.

### To do
- Automatically fund gas: sell 10-20 USDC or AERO for ETH when ETH balance drops below threshold
- Display AERO APR in bot status messages (I have no idea where to get this data)
