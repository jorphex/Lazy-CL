#
# Change sleep times to whatever works for you. 13s seems to be what's stable for Base, which is annoying given 2s block times
#

import asyncio
import json
import time
import logging
from decimal import Decimal, getcontext, ROUND_DOWN
import httpx
import requests
from web3 import Web3
from hexbytes import HexBytes
from web3.exceptions import TransactionNotFound, ContractLogicError
import functools
import telegramify_markdown
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext, CallbackQueryHandler
from telegram.constants import ParseMode

# --- CONFIGURATION (Hardcoded for testing) ---
# use .env here or die!
TELEGRAM_BOT_TOKEN = "your bot token"
TELEGRAM_ADMIN_USER_ID = your user id
ALCHEMY_RPC_URL = "https://base-mainnet.g.alchemy.com/v2/alchemy api key or replace entire url with your preferred provider"
BOT_WALLET_ADDRESS = Web3.to_checksum_address("your bot's eoa")
BOT_PRIVATE_KEY = "bot address private key. never expose it or you will die" # no 0x prefix

# Aerodrome
AERO_TOKEN_ADDRESS = Web3.to_checksum_address("0x940181a94A35A4569E4529A3CDfB74e38FD98631")
WBLT_TOKEN_ADDRESS = Web3.to_checksum_address("0x4E74D4Db6c0726ccded4656d0BCE448876BB4C7A")
USDC_TOKEN_ADDRESS = Web3.to_checksum_address("0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913")

AERODROME_CL_POOL_ADDRESS = Web3.to_checksum_address("0x7cE345561E1690445eEfA0dB04F59d64b65598A8")
AERODROME_SLIPSTREAM_NFT_MANAGER_ADDRESS = Web3.to_checksum_address("0x827922686190790b37229fd06084350E74485b72")
AERODROME_CL_GAUGE_ADDRESS = Web3.to_checksum_address("0xf00d67799Cd4E1A77D14671149b599A96DcD38eC")

# KyberSwap
KYBERSWAP_ROUTER_ADDRESS = Web3.to_checksum_address("0x6131B5fae19EA4f9D964eAc0408E4408b66337b5")
KYBERSWAP_API_BASE_URL = "https://aggregator-api.kyberswap.com/base/api/v1"
KYBERSWAP_X_CLIENT_ID = "AerodromeKyberBotV1" # any name

# Bot Settings (adjustablle to whatever you want)
SLIPPAGE_BPS = 50  # 0.5%
TARGET_RANGE_WIDTH_PERCENTAGE = Decimal("3.0")  # 3% total width for the LP position
REBALANCE_TRIGGER_BUFFER_PERCENTAGE = Decimal("5.0") # 5% buffer from each edge of the active range
AERO_CLAIM_THRESHOLD_AMOUNT = Decimal("50") # Claim AERO if pending > this value
AERO_CLAIM_TIME_THRESHOLD_SECONDS = 6 * 60 * 60
MAIN_LOOP_INTERVAL_SECONDS = 60 * 60
PERIODIC_STATUS_UPDATE_INTERVAL_SECONDS = 6 * 60 * 60
TRANSACTION_TIMEOUT_SECONDS = 360
USER_PROFIT_WITHDRAWAL_ADDRESS = Web3.to_checksum_address("eoa for profit withdrawal")
INITIAL_LP_NFT_ID_CONFIG = None # probably not needed anymore
MIN_SWAP_THRESHOLD_WBLT = Decimal("0.1")
MIN_SWAP_THRESHOLD_USDC = Decimal("1.0")

# Gas
MAX_FEE_PER_GAS_GWEI = Decimal("0.005")
MAX_PRIORITY_FEE_PER_GAS_GWEI = Decimal("0.005")

STATE_FILE = "aerodrome_bot_state.json"
ABI_DIR = "abis"

# Callback
CB_STARTUP_STAKE_NFT = "startup_stake_nft"
CB_STARTUP_WITHDRAW_UNSTAKED_NFT = "startup_withdraw_unstaked_nft"
CB_STARTUP_CONTINUE_MONITORING_STAKED = "startup_continue_monitoring_staked"
CB_STARTUP_UNSTAKE_AND_MANAGE_STAKED = "startup_unstake_and_manage_staked"

CONTRACT_NAME_MAP = {
    AERODROME_SLIPSTREAM_NFT_MANAGER_ADDRESS: "Aerodrome LP Position Manager",
    AERODROME_CL_GAUGE_ADDRESS: "Aerodrome LP Gauge",
    KYBERSWAP_ROUTER_ADDRESS: "KyberSwap Router"
}

# --- Precision for Decimal ---
getcontext().prec = 60

# --- Logging ---
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - %(message)s',
    level=logging.INFO,  # Your bot's default logging level
    handlers=[
        logging.FileHandler("aerodrome_bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Noise ---
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("telegram.ext.ExtBot").setLevel(logging.INFO)
logging.getLogger("telegram.request").setLevel(logging.WARNING)
logging.getLogger("telegram.vendor.ptb_urllib3.urllib3.connectionpool").setLevel(logging.WARNING)

# --- Global Bot State ---
bot_state = {
    "current_lp_principal_wblt_amount": Decimal("0"),
    "current_lp_principal_usdc_amount": Decimal("0"),
    "accumulated_profit_usdc": Decimal("0"),
    "current_strategy": "take_profit",
    "aerodrome_lp_nft_id": None,
    "last_telegram_status_update_time": 0,
    "last_aero_claim_time": 0,
    "operations_halted": True,
    "is_processing_action": False,
    "initial_setup_pending": True
}

# --- Web3 Setup ---
w3 = Web3(Web3.HTTPProvider(ALCHEMY_RPC_URL))

# --- ABI Loading ---
def load_abi(filename):
    full_path = f"{ABI_DIR}/{filename}"
    try:
        with open(full_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"ABI file {full_path} not found.")
        return None
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from ABI file {full_path}.")
        return None

# ABIs
ERC20_ABI = load_abi("erc20_abi_minimal.json")
AERODROME_NFT_MANAGER_ABI = load_abi("aerodrome_slipstream_nft_v1_abi.json")
AERODROME_GAUGE_ABI = load_abi("aerodrome_cl_gauge_abi.json")
AERODROME_POOL_ABI = load_abi("aerodrome_clpool_abi.json")

# --- Contracts ---
aero_token_contract = w3.eth.contract(address=AERO_TOKEN_ADDRESS, abi=ERC20_ABI) if ERC20_ABI else None
wblt_token_contract = w3.eth.contract(address=WBLT_TOKEN_ADDRESS, abi=ERC20_ABI) if ERC20_ABI else None
usdc_token_contract = w3.eth.contract(address=USDC_TOKEN_ADDRESS, abi=ERC20_ABI) if ERC20_ABI else None
aerodrome_nft_manager_contract = w3.eth.contract(address=AERODROME_SLIPSTREAM_NFT_MANAGER_ADDRESS, abi=AERODROME_NFT_MANAGER_ABI) if AERODROME_NFT_MANAGER_ABI else None
aerodrome_gauge_contract = w3.eth.contract(address=AERODROME_CL_GAUGE_ADDRESS, abi=AERODROME_GAUGE_ABI) if AERODROME_GAUGE_ABI else None
aerodrome_pool_contract = w3.eth.contract(address=AERODROME_CL_POOL_ADDRESS, abi=AERODROME_POOL_ABI) if AERODROME_POOL_ABI else None

# --- State ---
def save_state_sync():
    serializable_state = bot_state.copy()
    for key, value in serializable_state.items():
        if isinstance(value, Decimal):
            serializable_state[key] = str(value)
    try:
        with open(STATE_FILE, 'w') as f:
            json.dump(serializable_state, f, indent=4)
        logger.info(f"Bot state saved to {STATE_FILE}")
    except Exception as e:
        logger.error(f"Error saving state: {e}")

async def save_state_async():
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, save_state_sync)


def load_state_sync():
    global bot_state
    try:
        with open(STATE_FILE, 'r') as f:
            loaded_s = json.load(f)
            for key in bot_state.keys():
                if key in loaded_s:
                    if key in ["current_lp_principal_wblt_amount", "current_lp_principal_usdc_amount", "accumulated_profit_usdc", "target_range_width_percentage", "rebalance_buffer_percentage", "aero_claim_threshold_amount"]:
                        bot_state[key] = Decimal(loaded_s[key])
                    else:
                        bot_state[key] = loaded_s[key]
        logger.info(f"Bot state loaded from {STATE_FILE}")
    except FileNotFoundError:
        logger.warning(f"State file {STATE_FILE} not found. Using default state values.")
    except Exception as e:
        logger.error(f"Error loading state: {e}. Using default state values.")
    # Defaults
    bot_state.setdefault("current_lp_principal_wblt_amount", Decimal("0"))
    bot_state.setdefault("current_lp_principal_usdc_amount", Decimal("0"))
    bot_state.setdefault("accumulated_profit_usdc", Decimal("0"))
    bot_state.setdefault("aerodrome_lp_nft_id", None)


# --- Helpers ---
def to_wei(amount, decimals):
    return int(Decimal(amount) * (Decimal(10) ** decimals))

def from_wei(amount_wei, decimals):
    return Decimal(amount_wei) / (Decimal(10) ** decimals)

async def send_tg_message(context: CallbackContext, message: str, menu_type="main"):
    keyboard = None
    if menu_type == "main":
        keyboard = await get_main_menu_keyboard()
    elif menu_type == "profit_withdrawal":
        keyboard = await get_profit_withdrawal_keyboard()
    elif menu_type == "emergency_exit_confirm":
        keyboard = await get_emergency_exit_confirmation_keyboard()
    elif menu_type == "restart_confirm":
        keyboard = await get_restart_confirmation_keyboard()
    elif menu_type == "manage_principal":
        keyboard = await get_manage_principal_keyboard()
    elif menu_type == "startup_unstaked_lp":
        keyboard = await get_startup_unstaked_lp_menu()
    elif menu_type == "startup_staked_lp":
        keyboard = await get_startup_staked_lp_menu()
    # Add other menu types if needed

    try:
        safe_message = telegramify_markdown.markdownify(message)

        await context.bot.send_message(
            chat_id=TELEGRAM_ADMIN_USER_ID,
            text=safe_message, 
            reply_markup=keyboard,
            parse_mode=ParseMode.MARKDOWN_V2 
        )
    except Exception as e:
        logger.error(f"Error sending Telegram message: {e} - Original Message: {message[:200]} - Safe Message: {safe_message[:200] if 'safe_message' in locals() else 'N/A'}")

async def get_amounts_for_liquidity(
    liquidity: int,
    current_tick: int,
    tick_lower: int,
    tick_upper: int,
    token0_decimals: int,
    token1_decimals: int
) -> tuple[Decimal, Decimal]:
    if liquidity == 0:
        return Decimal(0), Decimal(0)

    sqrt_P_current = Decimal("1.0001") ** (Decimal(current_tick) / Decimal(2))
    sqrt_P_lower   = Decimal("1.0001") ** (Decimal(tick_lower) / Decimal(2))
    sqrt_P_upper   = Decimal("1.0001") ** (Decimal(tick_upper) / Decimal(2))

    amount0_wei = Decimal(0)
    amount1_wei = Decimal(0)
    L_decimal = Decimal(liquidity)

    if current_tick < tick_lower:
        if sqrt_P_lower > Decimal(0) and sqrt_P_upper > Decimal(0) and sqrt_P_lower < sqrt_P_upper :
            amount0_wei = L_decimal * ((Decimal(1) / sqrt_P_lower) - (Decimal(1) / sqrt_P_upper))
        amount1_wei = Decimal(0)
    elif current_tick >= tick_upper:
        if sqrt_P_lower < sqrt_P_upper:
            amount1_wei = L_decimal * (sqrt_P_upper - sqrt_P_lower)
        amount0_wei = Decimal(0)
    else:
        if sqrt_P_current > Decimal(0) and sqrt_P_upper > Decimal(0) and sqrt_P_current < sqrt_P_upper:
             amount0_wei = L_decimal * ((Decimal(1) / sqrt_P_current) - (Decimal(1) / sqrt_P_upper))
        
        if sqrt_P_lower < sqrt_P_current:
             amount1_wei = L_decimal * (sqrt_P_current - sqrt_P_lower)

    amount0_human = from_wei(amount0_wei.to_integral_value(rounding=ROUND_DOWN), token0_decimals)
    amount1_human = from_wei(amount1_wei.to_integral_value(rounding=ROUND_DOWN), token1_decimals)
    
    logger.debug(f"get_amounts_for_liquidity: L={liquidity}, tick_c={current_tick}, range=[{tick_lower},{tick_upper}] -> "
                 f"sqrtPs C={sqrt_P_current:.18f} L={sqrt_P_lower:.18f} U={sqrt_P_upper:.18f} -> "
                 f"RawAmts w0={amount0_wei:.0f}, w1={amount1_wei:.0f} -> "
                 f"HumanAmts h0={amount0_human}, h1={amount1_human}")

    return amount0_human, amount1_human

async def calculate_optimal_deposit_amounts(
    current_tick: int,
    tick_lower: int,
    tick_upper: int,
    balance0_wei: int,
    balance1_wei: int
) -> tuple[int, int]:

    logger.info(f"Calculating optimal deposit: current_tick={current_tick}, range=[{tick_lower},{tick_upper}], bal0={balance0_wei}, bal1={balance1_wei}")

    if tick_lower >= tick_upper:
        logger.error("Invalid tick range: tick_lower >= tick_upper.")
        return 0, 0

    # Convert balances to Decimal for precision
    dec_balance0 = Decimal(balance0_wei)
    dec_balance1 = Decimal(balance1_wei)

    # Calculate SqrtPrices (raw, not X96)
    sqrt_P_current = Decimal("1.0001") ** (Decimal(current_tick) / Decimal(2))
    sqrt_P_lower   = Decimal("1.0001") ** (Decimal(tick_lower) / Decimal(2))
    sqrt_P_upper   = Decimal("1.0001") ** (Decimal(tick_upper) / Decimal(2))

    logger.debug(f"SqrtPrices: current={sqrt_P_current}, lower={sqrt_P_lower}, upper={sqrt_P_upper}")

    optimal_amount0_wei = Decimal(0)
    optimal_amount1_wei = Decimal(0)

    # Case A: Price is below the range (or at the lower tick) - only token0
    if sqrt_P_current <= sqrt_P_lower:
        logger.info("Price is at or below lower tick. Will use only token0 (WBLT).")
        optimal_amount0_wei = dec_balance0
        optimal_amount1_wei = Decimal(0)

    # Case B: Price is at or above the upper tick - only token1
    elif sqrt_P_current >= sqrt_P_upper:
        logger.info("Price is at or above upper tick. Will use only token1 (USDC).")
        optimal_amount0_wei = Decimal(0)
        optimal_amount1_wei = dec_balance1
        
    # Case C: Price is within the range - potentially both tokens
    else:
        logger.info("Price is within the range. Calculating optimal L and amounts for both tokens.")

        denom_for_L_from_bal0 = (Decimal(1) / sqrt_P_current) - (Decimal(1) / sqrt_P_upper)
        denom_for_L_from_bal1 = sqrt_P_current - sqrt_P_lower
        
        logger.debug(f"Denom0 (for L from bal0): {denom_for_L_from_bal0}, Denom1 (for L from bal1): {denom_for_L_from_bal1}")

        L_max_from_token0 = Decimal('inf')
        if denom_for_L_from_bal0 > Decimal("1e-18"):
            L_max_from_token0 = dec_balance0 / denom_for_L_from_bal0
        else:
            logger.info("Denominator for L from balance0 is near zero; token0 is not the constraint from this side.")


        L_max_from_token1 = Decimal('inf')
        if denom_for_L_from_bal1 > Decimal("1e-18"):
            L_max_from_token1 = dec_balance1 / denom_for_L_from_bal1
        else:
            logger.info("Denominator for L from balance1 is near zero; token1 is not the constraint from this side.")
            
        logger.info(f"L_max if all token0 used: {L_max_from_token0}, L_max if all token1 used: {L_max_from_token1}")

        L_optimal = min(L_max_from_token0, L_max_from_token1)
        
        if L_optimal == Decimal('inf') or L_optimal <= Decimal(0):
            logger.error(f"Optimal L calculated as {L_optimal}. This is unexpected. Defaulting to 0 amounts.")
            optimal_amount0_wei = Decimal(0)
            optimal_amount1_wei = Decimal(0)
        else:
            logger.info(f"Optimal L determined: {L_optimal}")
            
            if denom_for_L_from_bal0 > Decimal("1e-18"):
                 optimal_amount0_wei = L_optimal * denom_for_L_from_bal0
            else:
                 optimal_amount0_wei = Decimal(0)

            if denom_for_L_from_bal1 > Decimal("1e-18"):
                 optimal_amount1_wei = L_optimal * denom_for_L_from_bal1
            else:
                 optimal_amount1_wei = Decimal(0)

    final_amount0 = min(dec_balance0, optimal_amount0_wei.to_integral_value(rounding=ROUND_DOWN))
    final_amount1 = min(dec_balance1, optimal_amount1_wei.to_integral_value(rounding=ROUND_DOWN))
    
    logger.info(f"Calculated optimal deposit amounts (wei): token0={final_amount0}, token1={final_amount1}")
    return int(final_amount0), int(final_amount1)

# --- Web3 ---
def check_connection():
    if not w3.is_connected():
        logger.error("Web3 not connected!")
        return False
    return True

def get_nonce():
    return w3.eth.get_transaction_count(BOT_WALLET_ADDRESS, 'pending')

def _send_and_wait_for_transaction(
    tx_params_dict_to_sign,
    description="Transaction",
    max_retries: int = 2, # 2 retries + 1 original attempt
    retry_delay_seconds: int = 13
):
    last_exception = None
    for attempt in range(max_retries + 1):
        tx_hash = None
        try:
            logger.debug(f"Attempting to sign and send for {description} (Attempt {attempt + 1}/{max_retries + 1}).")

            if not isinstance(tx_params_dict_to_sign, dict):
                logger.error(f"CRITICAL DEBUG: _send_and_wait_for_transaction expected a dict for {description}, but got {type(tx_params_dict_to_sign)}")
                return None

            tx_params = tx_params_dict_to_sign.copy()

            if 'nonce' not in tx_params or attempt > 0:
                 current_wallet_nonce = w3.eth.get_transaction_count(BOT_WALLET_ADDRESS, 'pending')
                 logger.info(f"Fetching fresh nonce for {description} (Attempt {attempt+1}): {current_wallet_nonce}")
                 tx_params['nonce'] = current_wallet_nonce

            # Gas and ChainID
            if 'gasPrice' not in tx_params and ('maxFeePerGas' not in tx_params or 'maxPriorityFeePerGas' not in tx_params):
                base_fee = w3.eth.get_block('latest')['baseFeePerGas']
                tx_params['maxPriorityFeePerGas'] = w3.to_wei(MAX_PRIORITY_FEE_PER_GAS_GWEI, 'gwei')
                calculated_max_fee = base_fee + tx_params['maxPriorityFeePerGas']
                buffer = w3.to_wei('0.001', 'gwei')
                tx_params['maxFeePerGas'] = calculated_max_fee + buffer
                if tx_params['maxFeePerGas'] < tx_params['maxPriorityFeePerGas']:
                    tx_params['maxFeePerGas'] = tx_params['maxPriorityFeePerGas'] + buffer
            
            if 'chainId' not in tx_params:
                tx_params['chainId'] = w3.eth.chain_id

            # Gas estimation
            if 'gas' not in tx_params:
                try:
                    logger.debug(f"Estimating gas for {description} with params: {tx_params}")
                    tx_params['gas'] = w3.eth.estimate_gas(tx_params)
                except ContractLogicError as e_gas_estim_cle:
                    logger.error(f"ContractLogicError during gas estimation for {description} (Attempt {attempt+1}): {e_gas_estim_cle}. Params: {tx_params}")
                    last_exception = e_gas_estim_cle
                    raise
                except Exception as e_gas_estim:
                    logger.warning(f"Could not estimate gas for {description} (Attempt {attempt+1}): {e_gas_estim}. Using default 800,000. Params: {tx_params}")
                    tx_params['gas'] = 800000

            logger.debug(f"Final transaction parameters for signing ({description}, Attempt {attempt+1}): {tx_params}")
            signed_tx_object = w3.eth.account.sign_transaction(tx_params, BOT_PRIVATE_KEY)

            if not hasattr(signed_tx_object, 'raw_transaction') or signed_tx_object.raw_transaction is None:
                logger.error(f"Failed to sign transaction properly for {description} (Attempt {attempt+1}). No raw_transaction.")
                return None

            tx_hash = w3.eth.send_raw_transaction(signed_tx_object.raw_transaction)
            logger.info(f"{description} sent (Attempt {attempt+1}). Tx Hash: {tx_hash.hex()}")

            receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=TRANSACTION_TIMEOUT_SECONDS)

            if receipt and receipt.status == 1:
                logger.info(f"{description} successful. Tx: {tx_hash.hex()}, Gas used: {receipt.gasUsed}")
                return receipt
            elif receipt and receipt.status == 0:
                logger.error(f"{description} FAILED (Receipt Status 0) on chain. Tx: {tx_hash.hex()}, Gas used: {receipt.gasUsed}")
                return receipt
            else:
                logger.error(f"{description} returned unexpected receipt: {receipt}. Assuming failure.")
                last_exception = RuntimeError(f"Unexpected receipt for {description}")
                if attempt < max_retries:
                    logger.info(f"Retrying in {retry_delay_seconds}s...")
                    time.sleep(retry_delay_seconds)
                else:
                    break
                continue

        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout,
                  httpx.ReadTimeout, httpx.ConnectTimeout, httpx.ConnectError, httpx.ReadError, httpx.PoolTimeout) as net_err:
            logger.warning(f"Network-related error during {description} (Attempt {attempt + 1}/{max_retries + 1}): {net_err}")
            last_exception = net_err
        except TransactionNotFound:
            tx_hash_str = tx_hash.hex() if tx_hash else "N/A"
            logger.error(f"{description} timed out (TransactionNotFound after {TRANSACTION_TIMEOUT_SECONDS}s) (Attempt {attempt + 1}/{max_retries + 1}). Tx Hash: {tx_hash_str}")
            last_exception = TransactionNotFound(f"Transaction {tx_hash_str} not found after timeout.")
        except ContractLogicError as cle:
            logger.error(f"{description} ContractLogicError during prep/gas estimation (Attempt {attempt + 1}/{max_retries + 1}): {cle}")
            last_exception = cle
            raise
        except ValueError as ve:
            logger.error(f"{description} ValueError (Attempt {attempt + 1}/{max_retries + 1}): {ve}")
            last_exception = ve
            if "replacement transaction underpriced" in str(ve).lower() or \
               "nonce too low" in str(ve).lower() or \
               "known transaction" in str(ve).lower():
                logger.warning("Nonce, gas pricing, or known transaction issue detected.")
            else:
                raise
        except Exception as e:
            tx_hash_str = tx_hash.hex() if tx_hash else "N/A"
            logger.error(f"An unexpected error occurred during {description} (Attempt {attempt + 1}/{max_retries + 1}): {e}. Tx Hash: {tx_hash_str}", exc_info=True)
            last_exception = e

        if attempt < max_retries:
            logger.info(f"Retrying {description} in {retry_delay_seconds}s due to: {type(last_exception).__name__}")
            time.sleep(retry_delay_seconds)
        else:
            logger.error(f"{description} ultimately failed after {max_retries + 1} attempts. Last error: {type(last_exception).__name__}: {last_exception}")
            return None

    return None


async def approve_token_spending(context: CallbackContext, token_contract, spender_address, amount_decimal):
    try:
        decimals = await asyncio.to_thread(token_contract.functions.decimals().call)
        if not isinstance(amount_decimal, Decimal):
            amount_decimal = Decimal(str(amount_decimal))

        amount_wei = to_wei(amount_decimal, decimals)
        token_symbol_for_log = await asyncio.to_thread(token_contract.functions.symbol().call)

        # Get human-readable spender name
        spender_name = CONTRACT_NAME_MAP.get(Web3.to_checksum_address(spender_address), spender_address)

        current_allowance = await asyncio.to_thread(
            token_contract.functions.allowance(BOT_WALLET_ADDRESS, spender_address).call
        )
        comparison_amount_wei = amount_wei
        if amount_wei < (2**256 - 1 - (2**256 // 1000)):
             comparison_amount_wei = int(Decimal(amount_wei) * Decimal("0.999"))

        if current_allowance < comparison_amount_wei:
            display_amount = from_wei(amount_wei, decimals)
            if display_amount < Decimal("0.0001") and display_amount > 0:
                display_amount_str = f"{display_amount:.{decimals}f}"
            else:
                display_amount_str = f"{display_amount:.4f}"


            await send_tg_message(
                context, 
                f"ℹ️ Approving `{display_amount_str}` {token_symbol_for_log} for spender: **{spender_name}**...",
                menu_type=None
            )
            
            approve_tx_params_for_build = {
                'from': BOT_WALLET_ADDRESS,
            }
            
            approve_tx_dict = token_contract.functions.approve(spender_address, amount_wei).build_transaction(approve_tx_params_for_build)
            
            logger.debug(f"Built approve_tx_dict for {token_symbol_for_log} to {spender_name}: {approve_tx_dict}")

            receipt = await asyncio.to_thread(
                _send_and_wait_for_transaction, 
                approve_tx_dict, 
                f"Approve {token_symbol_for_log} for {spender_name}"
            )
            
            if receipt and receipt.status == 1:
                await send_tg_message(context, f"✅ Approved {token_symbol_for_log} for **{spender_name}**!", menu_type=None)
                return True
            else:
                await send_tg_message(context, f"❌ Failed to approve {token_symbol_for_log} for **{spender_name}**.", menu_type=None)
                return False
        else:
            logger.info(f"Sufficient allowance for {token_symbol_for_log} by {spender_name} ({spender_address}) already exists.")
            return True
    except Exception as e:
        spender_name_for_error = CONTRACT_NAME_MAP.get(Web3.to_checksum_address(spender_address), spender_address)
        logger.error(f"Error in approve_token_spending for {token_contract.address} to {spender_name_for_error}: {e}", exc_info=True)
        await send_tg_message(context, f"Error approving token for {spender_name_for_error}: {str(e)[:100]}", menu_type=None)
        return False

# --- Token Math & Price Helpers ---

TICK_BASE = Decimal("1.0001")

def calculate_ticks_for_range(center_price_human_readable_t0_in_t1: Decimal,
                              range_width_percentage: Decimal,
                              tick_spacing: int,
                              decimals_t0: int,
                              decimals_t1: int
                             ):
    if center_price_human_readable_t0_in_t1 <= 0:
        raise ValueError("Center price must be positive.")

    decimal_adjustment_factor = Decimal(10)**(decimals_t0 - decimals_t1)
    
    center_raw_price_t0_per_t1 = center_price_human_readable_t0_in_t1 / decimal_adjustment_factor

    half_width_percentage = range_width_percentage / Decimal("200") 
    
    raw_price_lower_boundary = center_raw_price_t0_per_t1 * (Decimal(1) - half_width_percentage)
    raw_price_upper_boundary = center_raw_price_t0_per_t1 * (Decimal(1) + half_width_percentage)

    if raw_price_lower_boundary <= 0:
        center_tick_raw = (center_raw_price_t0_per_t1.ln() / TICK_BASE.ln())
        tick_lower_raw = center_tick_raw - Decimal(tick_spacing * 5)
        logger.warning(f"Raw price lower boundary was <=0. Fallback tick_lower_raw: {tick_lower_raw}")
    else:
        tick_lower_raw = (raw_price_lower_boundary.ln() / TICK_BASE.ln())

    if raw_price_upper_boundary <= 0:
        center_tick_raw = (center_raw_price_t0_per_t1.ln() / TICK_BASE.ln())
        tick_upper_raw = center_tick_raw + Decimal(tick_spacing * 5)
        logger.warning(f"Raw price upper boundary was <=0. Fallback tick_upper_raw: {tick_upper_raw}")
    else:
        tick_upper_raw = (raw_price_upper_boundary.ln() / TICK_BASE.ln())
    
    tick_lower = int(tick_lower_raw / tick_spacing) * tick_spacing
    tick_upper = int(tick_upper_raw / tick_spacing) * tick_spacing

    if tick_lower >= tick_upper:
        tick_lower = (int(tick_lower_raw / tick_spacing) -1) * tick_spacing 
        tick_upper = (int(tick_upper_raw / tick_spacing) +1) * tick_spacing 
        if tick_lower >= tick_upper: 
             tick_upper = tick_lower + tick_spacing

    if tick_lower == tick_upper:
        tick_upper = tick_lower + tick_spacing

    MIN_TICK = -887272
    MAX_TICK = 887272
    tick_lower = max(MIN_TICK, min(tick_lower, MAX_TICK - tick_spacing))
    tick_upper = max(MIN_TICK + tick_spacing, min(tick_upper, MAX_TICK))
    
    if tick_lower >= tick_upper : 
        tick_upper = tick_lower + tick_spacing
        if tick_upper > MAX_TICK:
            tick_lower = MAX_TICK - tick_spacing
            tick_upper = MAX_TICK

    logger.info(f"Calculated Ticks: Target Range {range_width_percentage}%, Center Human Price {center_price_human_readable_t0_in_t1:.6f} (Raw Center Price {center_raw_price_t0_per_t1:.18e}) -> Lower: {tick_lower}, Upper: {tick_upper}")
    return tick_lower, tick_upper


async def get_token_balance(token_contract, owner_address):
    try:
        balance_wei = await asyncio.to_thread(token_contract.functions.balanceOf(owner_address).call)
        decimals = await asyncio.to_thread(token_contract.functions.decimals().call)
        return from_wei(balance_wei, decimals)
    except Exception as e:
        logger.error(f"Error getting balance for token {token_contract.address}: {e}")
        return Decimal("0")

async def get_aerodrome_pool_price_and_tick():
    """Returns current human-readable price of WBLT in terms of USDC, and current tick."""
    try:
        slot0 = await asyncio.to_thread(aerodrome_pool_contract.functions.slot0().call)
        sqrt_price_x96_decimal = Decimal(slot0[0])
        current_tick = slot0[1]

        raw_price_t0_per_t1 = (sqrt_price_x96_decimal / (Decimal(2)**96))**2
        
        wblt_decimals_val = 18 # await asyncio.to_thread(wblt_token_contract.functions.decimals().call)
        usdc_decimals_val = 6  # await asyncio.to_thread(usdc_token_contract.functions.decimals().call)
        
        price_wblt_in_usdc = raw_price_t0_per_t1 * (Decimal(10)**(wblt_decimals_val - usdc_decimals_val))

        price_wblt_in_usdc = (Decimal("1.0001") ** Decimal(current_tick)) * (Decimal(10)**(wblt_decimals_val - usdc_decimals_val))

        logger.info(f"Slot0: sqrtPriceX96={slot0[0]}, tick={current_tick}. Calculated WBLT/USDC price: {price_wblt_in_usdc:.6f}")
        return price_wblt_in_usdc, current_tick

    except Exception as e:
        logger.error(f"Error getting Aerodrome pool price/tick: {e}", exc_info=True)
        return None, None

def tick_to_price(tick, token0_decimals, token1_decimals):
    return Decimal("1.0001") ** tick

def price_to_tick(price_ratio):
    return round(Decimal(price_ratio).ln() / Decimal("1.0001").ln())

# --- Aerodrome ---
async def discover_lp_state(context: CallbackContext, wallet_address: str):
    """
    Discovers the state of the bot's WBLT/USDC LP NFT.
    Checks both staked (in gauge) and unstaked (in wallet) positions.

    Returns:
        tuple: (tokenId, status_string) or (None, None)
               status_string can be "staked" or "unstaked_in_wallet".
    """
    logger.info(f"Attempting to discover LP state for wallet: {wallet_address}")

    if not aerodrome_gauge_contract or not aerodrome_nft_manager_contract:
        logger.error("Gauge or NFT Manager contract not loaded. Cannot discover LP state.")
        await send_tg_message(context, "Error: Contracts not loaded for LP discovery.", menu_type=None)
        return None, None

    # --- 1. Check Staked Positions ---
    try:
        logger.debug(f"Checking staked positions in gauge {AERODROME_CL_GAUGE_ADDRESS}...")
        staked_token_ids = await asyncio.to_thread(
            aerodrome_gauge_contract.functions.stakedValues(wallet_address).call
        )
        logger.info(f"Found {len(staked_token_ids)} token(s) staked by {wallet_address} in this gauge: {staked_token_ids}")

        for token_id in staked_token_ids:
            try:
                position_details_raw = await asyncio.to_thread(
                    aerodrome_nft_manager_contract.functions.positions(token_id).call
                )
                pos_token0 = Web3.to_checksum_address(position_details_raw[2])
                pos_token1 = Web3.to_checksum_address(position_details_raw[3])
                pos_liquidity = position_details_raw[7]

                is_our_pool = (
                    (pos_token0 == WBLT_TOKEN_ADDRESS and pos_token1 == USDC_TOKEN_ADDRESS) or
                    (pos_token0 == USDC_TOKEN_ADDRESS and pos_token1 == WBLT_TOKEN_ADDRESS)
                )

                if is_our_pool:
                    logger.info(f"Staked token ID {token_id} is for WBLT/USDC pool. Liquidity: {pos_liquidity}")
                    if pos_liquidity > 0:
                        logger.info(f"Found ACTIVE STAKED LP: Token ID {token_id} with liquidity {pos_liquidity}.")
                        return int(token_id), "staked"
                    else:
                        logger.info(f"Staked token ID {token_id} (WBLT/USDC) has 0 liquidity. Ignoring.")

            except Exception as e_pos:
                logger.warning(f"Could not get position details for staked token ID {token_id}: {e_pos}")
                continue

    except Exception as e_staked:
        logger.error(f"Error querying staked values from gauge: {e_staked}", exc_info=True)

    # --- 2. Check Unstaked Positions in Wallet ---
    try:
        logger.debug(f"Checking unstaked positions in NFT manager {AERODROME_SLIPSTREAM_NFT_MANAGER_ADDRESS}...")
        balance = await asyncio.to_thread(
            aerodrome_nft_manager_contract.functions.balanceOf(wallet_address).call
        )
        logger.info(f"Wallet {wallet_address} owns {balance} NFT(s) from this manager.")

        if balance == 0:
            logger.info("No NFTs owned by wallet in this manager. No unstaked LP found.")
            return None, None

        for i in range(balance):
            token_id = None
            try:
                token_id = await asyncio.to_thread(
                    aerodrome_nft_manager_contract.functions.tokenOfOwnerByIndex(wallet_address, i).call
                )
                logger.debug(f"Checking wallet NFT at index {i}: Token ID {token_id}")

                position_details_raw = await asyncio.to_thread(
                    aerodrome_nft_manager_contract.functions.positions(token_id).call
                )
                pos_token0 = Web3.to_checksum_address(position_details_raw[2])
                pos_token1 = Web3.to_checksum_address(position_details_raw[3])
                pos_liquidity = position_details_raw[7]

                is_our_pool = (
                    (pos_token0 == WBLT_TOKEN_ADDRESS and pos_token1 == USDC_TOKEN_ADDRESS) or
                    (pos_token0 == USDC_TOKEN_ADDRESS and pos_token1 == WBLT_TOKEN_ADDRESS)
                )

                if is_our_pool:
                    logger.info(f"Wallet token ID {token_id} is for WBLT/USDC pool. Liquidity: {pos_liquidity}")
                    if pos_liquidity > 0:
                        logger.info(f"Found ACTIVE UNSTAKED LP in wallet: Token ID {token_id} with liquidity {pos_liquidity}.")
                        return int(token_id), "unstaked_in_wallet"
                    else:
                        logger.info(f"Wallet token ID {token_id} (WBLT/USDC) has 0 liquidity. Ignoring (likely a burned or empty NFT).")

            except Exception as e_wallet_pos:
                logger.warning(f"Could not process wallet token ID {token_id if token_id else f'(index {i})'}: {e_wallet_pos}")
                continue

    except Exception as e_wallet:
        logger.error(f"Error querying wallet NFTs from manager: {e_wallet}", exc_info=True)

    logger.info("No active WBLT/USDC LP position found either staked or in wallet.")
    return None, None

async def get_lp_position_details(context: CallbackContext, nft_id):
    if not nft_id: return None
    try:
        position = await asyncio.to_thread(aerodrome_nft_manager_contract.functions.positions(nft_id).call)
        if not ( (position[2] == WBLT_TOKEN_ADDRESS and position[3] == USDC_TOKEN_ADDRESS) or \
                 (position[2] == USDC_TOKEN_ADDRESS and position[3] == WBLT_TOKEN_ADDRESS) ):
            await send_tg_message(context, f"Warning: LP NFT {nft_id} is not for WBLT/USDC pair.", menu_type=None)
            return None

        return {
            "token0": position[2],
            "token1": position[3],
            "tickLower": position[5],
            "tickUpper": position[6],
            "liquidity": position[7],
            "tokensOwed0_wei": position[10],
            "tokensOwed1_wei": position[11]
        }
    except Exception as e:
        logger.error(f"Error getting LP position details for NFT {nft_id}: {e}")
        return None

async def get_pending_aero_rewards(context: CallbackContext, nft_id_to_check: int):
    if not nft_id_to_check: return Decimal("0")
    try:
        is_still_staked_by_bot = await asyncio.to_thread(
            aerodrome_gauge_contract.functions.stakedContains(BOT_WALLET_ADDRESS, nft_id_to_check).call
        )

        if not is_still_staked_by_bot:
            logger.info(f"NFT {nft_id_to_check} is not currently staked by {BOT_WALLET_ADDRESS} in gauge {AERODROME_CL_GAUGE_ADDRESS}. Pending AERO assumed to be 0 or claimed.")
            return Decimal("0")

        earned_wei = await asyncio.to_thread(
            aerodrome_gauge_contract.functions.earned(BOT_WALLET_ADDRESS, nft_id_to_check).call
        )
        aero_decimals = await asyncio.to_thread(aero_token_contract.functions.decimals().call)
        return from_wei(earned_wei, aero_decimals)
    except ContractLogicError as cle:
        if cle.message and "NA" in cle.message:
             logger.warning(f"Gauge reverted with 'NA' for NFT {nft_id_to_check} even after stakedContains check (or if check was bypassed). Assuming 0 rewards. Error: {cle}")
             return Decimal("0")
        logger.error(f"ContractLogicError getting pending AERO rewards for NFT {nft_id_to_check}: {cle}", exc_info=True)
        return Decimal("0")
    except Exception as e:
        logger.error(f"Error getting pending AERO rewards for NFT {nft_id_to_check}: {e}", exc_info=True)
        return Decimal("0")

# --- KyberSwap ---
async def get_kyberswap_swap_route(token_in_address, token_out_address, amount_in_wei):
    url = f"{KYBERSWAP_API_BASE_URL}/routes"
    params = {
        "tokenIn": token_in_address,
        "tokenOut": token_out_address,
        "amountIn": str(amount_in_wei)
    }
    headers = {"X-Client-Id": KYBERSWAP_X_CLIENT_ID}
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.get(url, params=params, headers=headers)
        response.raise_for_status()
        data = response.json()
        if data.get("code") == 0 and data.get("data"):
            return data["data"]
        else:
            logger.error(f"KyberSwap GET route error: {data.get('message')}")
            return None
    except httpx.HTTPStatusError as e:
        logger.error(f"KyberSwap GET route HTTP error: {e.response.status_code} - {e.response.text}")
        return None
    except httpx.RequestError as e:
        logger.error(f"KyberSwap GET route request exception: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error in get_kyberswap_swap_route: {e}")
        return None

async def build_kyberswap_swap_data(route_summary, sender_address, recipient_address, slippage_bps_val):
    url = f"{KYBERSWAP_API_BASE_URL}/route/build"
    payload = {
        "routeSummary": route_summary,
        "sender": sender_address,
        "recipient": recipient_address,
        "slippageTolerance": slippage_bps_val,
        "source": KYBERSWAP_X_CLIENT_ID
    }
    headers = {"X-Client-Id": KYBERSWAP_X_CLIENT_ID, "Content-Type": "application/json"}
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        if data.get("code") == 0 and data.get("data"):
            return data["data"]
        else:
            logger.error(f"KyberSwap POST build route error: {data.get('message')}")
            return None
    except httpx.HTTPStatusError as e:
        logger.error(f"KyberSwap POST build route HTTP error: {e.response.status_code} - {e.response.text}")
        return None
    except httpx.RequestError as e:
        logger.error(f"KyberSwap POST build route request exception: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error in build_kyberswap_swap_data: {e}")
        return None

async def execute_kyberswap_swap(context: CallbackContext, token_in_contract, token_out_address, amount_in_decimal):
    """Swaps token_in for token_out using KyberSwap."""
    token_out_symbol = token_out_address[:6] + "..."
    try:
        token_in_decimals = await asyncio.to_thread(token_in_contract.functions.decimals().call)
        token_in_symbol = await asyncio.to_thread(token_in_contract.functions.symbol().call)
        amount_in_wei_for_route_and_approval = to_wei(amount_in_decimal, token_in_decimals)

        await send_tg_message(context, f"ℹ️ Fetching KyberSwap route to sell `{amount_in_decimal:.4f}` {token_in_symbol} for token `{token_out_address}`...", menu_type=None)
        route_data = await get_kyberswap_swap_route(token_in_contract.address, token_out_address, amount_in_wei_for_route_and_approval)

        if not route_data or not route_data.get("routeSummary"):
            await send_tg_message(context, f"Could not get swap route from KyberSwap for {token_in_symbol}.", menu_type=None)
            return False, Decimal("0")

        route_summary = route_data["routeSummary"]
        
        await send_tg_message(context, f"ℹ️ Building KyberSwap transaction data...", menu_type=None)
        swap_build_data = await build_kyberswap_swap_data(route_summary, BOT_WALLET_ADDRESS, BOT_WALLET_ADDRESS, SLIPPAGE_BPS)

        if not swap_build_data or not swap_build_data.get("data"):
            await send_tg_message(context, f"Could not build swap data from KyberSwap for {token_in_symbol}.", menu_type=None)
            return False, Decimal("0")

        api_requested_amount_in_wei_str = swap_build_data.get("amountIn", "0")
        api_requested_amount_in_wei = int(api_requested_amount_in_wei_str)

        logger.info(f"KyberSwap Swap Debug: Intended swap amount (for approval): {amount_in_wei_for_route_and_approval} {token_in_symbol}_wei.")
        logger.info(f"KyberSwap Swap Debug: API route's actual amountIn to be used in tx: {api_requested_amount_in_wei} {token_in_symbol}_wei.")

        if api_requested_amount_in_wei > amount_in_wei_for_route_and_approval:
            logger.warning(f"KyberSwap API amountIn ({api_requested_amount_in_wei}) is GREATER than our initial intended/approval amount ({amount_in_wei_for_route_and_approval}). This could cause TRANSFER_FROM_FAILED.")
        elif api_requested_amount_in_wei < amount_in_wei_for_route_and_approval:
            logger.warning(f"KyberSwap API amountIn ({api_requested_amount_in_wei}) is LESS than our initial intended amount ({amount_in_wei_for_route_and_approval}). Swap will use the smaller API amount.")

        actual_router_for_tx = Web3.to_checksum_address(swap_build_data["routerAddress"])

        approved = await approve_token_spending(context, token_in_contract, actual_router_for_tx, amount_in_decimal)
        if not approved:
            await send_tg_message(context, f"Failed to approve {token_in_symbol} for KyberSwap (amount: {amount_in_decimal}).", menu_type=None)
            return False, Decimal("0")

        # Execute swap
        tx_calldata = swap_build_data["data"]
        tx_value = int(swap_build_data.get("value", "0")) 

        swap_tx_params = {
            'to': actual_router_for_tx,
            'from': BOT_WALLET_ADDRESS,
            'value': tx_value,
            'data': tx_calldata,
            'nonce': get_nonce(),
        }
        
        await send_tg_message(context, f"ℹ️ Executing KyberSwap swap (API expects to use `{from_wei(api_requested_amount_in_wei, token_in_decimals):.4f}` {token_in_symbol})...", menu_type=None)
        receipt = await asyncio.to_thread(_send_and_wait_for_transaction, swap_tx_params, f"KyberSwap {token_in_symbol} Swap")

        if receipt and receipt.status == 1:
            token_out_decimals = 18
            
            if token_out_address == USDC_TOKEN_ADDRESS:
                token_out_decimals = await asyncio.to_thread(usdc_token_contract.functions.decimals().call)
                token_out_symbol = await asyncio.to_thread(usdc_token_contract.functions.symbol().call)
            elif token_out_address == WBLT_TOKEN_ADDRESS:
                token_out_decimals = await asyncio.to_thread(wblt_token_contract.functions.decimals().call)
                token_out_symbol = await asyncio.to_thread(wblt_token_contract.functions.symbol().call)
            elif token_out_address == AERO_TOKEN_ADDRESS:
                token_out_decimals = await asyncio.to_thread(aero_token_contract.functions.decimals().call)
                token_out_symbol = await asyncio.to_thread(aero_token_contract.functions.symbol().call)
            else:
                logger.warning(f"Swapped to an unknown token {token_out_address}. Cannot determine symbol/decimals easily.")
            
            amount_out_wei_str = swap_build_data.get("amountOut", "0")
            amount_out_decimal = from_wei(amount_out_wei_str, token_out_decimals)
            
            actual_input_used_by_api = from_wei(api_requested_amount_in_wei, token_in_decimals)
            
            input_format_str = f":.{min(token_in_decimals, 6)}f"
            output_format_str = f":.{min(token_out_decimals, 6)}f"
            if token_in_symbol == "USDC": input_format_str = ":.2f"
            if token_out_symbol == "USDC": output_format_str = ":.2f"

            logger.info(f"DEBUG: actual_input_used_by_api type: {type(actual_input_used_by_api)}, value: {actual_input_used_by_api}")
            logger.info(f"DEBUG: input_format_str: '{input_format_str}'")
            logger.info(f"DEBUG: token_in_symbol: '{token_in_symbol}'")
            logger.info(f"DEBUG: amount_out_decimal type: {type(amount_out_decimal)}, value: {amount_out_decimal}")
            logger.info(f"DEBUG: output_format_str: '{output_format_str}'")
            logger.info(f"DEBUG: token_out_symbol: '{token_out_symbol}'")
            
            fmt_actual_input = ("{:" + input_format_str.strip(':') + "}").format(actual_input_used_by_api)
            fmt_amount_out = ("{:" + output_format_str.strip(':') + "}").format(amount_out_decimal)

            success_message = (
                f"✅ KyberSwap successful!\n"
                f"Swapped `{fmt_actual_input}` {token_in_symbol} "
                f"for `{fmt_amount_out}` {token_out_symbol}."
            )
            await send_tg_message(context, success_message, menu_type=None)
            return True, amount_out_decimal
        else:
            await send_tg_message(context, f"⚠️ KyberSwap swap for {token_in_symbol} FAILED (tx status 0 or not confirmed).", menu_type=None)
            return False, Decimal("0")

    except Exception as e:
        logger.error(f"Error in execute_kyberswap_swap for {token_in_contract.address} to {token_out_address}: {e}", exc_info=True)
        await send_tg_message(context, f"❌ Critical error during KyberSwap operation: {e}", menu_type=None)
        return False, Decimal("0")

async def attempt_operation_with_retries(
    operation_coro,
    operation_name: str,
    context: CallbackContext,
    max_retries: int = 3,
    delay_seconds: int = 13
):
    """
    Attempts an operation, retrying on failure.
    operation_coro should be an awaitable that returns a truthy value on success
    or a specific structure indicating success/failure and any results.
    For simplicity, let's assume it returns True on success, False on failure for now.
    Or for swaps, (True, amount_out) or (False, Decimal(0)).
    """
    for attempt in range(max_retries + 1):
        try:
            logger.info(f"Attempting {operation_name}, attempt {attempt + 1}/{max_retries + 1}...")

            success, *results = await operation_coro()

            if success:
                logger.info(f"{operation_name} successful on attempt {attempt + 1}.")
                return True, results[0] if results else None
            else:
                logger.warning(f"{operation_name} failed on attempt {attempt + 1} (returned False).")
                if attempt < max_retries:
                    await send_tg_message(context, f"⚠️ {operation_name} failed (attempt {attempt+1}). Retrying in {delay_seconds}s...", menu_type=None)
                    await asyncio.sleep(delay_seconds)
                else:
                    logger.error(f"{operation_name} failed after {max_retries + 1} attempts (returned False).")
                    await send_tg_message(context, f"❌ {operation_name} FAILED after {max_retries + 1} attempts. Please check logs.", menu_type=None)
                    return False, results[0] if results else None

        except ContractLogicError as cle:
            logger.error(f"ContractLogicError during {operation_name} (attempt {attempt + 1}): {cle.message} Data: {cle.data}", exc_info=True)
            if attempt < max_retries:
                await send_tg_message(context, f"⚠️ {operation_name} failed with ContractLogicError (attempt {attempt+1}): {str(cle.message)[:50]}. Retrying in {delay_seconds}s...", menu_type=None)
                await asyncio.sleep(delay_seconds)
            else:
                logger.error(f"{operation_name} failed with ContractLogicError after {max_retries + 1} attempts.")
                await send_tg_message(context, f"❌ {operation_name} FAILED with ContractLogicError after {max_retries + 1} attempts: {str(cle.message)[:50]}. Please check logs.", menu_type=None)
                return False, None
        except Exception as e:
            logger.error(f"Unexpected error during {operation_name} (attempt {attempt + 1}): {e}", exc_info=True)
            if attempt < max_retries:
                await send_tg_message(context, f"⚠️ {operation_name} failed with error (attempt {attempt+1}): {str(e)[:50]}. Retrying in {delay_seconds}s...", menu_type=None)
                await asyncio.sleep(delay_seconds)
            else:
                logger.error(f"{operation_name} failed with error after {max_retries + 1} attempts.")
                await send_tg_message(context, f"❌ {operation_name} FAILED with error after {max_retries + 1} attempts: {str(e)[:50]}. Please check logs.", menu_type=None)
                return False, None
    return False, None

async def _perform_targeted_swap_for_optimal_ratio(
    context: CallbackContext,
    current_pool_sqrt_price_x96: int,
    tick_lower: int, tick_upper: int,
    current_wblt_balance: Decimal, current_usdc_balance: Decimal,
    wblt_price_in_usdc: Decimal,
    wblt_decimals_val: int,    
    usdc_decimals_val: int     
) -> tuple[Decimal, Decimal]:

    logger.info(f"Targeted Swap INPUTS: current_sqrtP_x96={current_pool_sqrt_price_x96}, range=[{tick_lower},{tick_upper}], "
                f"wblt_bal={current_wblt_balance:.8f}, usdc_bal={current_usdc_balance:.8f}, "
                f"price (USDC/WBLT)={wblt_price_in_usdc:.6f}")

    # Convert current_pool_sqrt_price_x96 to the raw sqrtP ratio (not X96)
    # sqrtP = sqrtPriceX96 / 2^96
    sqrt_P_current = Decimal(current_pool_sqrt_price_x96) / (Decimal(2)**96)
    
    # SqrtPrices for the range boundaries are still best derived from ticks
    sqrt_P_lower   = Decimal("1.0001") ** (Decimal(tick_lower) / Decimal(2))
    sqrt_P_upper   = Decimal("1.0001") ** (Decimal(tick_upper) / Decimal(2))

    logger.info(f"Targeted Swap SqrtPs: Current={sqrt_P_current:.30f}, Lower={sqrt_P_lower:.30f}, Upper={sqrt_P_upper:.30f}")

    # Initialize final balances to current balances; they will be updated if a swap occurs
    final_wblt = current_wblt_balance
    final_usdc = current_usdc_balance

    # Case A: Price at or below the lower tick -> Target 100% WBLT
    if sqrt_P_current <= sqrt_P_lower:
        logger.info("Targeted Swap: Price at/below range. Aiming for all capital as WBLT.")
        if current_usdc_balance > MIN_SWAP_THRESHOLD_USDC:
            await send_tg_message(context, f"Optimal Ratio Swap: Converting all {current_usdc_balance:.2f} USDC to WBLT.", menu_type=None)
            swap_op = functools.partial(execute_kyberswap_swap, context, usdc_token_contract, WBLT_TOKEN_ADDRESS, current_usdc_balance)
            success, wblt_gained = await attempt_operation_with_retries(swap_op, "Swap all USDC to WBLT (Optimal Ratio)", context)
            if success:
                final_wblt += (wblt_gained if wblt_gained else Decimal(0))
                final_usdc = Decimal(0) 
            else:
                logger.warning("Failed to swap all USDC to WBLT for optimal ratio. Using current balances.")
        return final_wblt, final_usdc

    # Case B: Price at or above upper tick -> Target 100% USDC
    elif sqrt_P_current >= sqrt_P_upper:
        logger.info("Targeted Swap: Price at/above range. Aiming for all capital as USDC.")
        if current_wblt_balance > MIN_SWAP_THRESHOLD_WBLT:
            await send_tg_message(context, f"Optimal Ratio Swap: Converting all {current_wblt_balance:.4f} WBLT to USDC.", menu_type=None)
            swap_op = functools.partial(execute_kyberswap_swap, context, wblt_token_contract, USDC_TOKEN_ADDRESS, current_wblt_balance)
            success, usdc_gained = await attempt_operation_with_retries(swap_op, "Swap all WBLT to USDC (Optimal Ratio)", context)
            if success:
                final_usdc += (usdc_gained if usdc_gained else Decimal(0))
                final_wblt = Decimal(0)
            else:
                logger.warning("Failed to swap all WBLT to USDC for optimal ratio. Using current balances.")
        return final_wblt, final_usdc

    # Case C: Price within the range
    else:
        logger.info("Targeted Swap: Price within range. Calculating ideal token ratio for full capital deployment.")
        term0_for_L = (Decimal(1) / sqrt_P_current) - (Decimal(1) / sqrt_P_upper)
        term1_for_L = sqrt_P_current - sqrt_P_lower

        if not (term0_for_L > Decimal('1e-30') and term1_for_L > Decimal('1e-30')):
            logger.warning(f"Targeted Swap: Terms for L are not both sufficiently positive (term0={term0_for_L}, term1={term1_for_L}). Skipping targeted swap.")
            return final_wblt, final_usdc

        ideal_smallest_unit_ratio_0_to_1 = term0_for_L / term1_for_L
        decimal_adjustment_for_human_ratio = Decimal(10)**(wblt_decimals_val - usdc_decimals_val)
        ideal_human_unit_ratio_wblt_per_usdc = ideal_smallest_unit_ratio_0_to_1 / decimal_adjustment_for_human_ratio
        
        logger.info(f"Targeted Swap: Ideal HUMAN unit ratio (WBLT_human / USDC_human) for range: {ideal_human_unit_ratio_wblt_per_usdc:.8f}")

        total_capital_in_usdc_value = (current_wblt_balance * wblt_price_in_usdc) + current_usdc_balance
        logger.info(f"Targeted Swap: Total capital value: ${total_capital_in_usdc_value:.2f} (in USDC terms)")
        
        denominator_for_target_usdc = (ideal_human_unit_ratio_wblt_per_usdc * wblt_price_in_usdc) + Decimal(1)
        
        if denominator_for_target_usdc.is_zero() or abs(denominator_for_target_usdc) < Decimal('1e-18'):
             logger.error("Targeted Swap: Denominator for target USDC calculation is zero or too small.")
             return final_wblt, final_usdc

        target_usdc_human_amount = total_capital_in_usdc_value / denominator_for_target_usdc
        target_wblt_human_amount = ideal_human_unit_ratio_wblt_per_usdc * target_usdc_human_amount
        
        logger.info(f"Targeted Swap: Ideal target amounts for full capital: WBLT={target_wblt_human_amount:.8f}, USDC={target_usdc_human_amount:.8f}")

        wblt_needed_or_excess = target_wblt_human_amount - current_wblt_balance
        logger.info(f"Targeted Swap: WBLT needed (positive) or excess (negative): {wblt_needed_or_excess:.8f}")
        value_of_wblt_to_acquire_in_usdc = wblt_needed_or_excess * wblt_price_in_usdc
        MIN_SWAP_VALUE_THRESHOLD_USD = Decimal("1.0") 

        if abs(value_of_wblt_to_acquire_in_usdc) < MIN_SWAP_VALUE_THRESHOLD_USD:
            logger.info(f"Targeted Swap: Difference to target ratio is too small (value ~${abs(value_of_wblt_to_acquire_in_usdc):.2f}). No swap needed.")
            return final_wblt, final_usdc

        if wblt_needed_or_excess > Decimal("1e-9"): # Need to BUY WBLT (sell USDC)
            amount_wblt_to_buy = wblt_needed_or_excess
            usdc_to_sell = amount_wblt_to_buy * wblt_price_in_usdc 
            usdc_to_sell = min(usdc_to_sell, current_usdc_balance) 
            
            if usdc_to_sell > MIN_SWAP_THRESHOLD_USDC:
                logger.info(f"Targeted Swap: Need to buy ~{amount_wblt_to_buy:.4f} WBLT. Selling ~{usdc_to_sell:.2f} USDC.")
                await send_tg_message(context, f"ℹ️ Optimal Ratio Swap: Selling `{usdc_to_sell:.2f}` USDC for WBLT.", menu_type=None)
                swap_op = functools.partial(execute_kyberswap_swap, context, usdc_token_contract, WBLT_TOKEN_ADDRESS, usdc_to_sell)
                success, wblt_gained = await attempt_operation_with_retries(swap_op, "Swap USDC to WBLT (Optimal Ratio)", context)
                if success:
                    final_wblt += (wblt_gained if wblt_gained else Decimal(0))
                    final_usdc -= usdc_to_sell 
                else:
                    logger.warning("Failed to swap USDC to WBLT for optimal ratio. Using current balances.")
            else:
                logger.info(f"Targeted Swap: Calculated USDC amount to sell ({usdc_to_sell:.2f}) for WBLT is below MIN_SWAP_THRESHOLD_USDC.")

        elif wblt_needed_or_excess < Decimal("-1e-9"): # Have excess WBLT (sell WBLT)
            amount_wblt_to_sell = abs(wblt_needed_or_excess)
            amount_wblt_to_sell = min(amount_wblt_to_sell, current_wblt_balance)

            if amount_wblt_to_sell > MIN_SWAP_THRESHOLD_WBLT:
                logger.info(f"Targeted Swap: Have ~{amount_wblt_to_sell:.4f} excess WBLT. Selling this amount.")
                await send_tg_message(context, f"ℹ️ Optimal Ratio Swap: Selling `{amount_wblt_to_sell:.4f}` WBLT for USDC.", menu_type=None)
                swap_op = functools.partial(execute_kyberswap_swap, context, wblt_token_contract, USDC_TOKEN_ADDRESS, amount_wblt_to_sell)
                success, usdc_gained = await attempt_operation_with_retries(swap_op, "Swap WBLT to USDC (Optimal Ratio)", context)
                if success:
                    final_usdc += (usdc_gained if usdc_gained else Decimal(0))
                    final_wblt -= amount_wblt_to_sell
                else:
                    logger.warning("Failed to swap WBLT to USDC for optimal ratio. Using current balances.")
            else:
                logger.info(f"Targeted Swap: Calculated WBLT amount to sell ({amount_wblt_to_sell:.4f}) for USDC is below MIN_SWAP_THRESHOLD_WBLT.")
        
        return final_wblt, final_usdc

# --- Telegram Menu ---
async def get_main_menu_keyboard():
    keyboard_rows = []

    # Row 1: Primary Action (Start/Resume or Pause/Exit)
    if bot_state.get("operations_halted", True):
        if bot_state.get("initial_setup_pending", True) and not bot_state.get("aerodrome_lp_nft_id"):
            keyboard_rows.append([InlineKeyboardButton("▶️ START (Setup New LP)", callback_data="start_bot_operations")])
        else:
            keyboard_rows.append([InlineKeyboardButton("▶️ RESUME OPERATIONS", callback_data="start_bot_operations")])
    else:
        keyboard_rows.append(
            [InlineKeyboardButton("⏸️ PAUSE", callback_data="pause_bot_operations"),
             InlineKeyboardButton("🛑 EMERGENCY EXIT", callback_data="emergency_exit_confirm")]
        )

    # Row 2: Status and Strategy
    keyboard_rows.append(
        [InlineKeyboardButton("📊 Status", callback_data="status"),
         InlineKeyboardButton(f"⚙️ Strat: {bot_state['current_strategy'][:4]}", callback_data="toggle_strategy")]
    )

    # Row 3: Manual Actions / Management
    keyboard_rows.append(
        [InlineKeyboardButton("💰 Claim AERO", callback_data="claim_sell_aero"),
         InlineKeyboardButton("🔄 Force Rebalance", callback_data="force_rebalance")]
    )
    
    # Row 4: Financial Management
    keyboard_rows.append(
        [InlineKeyboardButton("💸 Withdraw Profit", callback_data="withdraw_profit_menu"),
         InlineKeyboardButton("🏦 Manage Principal", callback_data="manage_principal_menu")]
    )
    
    return InlineKeyboardMarkup(keyboard_rows)

async def get_startup_unstaked_lp_menu():
    keyboard = [
        [InlineKeyboardButton("✅ Stake this NFT", callback_data=CB_STARTUP_STAKE_NFT)],
        [InlineKeyboardButton("🛑 Withdraw Liquidity from this NFT", callback_data=CB_STARTUP_WITHDRAW_UNSTAKED_NFT)]
    ]
    return InlineKeyboardMarkup(keyboard)

async def get_startup_staked_lp_menu():
    keyboard = [
        [InlineKeyboardButton("▶️ Continue Monitoring (Normal Loop)", callback_data=CB_STARTUP_CONTINUE_MONITORING_STAKED)],
        [InlineKeyboardButton("🛠️ Unstake & Manage (Withdraw/Rebalance)", callback_data=CB_STARTUP_UNSTAKE_AND_MANAGE_STAKED)]
    ]
    return InlineKeyboardMarkup(keyboard)

async def get_profit_withdrawal_keyboard():
    profit_str = f"{bot_state['accumulated_profit_usdc']:.2f} USDC"
    keyboard = [
        [InlineKeyboardButton(f"Withdraw ALL ({profit_str})", callback_data="withdraw_profit_all")],
        [InlineKeyboardButton("Enter Custom Amount", callback_data="withdraw_profit_custom")],
        [InlineKeyboardButton("⬅️ Back to Main Menu", callback_data="main_menu")]
    ]
    return InlineKeyboardMarkup(keyboard)

async def get_emergency_exit_confirmation_keyboard():
    keyboard = [
        [InlineKeyboardButton("✅ YES, EXECUTE EXIT!", callback_data="emergency_exit_execute")],
        [InlineKeyboardButton("❌ NO, CANCEL", callback_data="main_menu")]
    ]
    return InlineKeyboardMarkup(keyboard)

async def get_restart_confirmation_keyboard():
    keyboard = [
        [InlineKeyboardButton("✅ YES, RESTART BOT!", callback_data="restart_operations_execute")],
        [InlineKeyboardButton("❌ NO, CANCEL", callback_data="main_menu")]
    ]
    return InlineKeyboardMarkup(keyboard)

async def get_manage_principal_keyboard():
    keyboard = [
        [InlineKeyboardButton("ℹ️ View Principal", callback_data="view_principal")],
        [InlineKeyboardButton(f"➕ Add Funds (to {BOT_WALLET_ADDRESS[:8]}..)", callback_data="add_funds_info")],
        [InlineKeyboardButton("🛠️ Set Initial LP NFT ID", callback_data="set_initial_lp_nft_id_prompt")],
        [InlineKeyboardButton("⬅️ Back to Main Menu", callback_data="main_menu")]
    ]
    return InlineKeyboardMarkup(keyboard)


# --- Telegram Command Handlers ---
async def start_command(update: Update, context: CallbackContext):
    if update.effective_user.id != TELEGRAM_ADMIN_USER_ID:
        await update.message.reply_text("You are not authorized to use this bot.")
        return
    bot_state["operations_halted"] = False
    await save_state_async()
    await send_tg_message(context, "Welcome to the Aerodrome LP Manager Bot! What would you like to do?")

async def menu_command(update: Update, context: CallbackContext):
    if update.effective_user.id != TELEGRAM_ADMIN_USER_ID: return
    await send_tg_message(context, "Main Menu:")

async def help_command(update: Update, context: CallbackContext):
    if update.effective_user.id != TELEGRAM_ADMIN_USER_ID: return
    help_text = (
        "Aerodrome LP Manager Bot Commands:\n"
        "/start or /menu - Show the main menu.\n"
        # Add more help text as needed
    )
    await send_tg_message(context, help_text)

# --- Telegram Callback Button Handlers ---
async def button_handler(update: Update, context: CallbackContext):
    if update.effective_user.id != TELEGRAM_ADMIN_USER_ID:
        await context.bot.answer_callback_query(update.callback_query.id, "Unauthorized", show_alert=True)
        return
    
    query = update.callback_query
    await query.answer()

    action = query.data
    logger.info(f"Button pressed: {action} by user {update.effective_user.id}")

    if bot_state.get("is_processing_action") and action not in [
        "status", "main_menu", "view_principal", "add_funds_info",
        CB_STARTUP_STAKE_NFT, CB_STARTUP_WITHDRAW_UNSTAKED_NFT,
        CB_STARTUP_CONTINUE_MONITORING_STAKED, CB_STARTUP_UNSTAKE_AND_MANAGE_STAKED
    ]:
        await send_tg_message(context, "⚠️ Another action is currently in progress. Please wait.", menu_type=None)
        return

    bot_state["is_processing_action"] = True

    try:
        if action == "status":
            await handle_status_action(context)
        elif action == "toggle_strategy":
            await handle_toggle_strategy_action(context)
        elif action == "claim_sell_aero":
            await handle_claim_sell_aero_action(context)
        elif action == "force_rebalance":
            await handle_force_rebalance_action(context)
        elif action == "withdraw_profit_menu":
            await handle_withdraw_profit_menu_action(context)
        elif action == "withdraw_profit_all":
            await handle_withdraw_profit_all_action(context)
        elif action == "withdraw_profit_custom":
            context.user_data['awaiting_profit_withdrawal_amount'] = True
            await send_tg_message(context, "Please type the amount of USDC you wish to withdraw (e.g., 10.75):", menu_type=None)
        elif action == "manage_principal_menu":
            await send_tg_message(context, "Principal Management Options:", menu_type="manage_principal")
        elif action == "view_principal":
            await handle_view_principal_action(context)
        elif action == "add_funds_info":
            await send_tg_message(context, f"To add funds, send WBLT or USDC to the bot's address: `{BOT_WALLET_ADDRESS}`. Funds will be included in the next rebalance.", menu_type="main")
        elif action == "set_initial_lp_nft_id_prompt":
            context.user_data['awaiting_initial_lp_nft_id'] = True
            await send_tg_message(context, "Please type the Token ID of your existing WBLT-USDC LP NFT owned by the bot:", menu_type=None)
        elif action == "emergency_exit_confirm":
            await send_tg_message(context, "⚠️ **WARNING!** This will unstake, withdraw all LP funds, convert WBLT and AERO to USDC, and halt bot operations. Are you sure?", menu_type="emergency_exit_confirm")
        elif action == "emergency_exit_execute":
            await handle_emergency_exit_action(context)
        # --- STARTUP LP DISCOVERY ACTIONS ---
        elif action == CB_STARTUP_STAKE_NFT:
            await handle_startup_stake_nft_action(context)
        elif action == CB_STARTUP_WITHDRAW_UNSTAKED_NFT:
            await handle_startup_withdraw_unstaked_nft_action(context)
        elif action == CB_STARTUP_CONTINUE_MONITORING_STAKED:
            await handle_startup_continue_monitoring_action(context)
        elif action == CB_STARTUP_UNSTAKE_AND_MANAGE_STAKED:
            await handle_startup_unstake_and_manage_action(context)
        # --- END STARTUP LP DISCOVERY ACTIONS ---
        elif action == "start_bot_operations":
            await handle_start_bot_operations_action(context)
        elif action == "pause_bot_operations":
            await handle_pause_bot_operations_action(context)
        elif action == "restart_operations_confirm":
            await send_tg_message(context, "Are you sure you want to restart bot operations? This will attempt to create a new LP position using available funds.", menu_type="restart_confirm")
        elif action == "main_menu":
            await send_tg_message(context, "Main Menu:")
        else:
            await send_tg_message(context, f"Unknown action: {action}")
    
    except Exception as e:
        logger.error(f"Error in button_handler for action {action}: {e}", exc_info=True)
        await send_tg_message(context, f"An error occurred processing your request: {str(e)[:1000]}")
    finally:
        bot_state["is_processing_action"] = False
        await save_state_async()


async def text_message_handler(update: Update, context: CallbackContext):
    if update.effective_user.id != TELEGRAM_ADMIN_USER_ID: return
    
    user_text = update.message.text
    logger.info(f"Received text message: {user_text} from user {update.effective_user.id}")

    if bot_state.get("is_processing_action"):
        await send_tg_message(context, "⚠️ Another action is currently in progress. Please wait.", menu_type=None)
        return
    
    bot_state["is_processing_action"] = True
    try:
        if context.user_data.get('awaiting_profit_withdrawal_amount'):
            del context.user_data['awaiting_profit_withdrawal_amount']
            try:
                amount = Decimal(user_text)
                if amount <= 0:
                    await send_tg_message(context, "Withdrawal amount must be positive.")
                else:
                    await handle_withdraw_profit_custom_action(context, amount)
            except ValueError:
                await send_tg_message(context, "Invalid amount. Please enter a number (e.g., 10.75).")
        
        elif context.user_data.get('awaiting_initial_lp_nft_id'):
            del context.user_data['awaiting_initial_lp_nft_id']
            try:
                nft_id = int(user_text)
                await handle_set_initial_lp_nft_id_action(context, nft_id)
            except ValueError:
                await send_tg_message(context, "Invalid NFT ID. Please enter a number.")
        else:
            await send_tg_message(context, f"I received: {user_text}\nUse the menu buttons for actions.")
    except Exception as e:
        logger.error(f"Error in text_message_handler: {e}", exc_info=True)
        await send_tg_message(context, f"An error occurred processing your message: {str(e)[:1000]}")
    finally:
        bot_state["is_processing_action"] = False
        await save_state_async()

# --- Action Handler (called by button_handler) ---

async def handle_status_action(context: CallbackContext):
    if not check_connection():
        await send_tg_message(context, "Web3 not connected. Cannot fetch status.")
        return

    status_lines = ["📊 **Bot Status**"]
    
    # Wallet Balances
    eth_balance_wei = await asyncio.to_thread(w3.eth.get_balance, BOT_WALLET_ADDRESS)
    status_lines.append(f"🔷 `{from_wei(eth_balance_wei, 18):.6f} ETH`")
    usdc_bal = await get_token_balance(usdc_token_contract, BOT_WALLET_ADDRESS)
    wblt_bal = await get_token_balance(wblt_token_contract, BOT_WALLET_ADDRESS)
    aero_bal = await get_token_balance(aero_token_contract, BOT_WALLET_ADDRESS)
    status_lines.append(f"💰 `{usdc_bal:.2f} USDC`")
    status_lines.append(f"🌯 `{wblt_bal:.4f} WBLT`")
    status_lines.append(f"✈️ `{aero_bal:.4f} AERO`")
    status_lines.append("---")

    # LP Position
    current_nft_id_in_state = bot_state.get("aerodrome_lp_nft_id")

    if current_nft_id_in_state:
        status_lines.append(f"💰 `{current_nft_id_in_state}`")
        position_details = await get_lp_position_details(context, current_nft_id_in_state)
        price_wblt_usdc, current_tick = await get_aerodrome_pool_price_and_tick()

        if position_details and price_wblt_usdc is not None and current_tick is not None:
            tick_lower_lp = position_details['tickLower']
            tick_upper_lp = position_details['tickUpper']
            lp_liquidity = position_details['liquidity']

            wblt_decimals_val = await asyncio.to_thread(wblt_token_contract.functions.decimals().call)
            usdc_decimals_val = await asyncio.to_thread(usdc_token_contract.functions.decimals().call)
            decimal_adj_factor_for_price = Decimal(10)**(wblt_decimals_val - usdc_decimals_val)
            
            price_at_tick_lower_lp = (Decimal("1.0001")**Decimal(tick_lower_lp)) * decimal_adj_factor_for_price
            price_at_tick_upper_lp = (Decimal("1.0001")**Decimal(tick_upper_lp)) * decimal_adj_factor_for_price
                        
            status_lines.append(f"📐 `{tick_lower_lp}` to `{tick_upper_lp}`")
            status_lines.append(f"💲 `{price_at_tick_lower_lp:.4f}` - `{price_at_tick_upper_lp:.4f} USDC` `({TARGET_RANGE_WIDTH_PERCENTAGE}%)`") 

            actual_tick_span_lp = tick_upper_lp - tick_lower_lp
            lower_trigger_tick_for_status = tick_lower_lp
            upper_trigger_tick_for_status = tick_upper_lp

            if actual_tick_span_lp > 0:
                try:
                    pool_tick_spacing = await asyncio.to_thread(aerodrome_pool_contract.functions.tickSpacing().call)
                    if pool_tick_spacing > 0:
                        raw_buffer_in_ticks = actual_tick_span_lp * (REBALANCE_TRIGGER_BUFFER_PERCENTAGE / Decimal(100))
                        num_tick_spacings_for_buffer = int(Decimal(raw_buffer_in_ticks) / Decimal(pool_tick_spacing))
                        if num_tick_spacings_for_buffer == 0 and raw_buffer_in_ticks > 0:
                            num_tick_spacings_for_buffer = 1
                        buffer_tick_amount_aligned = num_tick_spacings_for_buffer * pool_tick_spacing
                        if (2 * buffer_tick_amount_aligned) >= actual_tick_span_lp:
                            if actual_tick_span_lp > pool_tick_spacing : 
                                buffer_tick_amount_aligned = pool_tick_spacing
                            else: 
                                buffer_tick_amount_aligned = 0
                        
                        lower_trigger_tick_for_status = tick_lower_lp + buffer_tick_amount_aligned
                        upper_trigger_tick_for_status = tick_upper_lp - buffer_tick_amount_aligned

                        price_at_lower_trigger = (Decimal("1.0001")**Decimal(lower_trigger_tick_for_status)) * decimal_adj_factor_for_price
                        price_at_upper_trigger = (Decimal("1.0001")**Decimal(upper_trigger_tick_for_status)) * decimal_adj_factor_for_price
                        
                        status_lines.append(
                            f"🔔 Rebalance Triggers: `<{price_at_lower_trigger:.4f}` & `>{price_at_upper_trigger:.4f}` "
                            f"`({REBALANCE_TRIGGER_BUFFER_PERCENTAGE}%)`"
                        )
                    else:
                        status_lines.append("  ❗ Could not determine buffer: Invalid tickSpacing.")
                except Exception as e_buffer_calc:
                    logger.warning(f"Could not calculate buffer trigger prices for status: {e_buffer_calc}")
                    status_lines.append("  ❗ Could not determine buffer trigger prices.")
            else:
                 status_lines.append("  ❗ LP range span is zero or negative, cannot calculate buffer.")

            status_lines.append(f"📈 `{current_tick}`")
            status_lines.append(f"💵 `{price_wblt_usdc:.4f} USDC`")

            staked_status_str = "(Unknown)"
            try:
                is_staked = await asyncio.to_thread(
                    aerodrome_gauge_contract.functions.stakedContains(BOT_WALLET_ADDRESS, current_nft_id_in_state).call
                )
                staked_status_str = "(Staked)" if is_staked else "(Unstaked)"
            except Exception as e_staked_check:
                logger.warning(f"Could not check staked status for NFT {current_nft_id_in_state} in status: {e_staked_check}")

            is_in_actual_range = current_tick >= tick_lower_lp and current_tick < tick_upper_lp
            range_status_emoji = "✅ **In Range**" if is_in_actual_range else "❌ **Out of Range**"
            
            is_within_buffer_zone = current_tick >= lower_trigger_tick_for_status and current_tick < upper_trigger_tick_for_status
            
            status_line_text = f"✨ {range_status_emoji} {staked_status_str}"
            
            if is_in_actual_range:
                if not is_within_buffer_zone and actual_tick_span_lp > 0 :
                     status_line_text += " (Near Edge)"
            else:
                status_line_text += " (Rebalance Pending)"
            status_lines.append(status_line_text)
            
            status_lines.append(f"💧 `{position_details['liquidity']}`")

            actual_wblt_in_lp, actual_usdc_in_lp = await get_amounts_for_liquidity(
                lp_liquidity,
                current_tick,
                tick_lower_lp,
                tick_upper_lp,
                wblt_decimals_val,
                usdc_decimals_val
            )
            status_lines.append(f"💼 `{actual_wblt_in_lp:.4f} WBLT` & `{actual_usdc_in_lp:.2f} USDC`")
            
            est_value_from_actual = (actual_wblt_in_lp * price_wblt_usdc) + actual_usdc_in_lp
            status_lines.append(f"💲 `${est_value_from_actual:.2f}`")
            
            # Uncollected fees (tokensOwed from position_details)
            tokens_owed0_human = Decimal(0)
            tokens_owed1_human = Decimal(0)
            
            # Determine which token is which from the position details
            pos_token0_addr = Web3.to_checksum_address(position_details['token0'])
            # pos_token1_addr = Web3.to_checksum_address(position_details['token1']) # Not needed if we assume the other is USDC

            if pos_token0_addr == WBLT_TOKEN_ADDRESS:
                tokens_owed0_human = from_wei(position_details['tokensOwed0_wei'], wblt_decimals_val)
                tokens_owed1_human = from_wei(position_details['tokensOwed1_wei'], usdc_decimals_val)
                status_lines.append(f"🤑 `{tokens_owed0_human:.4f} WBLT` & `{tokens_owed1_human:.2f} USDC`")
            elif pos_token0_addr == USDC_TOKEN_ADDRESS:
                tokens_owed0_human = from_wei(position_details['tokensOwed0_wei'], usdc_decimals_val)
                tokens_owed1_human = from_wei(position_details['tokensOwed1_wei'], wblt_decimals_val)
                status_lines.append(f"🤑 `{tokens_owed1_human:.4f} WBLT` & `{tokens_owed0_human:.2f} USDC`")
            else:
                status_lines.append("🤑 (Could not determine token order for fees)")

            pending_aero = await get_pending_aero_rewards(context, current_nft_id_in_state)
            status_lines.append(f"🎁 `{pending_aero:.4f} AERO`")
        else:
            status_lines.append("🤷‍♀️ Could not fetch LP position details or pool price.")
    else:
        status_lines.append("❌ No active Aerodrome LP position.")
    status_lines.append("---")

    status_lines.append(f"🧠 `{bot_state['current_strategy']}`")
    profit_value_str = f"{bot_state['accumulated_profit_usdc']:.2f}"
    status_lines.append(f"💸 `${profit_value_str} USDC`")
    status_lines.append(f"🛑 `{'YES' if bot_state['operations_halted'] else 'NO'}`")
    status_lines.append(f"🔒 `{'ENGAGED' if bot_state.get('is_processing_action', False) else 'FREE'}`")
    status_lines.append(f"🛠️ `{'YES' if bot_state.get('initial_setup_pending', True) else 'NO'}`")

    bot_state["last_telegram_status_update_time"] = time.time()
    await save_state_async()
    await send_tg_message(context, "\n".join(status_lines))


async def handle_toggle_strategy_action(context: CallbackContext):
    if bot_state["current_strategy"] == "take_profit":
        bot_state["current_strategy"] = "compound"
    else:
        bot_state["current_strategy"] = "take_profit"
    await save_state_async()
    await send_tg_message(context, f"Strategy changed to: {bot_state['current_strategy']}.")


async def handle_withdraw_profit_menu_action(context: CallbackContext):
    if not USER_PROFIT_WITHDRAWAL_ADDRESS or USER_PROFIT_WITHDRAWAL_ADDRESS == "YOUR_PERSONAL_WALLET_ADDRESS_FOR_PROFITS":
        await send_tg_message(context, "⚠️ Profit withdrawal address is not configured. Please set it in the script.")
        return

    profit_str = f"{bot_state['accumulated_profit_usdc']:.2f} USDC"
    message = (
        f"Available profit for withdrawal: {profit_str}\n"
        f"Withdrawals will be sent to: `{USER_PROFIT_WITHDRAWAL_ADDRESS}`\n\n"
        "Choose an option:"
    )
    await send_tg_message(context, message, menu_type="profit_withdrawal")

async def _execute_profit_withdrawal(context: CallbackContext, amount_decimal: Decimal):
    if amount_decimal <= 0:
        await send_tg_message(context, "Withdrawal amount must be positive.")
        return

    if amount_decimal > bot_state["accumulated_profit_usdc"]:
        await send_tg_message(context, f"Insufficient profit. Requested: {amount_decimal:.2f}, Available: {bot_state['accumulated_profit_usdc']:.2f} USDC.")
        return

    bot_usdc_balance = await asyncio.to_thread(get_token_balance, usdc_token_contract, BOT_WALLET_ADDRESS)
    if bot_usdc_balance < amount_decimal:
        await send_tg_message(context, f"⚠️ Bot's USDC wallet balance ({bot_usdc_balance:.2f}) is less than requested profit withdrawal ({amount_decimal:.2f}). Manual check needed.")
        return

    await send_tg_message(context, f"ℹ️ Withdrawing `{amount_decimal:.2f}` USDC to `{USER_PROFIT_WITHDRAWAL_ADDRESS}`...", menu_type=None)
    
    usdc_decimals = await asyncio.to_thread(usdc_token_contract.functions.decimals().call)
    amount_wei = to_wei(amount_decimal, usdc_decimals)

    tx_params = {
        'from': BOT_WALLET_ADDRESS,
        'nonce': get_nonce(),
    }
    transfer_tx = usdc_token_contract.functions.transfer(USER_PROFIT_WITHDRAWAL_ADDRESS, amount_wei).build_transaction(tx_params)
    
    receipt = await asyncio.to_thread(_send_and_wait_for_transaction, transfer_tx, f"Withdraw {amount_decimal:.2f} USDC Profit")
    if receipt and receipt.status == 1:
        bot_state["accumulated_profit_usdc"] -= amount_decimal
        await save_state_async()
        await send_tg_message(context, f"✅ Successfully withdrew `{amount_decimal:.2f}` USDC. Remaining profit: `{bot_state['accumulated_profit_usdc']:.2f}` USDC.")
    else:
        await send_tg_message(context, f"❌ Profit withdrawal of {amount_decimal:.2f} USDC failed.")


async def handle_withdraw_profit_all_action(context: CallbackContext):
    await _execute_profit_withdrawal(context, bot_state["accumulated_profit_usdc"])

async def handle_withdraw_profit_custom_action(context: CallbackContext, amount: Decimal):
    await _execute_profit_withdrawal(context, amount)


async def handle_view_principal_action(context: CallbackContext):
    message_lines = ["🏦 **Current Principal Details**"]
    message_lines.append(f"  Tracked WBLT in LP: {bot_state['current_lp_principal_wblt_amount']:.4f}")
    message_lines.append(f"  Tracked USDC in LP: {bot_state['current_lp_principal_usdc_amount']:.2f}")

    price_wblt_usdc, _ = await get_aerodrome_pool_price_and_tick()
    if price_wblt_usdc is not None:
        principal_wblt_usd_value = bot_state['current_lp_principal_wblt_amount'] * price_wblt_usdc
        total_principal_usd_value = principal_wblt_usd_value + bot_state['current_lp_principal_usdc_amount']
        message_lines.append(f"  Estimated Total Principal Value: ${total_principal_usd_value:.2f} USD (at current price ${price_wblt_usdc:.4f}/WBLT)")
    else:
        message_lines.append("  Could not fetch current WBLT price to estimate USD value.")
    
    await send_tg_message(context, "\n".join(message_lines))


async def handle_set_initial_lp_nft_id_action(context: CallbackContext, nft_id: int):
    await send_tg_message(context, f"Attempting to load LP NFT ID: {nft_id}...", menu_type=None)
    try:
        owner = await asyncio.to_thread(aerodrome_nft_manager_contract.functions.ownerOf(nft_id).call)
        if owner != BOT_WALLET_ADDRESS:
            await send_tg_message(context, f"⚠️ Bot does not own NFT ID {nft_id}. Current owner: {owner}")
            return

        position_details = await get_lp_position_details(context, nft_id)
        if not position_details:
            await send_tg_message(context, f"⚠️ Could not fetch details for NFT ID {nft_id} or it's not a WBLT/USDC pair.")
            return
                
        bot_state["aerodrome_lp_nft_id"] = nft_id
        bot_state["initial_setup_pending"] = False
        
        await send_tg_message(context, f"✅ LP NFT ID set to {nft_id}. Principal amounts will be accurately set after the next rebalance. Current tracked principals might be approximate.")
        
        bot_state["current_lp_principal_wblt_amount"] = Decimal("0")
        bot_state["current_lp_principal_usdc_amount"] = Decimal("0")

        await save_state_async()
        await handle_status_action(context)

    except Exception as e:
        logger.error(f"Error setting initial LP NFT ID {nft_id}: {e}")
        await send_tg_message(context, f"❌ Error setting LP NFT ID: {e}")


# --- Bot Logic ---
async def process_full_rebalance(context: CallbackContext, triggered_by="auto"):
    if bot_state["operations_halted"]:
        logger.info("Full rebalance skipped: Operations halted.")
        if triggered_by == "manual": await send_tg_message(context, "Full rebalance skipped: Operations halted.", menu_type=None)
        return

    await send_tg_message(context, f"🤖 Initiating Full Rebalance (Trigger: `{triggered_by}`)...", menu_type=None)
    
    original_nft_id = bot_state.get("aerodrome_lp_nft_id")
    wblt_decimals_val = await asyncio.to_thread(wblt_token_contract.functions.decimals().call)
    usdc_decimals_val = await asyncio.to_thread(usdc_token_contract.functions.decimals().call)

    try:
        # 1. Unstake LP from Gauge (if an LP exists)
        if original_nft_id:
            is_staked_check = False
            try:
                is_staked_check = await asyncio.to_thread(
                    aerodrome_gauge_contract.functions.stakedContains(BOT_WALLET_ADDRESS, original_nft_id).call
                )
            except Exception as e_staked_check:
                logger.warning(f"Could not check if NFT {original_nft_id} is staked: {e_staked_check}. Assuming not staked for safety.")
            
            if is_staked_check:
                await send_tg_message(context, f"ℹ️ Unstaking LP `{original_nft_id}`...", menu_type=None)
                unstake_tx_params = {'from': BOT_WALLET_ADDRESS}
                unstake_tx = aerodrome_gauge_contract.functions.withdraw(original_nft_id).build_transaction(unstake_tx_params)
                unstake_receipt = await asyncio.to_thread(_send_and_wait_for_transaction, unstake_tx, f"Unstake LP NFT {original_nft_id}")
                
                if not (unstake_receipt and unstake_receipt.status == 1):
                    await send_tg_message(context, f"⚠️ Failed to unstake LP NFT {original_nft_id}. Rebalance aborted.", menu_type=None)
                    return 
                await send_tg_message(context, f"✅ LP `{original_nft_id}` unstaked!", menu_type=None)
                await asyncio.sleep(13)
                bot_state["last_aero_claim_time"] = time.time()
            else:
                logger.info(f"NFT {original_nft_id} found in state, but not currently staked in gauge. Skipping unstake step.")
                await send_tg_message(context, f"ℹ️ LP `{original_nft_id}` is already in wallet (not staked). Proceeding to withdraw liquidity.", menu_type=None)
        else:
            logger.info("No existing LP NFT ID in state. Proceeding with available wallet funds.")
            await send_tg_message(context, "No existing LP found. Will use wallet funds for new position.", menu_type=None)

        # 2. Withdraw Liquidity from Aerodrome LP (if an LP was unstaked)
        if original_nft_id:
            position_details = await get_lp_position_details(context, original_nft_id)
            
            liquidity_to_withdraw = Decimal(0)
            if position_details and position_details['liquidity'] > 0:
                liquidity_to_withdraw = Decimal(position_details['liquidity'])

            if liquidity_to_withdraw > 0:
                await send_tg_message(context, f"ℹ️ Withdrawing liquidity `({liquidity_to_withdraw})` from LP `{original_nft_id}`...", menu_type=None)
                decrease_params = {
                    'tokenId': original_nft_id, 'liquidity': int(liquidity_to_withdraw),
                    'amount0Min': 0, 'amount1Min': 0, 'deadline': int(time.time()) + 600 
                }
                decrease_tx_params = {'from': BOT_WALLET_ADDRESS}
                decrease_tx = aerodrome_nft_manager_contract.functions.decreaseLiquidity(decrease_params).build_transaction(decrease_tx_params)
                decrease_receipt = await asyncio.to_thread(_send_and_wait_for_transaction, decrease_tx, f"Decrease Liquidity NFT {original_nft_id}")
                
                if decrease_receipt and decrease_receipt.status == 1:
                    await send_tg_message(context, f"✅ Liquidity decrease successful for LP `{original_nft_id}`!", menu_type=None)
                    await asyncio.sleep(13)

                    await send_tg_message(context, f"ℹ️ Collecting tokens...", menu_type=None)
                    collect_params = {
                        'tokenId': original_nft_id, 'recipient': BOT_WALLET_ADDRESS,
                        'amount0Max': 2**128 - 1, 'amount1Max': 2**128 - 1
                    }
                    collect_tx_params = {'from': BOT_WALLET_ADDRESS}
                    collect_tx = aerodrome_nft_manager_contract.functions.collect(collect_params).build_transaction(collect_tx_params)
                    collect_receipt_obj = await asyncio.to_thread(_send_and_wait_for_transaction, collect_tx, f"Collect Tokens NFT {original_nft_id}")
                    
                    if collect_receipt_obj and collect_receipt_obj.status == 1:
                        await send_tg_message(context, f"✅ Tokens collected for LP `{original_nft_id}`!", menu_type=None)
                        await asyncio.sleep(13)
                    else:
                        await send_tg_message(context, f"⚠️ Failed to collect tokens for LP `{original_nft_id}`. Wallet balance will be used.", menu_type=None)
                else:
                    await send_tg_message(context, f"⚠️ Failed to decrease liquidity for LP `{original_nft_id}`. Rebalance aborted.", menu_type=None)
                    return
            
            elif position_details and position_details['liquidity'] == 0:
                await send_tg_message(context, f"ℹ️ LP `{original_nft_id}` already has 0 liquidity. Proceeding to burn.", menu_type=None)
            elif not position_details:
                 await send_tg_message(context, f"⚠️ Could not fetch details for LP `{original_nft_id}`. Attempting to burn if it exists.", menu_type=None)
            else:
                await send_tg_message(context, f"ℹ️ No active liquidity in LP `{original_nft_id}` to withdraw. Proceeding to burn check.", menu_type=None)

            logger.info(f"Checking state of NFT {original_nft_id} before attempting burn...")
            current_details_for_burn = await get_lp_position_details(context, original_nft_id)
            
            can_attempt_burn = False
            if current_details_for_burn:
                if current_details_for_burn['liquidity'] == 0:
                    logger.info(f"NFT {original_nft_id} confirmed to have 0 liquidity. Proceeding with burn.")
                    can_attempt_burn = True
                else:
                    logger.warning(f"NFT {original_nft_id} still has liquidity {current_details_for_burn['liquidity']} before burn attempt. This is unexpected if withdrawal was successful. Burn will be skipped.")
                    await send_tg_message(context, f"⚠️ LP `{original_nft_id}` not empty. Burn skipped. Manual check needed.", menu_type=None)
            else:
                try:
                    owner = await asyncio.to_thread(aerodrome_nft_manager_contract.functions.ownerOf(original_nft_id).call)
                    if owner == BOT_WALLET_ADDRESS:
                        logger.warning(f"Could not get position details for NFT {original_nft_id} owned by bot, but will attempt burn. It might be an invalid/corrupted NFT state.")
                        can_attempt_burn = True
                    else:
                        logger.warning(f"NFT {original_nft_id} is not owned by the bot ({owner}). Cannot burn.")
                except Exception:
                    logger.info(f"NFT {original_nft_id} likely does not exist or already burned (ownerOf reverted). Burn will be skipped.")

            if can_attempt_burn:
                await send_tg_message(context, f"ℹ️ Burning LP `{original_nft_id}`...", menu_type=None)
                burn_tx_params = {'from': BOT_WALLET_ADDRESS}
                burn_tx = aerodrome_nft_manager_contract.functions.burn(original_nft_id).build_transaction(burn_tx_params)
                burn_receipt = await asyncio.to_thread(_send_and_wait_for_transaction, burn_tx, f"Burn LP NFT {original_nft_id}")

                if burn_receipt and burn_receipt.status == 1:
                    await send_tg_message(context, f"✅ LP `{original_nft_id}` burned!", menu_type=None)
                else:
                    await send_tg_message(context, f"⚠️ Failed to burn LP `{original_nft_id}` (it might have been already burned or an issue occurred).", menu_type=None)
                await asyncio.sleep(13)

            bot_state["aerodrome_lp_nft_id"] = None

        # 3. Consolidate funds
        available_wblt = await get_token_balance(wblt_token_contract, BOT_WALLET_ADDRESS)
        available_usdc = await get_token_balance(usdc_token_contract, BOT_WALLET_ADDRESS)
        available_aero = await get_token_balance(aero_token_contract, BOT_WALLET_ADDRESS)
        logger.info(f"Consolidated Funds - WBLT: {available_wblt}, USDC: {available_usdc}, AERO: {available_aero}")

        # 4. Sell ALL AERO
        if available_aero > Decimal("1.0"):
            await send_tg_message(context, f"ℹ️ Selling `{available_aero:.6f}` AERO for USDC...", menu_type=None)
            swap_aero_op = functools.partial(execute_kyberswap_swap, context, aero_token_contract, USDC_TOKEN_ADDRESS, available_aero)
            swap_success, usdc_amount_from_aero = await attempt_operation_with_retries(
                swap_aero_op, "Sell AERO via KyberSwap", context
            )
            if swap_success:
                usdc_from_aero_sale = usdc_amount_from_aero if usdc_amount_from_aero else Decimal("0")
                available_usdc += usdc_from_aero_sale
                if bot_state["current_strategy"] == "take_profit":
                    bot_state["accumulated_profit_usdc"] += usdc_from_aero_sale
                    await send_tg_message(context, f"✅ AERO sold! Profit of `{usdc_from_aero_sale:.2f}` USDC added.", menu_type=None)
                    await asyncio.sleep(13)
                else:
                    await send_tg_message(context, f"✅ AERO sold! `{usdc_from_aero_sale:.2f}` USDC to be compounded.", menu_type=None)
                    await asyncio.sleep(13)
            else:
                await send_tg_message(context, "⚠️ Failed to sell AERO after retries. Continuing rebalance with current funds.", menu_type=None)
        logger.info(f"Funds after AERO sale - WBLT: {available_wblt}, USDC: {available_usdc}")
        
        # 5. Determine New Optimal LP Range
        price_wblt_human_for_range, pool_current_tick = await get_aerodrome_pool_price_and_tick()
        slot0_data = await asyncio.to_thread(aerodrome_pool_contract.functions.slot0().call)
        current_pool_sqrt_price_x96_raw = slot0_data[0]
        if price_wblt_human_for_range is None or pool_current_tick is None:
            await send_tg_message(context, "❌ Cannot get current pool price/tick. Rebalance aborted.", menu_type=None)
            return
        logger.info(f"Centering new LP range around pool price: ${price_wblt_human_for_range:.6f} (tick {pool_current_tick})")

        tick_spacing = await asyncio.to_thread(aerodrome_pool_contract.functions.tickSpacing().call)
        wblt_decimals_val = await asyncio.to_thread(wblt_token_contract.functions.decimals().call)
        usdc_decimals_val = await asyncio.to_thread(usdc_token_contract.functions.decimals().call)
        logger.info(f"Pool {AERODROME_CL_POOL_ADDRESS} uses tickSpacing: {tick_spacing}")

        current_target_range_width = TARGET_RANGE_WIDTH_PERCENTAGE
        logger.info(f"Using target range width: {current_target_range_width}% for this mint attempt.")

        new_tick_lower, new_tick_upper = await asyncio.to_thread(
            calculate_ticks_for_range,
            price_wblt_human_for_range,
            current_target_range_width,
            tick_spacing,
            wblt_decimals_val,
            usdc_decimals_val
        )
        logger.info(f"Calculated new LP ticks using calculate_ticks_for_range: Lower={new_tick_lower}, Upper={new_tick_upper}")
        await send_tg_message(context, f"ℹ️ New target LP range: Ticks `{new_tick_lower}, {new_tick_upper}`.", menu_type=None)

        # 6. Determine final WBLT & USDC amounts for mint by targeting optimal ratio for ALL capital
        logger.info("Performing targeted swap (if needed) to optimize token ratio for the chosen LP range using all available capital...")
        final_wblt_for_mint, final_usdc_for_mint = await _perform_targeted_swap_for_optimal_ratio(
            context,
        current_pool_sqrt_price_x96_raw,
        new_tick_lower, new_tick_upper,
        available_wblt, available_usdc, 
        price_wblt_human_for_range,
        wblt_decimals_val,
        usdc_decimals_val
        )
        logger.info(f"Final balances intended for mint (after potential targeted swap): WBLT={final_wblt_for_mint:.8f}, USDC={final_usdc_for_mint:.8f}")
        await asyncio.sleep(13)

        desired_wblt_wei = to_wei(final_wblt_for_mint, wblt_decimals_val)
        desired_usdc_wei = to_wei(final_usdc_for_mint, usdc_decimals_val)
        
        logger.info(f"Desired amounts for mint (wei): WBLT_wei={desired_wblt_wei}, USDC_wei={desired_usdc_wei}")

        # 7. Deposit Liquidity (Mint)
        min_deposit_value_usd = Decimal("1.0") 
        value_of_desired_wblt = final_wblt_for_mint * price_wblt_human_for_range
        value_of_desired_usdc = final_usdc_for_mint 
        total_desired_value_for_mint = value_of_desired_wblt + value_of_desired_usdc

        if total_desired_value_for_mint < min_deposit_value_usd:
            await send_tg_message(context, f"❌ Value of assets intended for mint (${total_desired_value_for_mint:.2f}) is below threshold (${min_deposit_value_usd:.2f}). Aborting mint.", menu_type=None)
            logger.warning(f"Mint aborted: value of assets for mint ${total_desired_value_for_mint:.2f} is too low. WBLT: {desired_wblt_wei}, USDC: {desired_usdc_wei}")
            return

        await send_tg_message(context, f"ℹ️ Preparing to mint new LP with: ~**{final_wblt_for_mint:.4f} WBLT** and ~**{final_usdc_for_mint:.2f} USDC**...", menu_type=None)
        
        # Approvals for the calculated desired amounts.
        if desired_wblt_wei > 0:
            approve_wblt_amount_dec_buffered = final_wblt_for_mint * Decimal("1.001") 
            approved_wblt = await approve_token_spending(context, wblt_token_contract, AERODROME_SLIPSTREAM_NFT_MANAGER_ADDRESS, approve_wblt_amount_dec_buffered)
            if not approved_wblt:
                logger.error("WBLT approval failed. Minting aborted.")
                return
            logger.info("WBLT approval for optimal amount successful or already sufficient.")
            await asyncio.sleep(13)

        if desired_usdc_wei > 0:
            approve_usdc_amount_dec_buffered = final_usdc_for_mint * Decimal("1.001")
            approved_usdc = await approve_token_spending(context, usdc_token_contract, AERODROME_SLIPSTREAM_NFT_MANAGER_ADDRESS, approve_usdc_amount_dec_buffered)
            if not approved_usdc:
                logger.error("USDC approval failed. Minting aborted.")
                return
            logger.info("USDC approval for optimal amount successful or already sufficient.")
            await asyncio.sleep(13)

        # Mint parameters
        amount0_min_wei = 0 
        amount1_min_wei = 0
        sqrt_price_x96_limit_for_mint = 0

        mint_params_as_tuple = (
            Web3.to_checksum_address(WBLT_TOKEN_ADDRESS), Web3.to_checksum_address(USDC_TOKEN_ADDRESS),
            int(tick_spacing), int(new_tick_lower), int(new_tick_upper),
            int(desired_wblt_wei),
            int(desired_usdc_wei),
            int(amount0_min_wei), int(amount1_min_wei),
            Web3.to_checksum_address(BOT_WALLET_ADDRESS), int(time.time()) + 600,
            int(sqrt_price_x96_limit_for_mint)
        )
        logger.info(f"Attempting to mint with 12-ELEMENT params tuple (optimal amounts): {mint_params_as_tuple}")

        mint_receipt = None
        try:
            if aerodrome_nft_manager_contract is None:
                logger.critical("CRITICAL during mint: aerodrome_nft_manager_contract is None!")
                await send_tg_message(context, "❌ CRITICAL ERROR: NFT Manager contract not loaded. Aborting mint.", menu_type=None)
                return

            prepared_mint_function_call = aerodrome_nft_manager_contract.functions.mint(mint_params_as_tuple)
            encoded_mint_data = prepared_mint_function_call._encode_transaction_data()
            tx_for_estimation_or_call = {
                'from': BOT_WALLET_ADDRESS, 'to': AERODROME_SLIPSTREAM_NFT_MANAGER_ADDRESS,
                'data': encoded_mint_data,
            }

            logger.warning("Attempting static call (w3.eth.call) for mint with optimal amounts...")
            await asyncio.to_thread(w3.eth.call, tx_for_estimation_or_call, 'latest')
            logger.info("Static call for mint with optimal amounts SUCCEEDED (or did not revert).")

            await send_tg_message(context, f"ℹ️ Minting new LP position...", menu_type=None)

            final_mint_tx_params_to_send = tx_for_estimation_or_call 
            mint_receipt = await asyncio.to_thread(
                 _send_and_wait_for_transaction,
                 final_mint_tx_params_to_send,
                 "Mint New Aerodrome LP (Optimal Amts)"
            )

        except ContractLogicError as cle:
            logger.error(f"ContractLogicError during mint (optimal amounts) prep/static call: {getattr(cle, 'message', str(cle))}", exc_info=True)
            await send_tg_message(context, f"❌ Minting FAILED (Optimal Amts - Contract Revert during Prep): {getattr(cle, 'message', str(cle))}", menu_type=None)
            return
        except Exception as e:
            logger.error(f"Unexpected error during mint (optimal amounts) preparation: {e}", exc_info=True)
            await send_tg_message(context, f"❌ Minting failed unexpectedly (Optimal Amts - Prep): {e}", menu_type=None)
            return

        # --- Check mint_receipt and proceed ---
        if not (mint_receipt and mint_receipt.status == 1):
            await send_tg_message(context, "❌ Minting new LP (Optimal Amts) transaction FAILED or was not confirmed after retries.", menu_type=None)
            return
        
        new_nft_id = None
        actual_wblt_deposited_wei = Decimal("0")
        actual_usdc_deposited_wei = Decimal("0")
        erc721_transfer_event_signature = w3.keccak(text="Transfer(address,address,uint256)").hex()
        for log_entry in mint_receipt.get('logs', []):
            if log_entry['address'] == AERODROME_SLIPSTREAM_NFT_MANAGER_ADDRESS and \
                len(log_entry['topics']) == 4 and log_entry['topics'][0].hex() == erc721_transfer_event_signature:
                if int(log_entry['topics'][1].hex(), 16) == 0:
                    to_address_str = "0x" + log_entry['topics'][2].hex()[-40:]
                    if Web3.to_checksum_address(to_address_str) == BOT_WALLET_ADDRESS:
                        new_nft_id = w3.to_int(hexstr=log_entry['topics'][3].hex())
                        logger.info(f"Found Transfer event for new NFT ID: {new_nft_id} to bot wallet.")
                        break
        if new_nft_id is None:
            await send_tg_message(context, "❌ Could not find new LP NFT ID from mint events. Manual check needed.", menu_type=None)
            return
        await send_tg_message(context, f"✅ New LP position `{new_nft_id}` minted!", menu_type=None)
        await asyncio.sleep(13)

        increase_liquidity_event_signature = w3.keccak(text="IncreaseLiquidity(uint256,uint128,uint256,uint256)").hex()
        found_increase_liquidity_event = False
        for log_entry_il in mint_receipt.get('logs', []):
            if log_entry_il['address'] == AERODROME_SLIPSTREAM_NFT_MANAGER_ADDRESS and \
                len(log_entry_il['topics']) > 1 and log_entry_il['topics'][0].hex() == increase_liquidity_event_signature:
                event_token_id = w3.to_int(hexstr=log_entry_il['topics'][1].hex())
                if event_token_id == new_nft_id:
                    found_increase_liquidity_event = True
                    data_hex_str = log_entry_il['data'].hex() if isinstance(log_entry_il['data'], HexBytes) else log_entry_il['data']
                    if data_hex_str.startswith("0x"): data_hex_str = data_hex_str[2:]
                    decoded_data = w3.codec.decode(['uint128','uint256','uint256'], bytes.fromhex(data_hex_str))
                    actual_wblt_deposited_wei = Decimal(decoded_data[1])
                    actual_usdc_deposited_wei = Decimal(decoded_data[2])
                    logger.info(f"Decoded from IncreaseLiquidity for NFT {new_nft_id}: WBLT_wei={actual_wblt_deposited_wei}, USDC_wei={actual_usdc_deposited_wei}")
                    break
        if not found_increase_liquidity_event:
            await send_tg_message(context, f"⚠️ Could not parse IncreaseLiquidity event for NFT {new_nft_id}. Principals may be inaccurate.", menu_type=None)

        # 9. Update Principal & State
        bot_state["aerodrome_lp_nft_id"] = new_nft_id
        bot_state["current_lp_principal_wblt_amount"] = from_wei(actual_wblt_deposited_wei, wblt_decimals_val)
        bot_state["current_lp_principal_usdc_amount"] = from_wei(actual_usdc_deposited_wei, usdc_decimals_val)
        bot_state["initial_setup_pending"] = False 
        logger.info(f"Updated LP Principal: {bot_state['current_lp_principal_wblt_amount']:.{wblt_decimals_val}f} WBLT, {bot_state['current_lp_principal_usdc_amount']:.{usdc_decimals_val}f} USDC from mint of NFT {new_nft_id}.")

        # 10. Stake New LP NFT
        approved_nft_for_gauge = await approve_nft_for_spending(context, aerodrome_nft_manager_contract, AERODROME_CL_GAUGE_ADDRESS, new_nft_id)
        if not approved_nft_for_gauge:
            logger.error(f"Failed to approve NFT {new_nft_id} for staking. Staking aborted.")
            await send_tg_message(context, f"❌ Failed to approve NFT {new_nft_id} for staking. Staking aborted.", menu_type=None)
            return
        else:
            await asyncio.sleep(13)
            await send_tg_message(context, f"ℹ️ Staking new LP `{new_nft_id}`...", menu_type=None)
            stake_tx_params = {'from': BOT_WALLET_ADDRESS}
            stake_tx = aerodrome_gauge_contract.functions.deposit(new_nft_id).build_transaction(stake_tx_params)
            stake_receipt = await asyncio.to_thread(_send_and_wait_for_transaction, stake_tx, f"Stake LP NFT {new_nft_id}")
            if not (stake_receipt and stake_receipt.status == 1):
                await send_tg_message(context, f"⚠️ Failed to stake LP NFT {new_nft_id}. Remains in wallet.", menu_type=None)
            else:
                # await asyncio.sleep(13) # probably not needed as this is the last transaction in the rebalance
                await send_tg_message(context, f"✅ LP `{new_nft_id}` staked!", menu_type=None)

    except Exception as e:
        logger.error(f"Critical error during process_full_rebalance: {e}", exc_info=True)
        await send_tg_message(context, f"⚠️ Critical error during rebalance: {str(e)[:200]}", menu_type=None)
    finally:
        await save_state_async()
        await send_tg_message(context, "🤖 Full Rebalance Process Attempt Finished.", menu_type=None)
        await asyncio.sleep(1) 
        await handle_status_action(context)


async def process_claim_sell_aero(context: CallbackContext, triggered_by="auto"):
    if bot_state["operations_halted"] or not bot_state["aerodrome_lp_nft_id"]:
        logger.info("AERO claim/sell skipped: Operations halted or no LP NFT.")
        if bot_state["operations_halted"]: await send_tg_message(context, "AERO claim/sell skipped: Operations halted.", menu_type=None)
        return

    await send_tg_message(context, f"ℹ️ Initiating AERO Claim & Sell (Trigger: `{triggered_by}`)...", menu_type=None)
    
    try:
        # 1. Claim AERO from Gauge
        nft_id_to_claim = bot_state["aerodrome_lp_nft_id"]
        pending_aero_before_claim = await get_pending_aero_rewards(context, nft_id_to_claim)
        
        if pending_aero_before_claim < Decimal("1.0"):
            await send_tg_message(context, "No significant AERO rewards to claim.", menu_type=None)
            bot_state["last_aero_claim_time"] = time.time()
            await save_state_async()
            return

        await send_tg_message(context, f"ℹ️ Attempting to claim `{pending_aero_before_claim:.6f}` AERO for NFT `{nft_id_to_claim}`...", menu_type=None)
        claim_tx_params = {'from': BOT_WALLET_ADDRESS, 'nonce': get_nonce()}
        claim_tx = aerodrome_gauge_contract.functions.getReward(nft_id_to_claim).build_transaction(claim_tx_params)
        claim_receipt = await asyncio.to_thread(_send_and_wait_for_transaction, claim_tx, f"Claim AERO for NFT {nft_id_to_claim}")
        if not (claim_receipt and claim_receipt.status == 1):
            await send_tg_message(context, "❌ AERO claim transaction failed or not confirmed.", menu_type=None)
            return      

        await send_tg_message(context, "✅ AERO rewards claimed successfully.", menu_type=None)
        await asyncio.sleep(13)
        bot_state["last_aero_claim_time"] = time.time()

        # 2. Get bot's AERO balance (should include newly claimed AERO)
        aero_balance = await get_token_balance(aero_token_contract, BOT_WALLET_ADDRESS)
        
        if aero_balance < Decimal("1.0"):
            await send_tg_message(context, "No significant AERO in wallet to sell after claim attempt.", menu_type=None)
            await save_state_async()
            return

        # 3. Sell ALL AERO for USDC (KyberSwap)
        await send_tg_message(context, f"ℹ️ Attempting to sell `{aero_balance:.6f}` AERO for USDC via KyberSwap...", menu_type=None)
        swap_aero_op = functools.partial(execute_kyberswap_swap, context, aero_token_contract, USDC_TOKEN_ADDRESS, aero_balance)
        swap_success, usdc_received_from_claim_sale = await attempt_operation_with_retries(
            swap_aero_op, "Sell Claimed AERO via KyberSwap", context
        )

        if swap_success:
            usdc_amount = usdc_received_from_claim_sale if usdc_received_from_claim_sale else Decimal("0")
            if bot_state["current_strategy"] == "take_profit":
                bot_state["accumulated_profit_usdc"] += usdc_amount
                await send_tg_message(context, f"✅ Sold claimed AERO for `{usdc_amount:.2f}` USDC. Profit added. Total profit: `${bot_state['accumulated_profit_usdc']:.2f}` USDC.")
            else:
                await send_tg_message(context, f"✅ Sold claimed AERO for `{usdc_amount:.2f}` USDC. This amount is now in the bot's wallet and will be compounded during the next rebalance.")
        else:
            await send_tg_message(context, f"❌ Failed to sell claimed AERO for USDC. AERO remains in bot wallet.", menu_type=None)

    except Exception as e:
        logger.error(f"Error during process_claim_sell_aero: {e}", exc_info=True)
        await send_tg_message(context, f"⚠️ Error during AERO claim/sell: {str(e)[:200]}", menu_type=None)
    finally:
        await save_state_async()


async def handle_claim_sell_aero_action(context: CallbackContext):
    action_description = "Claim & Sell AERO"
    if bot_state.get("operations_halted", True):
        await send_tg_message(context, f"Operations were paused. Resuming to {action_description}...", menu_type=None)
        bot_state["operations_halted"] = False
        await save_state_async()
        await asyncio.sleep(0.5) 
        
        # Re-check the processing lock in case of rapid clicks, though the main one should catch it.
        if bot_state.get("is_processing_action"):
             pass

    await process_claim_sell_aero(context, triggered_by="manual")

async def handle_force_rebalance_action(context: CallbackContext):
    action_description = "Force Rebalance"
    if bot_state.get("operations_halted", True):
        await send_tg_message(context, f"Operations were paused. Resuming to {action_description}...", menu_type=None)
        bot_state["operations_halted"] = False
        await save_state_async()
        await asyncio.sleep(0.5) 
    
    await process_full_rebalance(context, triggered_by="manual")


async def handle_emergency_exit_action(context: CallbackContext):
    await send_tg_message(context, "🚨 **EMERGENCY EXIT INITIATED!** Attempting to withdraw all funds and halt operations...", menu_type=None)
    
    # 1. Unstake if an LP NFT ID exists
    if bot_state["aerodrome_lp_nft_id"]:
        await send_tg_message(context, "ℹ️ Unstaking LP from gauge...", menu_type=None)
        unstake_tx_params = { 'from': BOT_WALLET_ADDRESS, 'nonce': get_nonce() }
        unstake_tx = aerodrome_gauge_contract.functions.withdraw(bot_state["aerodrome_lp_nft_id"]).build_transaction(unstake_tx_params)
        unstake_receipt = await asyncio.to_thread(_send_and_wait_for_transaction, unstake_tx, "Emergency Unstake")
        if not unstake_receipt or unstake_receipt.status != 1:
            await send_tg_message(context, "⚠️ Failed to unstake LP. Manual intervention may be required.", menu_type=None)
        else:
            await send_tg_message(context, "✅ LP NFT unstaked!", menu_type=None)
            await asyncio.sleep(13)

    # 2. Withdraw liquidity if an LP NFT ID exists
    if bot_state["aerodrome_lp_nft_id"]:
        position_details = await get_lp_position_details(context, bot_state["aerodrome_lp_nft_id"])
        if position_details and position_details['liquidity'] > 0:
            await send_tg_message(context, f"ℹ️ Withdrawing liquidity `({position_details['liquidity']})` from LP `{bot_state['aerodrome_lp_nft_id']}`...", menu_type=None)
            
            # Decrease Liquidity
            decrease_params = {
                'tokenId': bot_state["aerodrome_lp_nft_id"],
                'liquidity': position_details['liquidity'],
                'amount0Min': 0,
                'amount1Min': 0,
                'deadline': int(time.time()) + 600 # 10 min deadline
            }
            decrease_tx_params = {'from': BOT_WALLET_ADDRESS, 'nonce': get_nonce()}
            decrease_tx = aerodrome_nft_manager_contract.functions.decreaseLiquidity(decrease_params).build_transaction(decrease_tx_params)
            decrease_receipt = await asyncio.to_thread(_send_and_wait_for_transaction, decrease_tx, "Emergency Decrease Liquidity")

            if decrease_receipt and decrease_receipt.status == 1:
                await send_tg_message(context, "✅ Liquidity withdrawn successfully!", menu_type=None)
                await asyncio.sleep(13)
                # Collect Tokens
                collect_params = {
                    'tokenId': bot_state["aerodrome_lp_nft_id"],
                    'recipient': BOT_WALLET_ADDRESS,
                    'amount0Max': 2**128 -1,
                    'amount1Max': 2**128 -1
                }
                await send_tg_message(context, f"ℹ️ Collecting tokens...", menu_type=None)
                collect_tx_params = {'from': BOT_WALLET_ADDRESS, 'nonce': get_nonce()}
                collect_tx = aerodrome_nft_manager_contract.functions.collect(collect_params).build_transaction(collect_tx_params)
                collect_receipt = await asyncio.to_thread(_send_and_wait_for_transaction, collect_tx, "Emergency Collect Tokens")
                if collect_receipt and collect_receipt.status == 1:
                    await send_tg_message(context, "✅ Tokens collected!", menu_type=None)
                    await asyncio.sleep(13)
                else:
                    await send_tg_message(context, "⚠️ Failed to collect tokens after decreasing liquidity.", menu_type=None)
            else:
                await send_tg_message(context, "⚠️ Failed to decrease liquidity.", menu_type=None)
        else:
            await send_tg_message(context, "No liquidity found in the LP NFT or details unavailable.", menu_type=None)

    # 3. Claim any AERO from gauge (might have been auto-claimed on unstake, but try again)
    if bot_state["aerodrome_lp_nft_id"]:
        try:
            pending_aero_rewards = await get_pending_aero_rewards(context, bot_state["aerodrome_lp_nft_id"])
            if pending_aero_rewards > Decimal("0.1"):
                await send_tg_message(context, f"ℹ️ Attempting to claim `{pending_aero_rewards:.4f}` AERO...", menu_type=None)
                claim_tx_params = {'from': BOT_WALLET_ADDRESS, 'nonce': get_nonce()}
                claim_tx = aerodrome_gauge_contract.functions.getReward(bot_state["aerodrome_lp_nft_id"]).build_transaction(claim_tx_params)
                claim_receipt = await asyncio.to_thread(_send_and_wait_for_transaction, claim_tx, "Emergency Claim AERO")
                await asyncio.sleep(13)
                if claim_receipt and claim_receipt.status == 1:
                     await send_tg_message(context, "✅ AERO claimed!", menu_type=None)
                else:
                     await send_tg_message(context, "⚠️ Failed to claim AERO or no AERO to claim.", menu_type=None)
        except Exception as e:
            logger.warning(f"Could not attempt emergency AERO claim for NFT {bot_state['aerodrome_lp_nft_id']}: {e}")


    # 4. Sell all WBLT for USDC
    wblt_balance = await get_token_balance(wblt_token_contract, BOT_WALLET_ADDRESS)
    if wblt_balance > Decimal("0.1"):
        await send_tg_message(context, f"ℹ️ Selling `{wblt_balance:.4f}` WBLT for USDC...", menu_type=None)
        success, _ = await execute_kyberswap_swap(context, wblt_token_contract, USDC_TOKEN_ADDRESS, wblt_balance)
        if success:
            await send_tg_message(context, "✅ WBLT sold for USDC!", menu_type=None)
            await asyncio.sleep(13)
        else:
            await send_tg_message(context, "⚠️ Failed to sell WBLT.", menu_type=None)

    # 5. Sell all AERO for USDC
    aero_balance = await get_token_balance(aero_token_contract, BOT_WALLET_ADDRESS)
    if aero_balance > Decimal("0.1"):
        await send_tg_message(context, f"ℹ️ Selling `{aero_balance:.4f}` AERO for USDC...", menu_type=None)
        success, _ = await execute_kyberswap_swap(context, aero_token_contract, USDC_TOKEN_ADDRESS, aero_balance)
        if success:
            await send_tg_message(context, "✅ AERO sold for USDC!", menu_type=None)
        else:
            await send_tg_message(context, "⚠️ Failed to sell AERO.", menu_type=None)

    # Finalize Exit
    bot_state["operations_halted"] = True
    bot_state["aerodrome_lp_nft_id"] = None
    bot_state["current_lp_principal_wblt_amount"] = Decimal("0")
    bot_state["current_lp_principal_usdc_amount"] = Decimal("0")
    bot_state["initial_setup_pending"] = True

    final_usdc_balance = await get_token_balance(usdc_token_contract, BOT_WALLET_ADDRESS)
    await save_state_async()
    await send_tg_message(context, f"🚨 **EMERGENCY EXIT COMPLETE!** Operations halted. Bot wallet has `{final_usdc_balance:.2f}` USDC. Please verify all transactions.", menu_type="main") # Show main menu which will now have restart button


async def handle_start_bot_operations_action(context: CallbackContext):
    if not bot_state.get("operations_halted", True):
        await send_tg_message(context, "Bot operations are already active.")
        return

    bot_state["operations_halted"] = False
    await save_state_async()
    await send_tg_message(context, "✅ Bot operations started/resumed. Main loop is now active.")

async def handle_pause_bot_operations_action(context: CallbackContext):
    if bot_state.get("operations_halted", True):
        await send_tg_message(context, "Bot operations are already paused.")
        return
    bot_state["operations_halted"] = True
    await save_state_async()
    await send_tg_message(context, "⏸️ Bot operations PAUSED. Main loop will not perform automated actions. You can resume via the menu.")

async def handle_startup_stake_nft_action(context: CallbackContext):
    nft_id = bot_state.get("aerodrome_lp_nft_id")
    if not nft_id:
        await send_tg_message(context, "⚠️ No NFT ID found in state to stake. Please restart bot.", menu_type="main")
        return

    await send_tg_message(context, f"ℹ️ Approving LP `{nft_id}` to stake...", menu_type=None)
    
    approved_nft_for_gauge = await approve_nft_for_spending(context, aerodrome_nft_manager_contract, AERODROME_CL_GAUGE_ADDRESS, nft_id)

    if not approved_nft_for_gauge:
        await send_tg_message(context, f"❌ Failed to approve NFT {nft_id} for staking. Staking aborted. Bot remains HALTED.", menu_type="main")
        bot_state["operations_halted"] = True
        return
    await asyncio.sleep(13)

    await send_tg_message(context, f"ℹ️ Staking LP...", menu_type=None)
    stake_tx_params = {'from': BOT_WALLET_ADDRESS, 'nonce': await asyncio.to_thread(get_nonce), 'chainId': await asyncio.to_thread(lambda: w3.eth.chain_id)}
    stake_tx = aerodrome_gauge_contract.functions.deposit(nft_id).build_transaction(stake_tx_params)
    stake_receipt = await asyncio.to_thread(_send_and_wait_for_transaction, stake_tx, f"Stake LP NFT {nft_id}")
    if stake_receipt and stake_receipt.status == 1:
        await send_tg_message(context, f"✅ LP {nft_id} successfully STAKED! Resuming normal operations.", menu_type="main")
        await asyncio.sleep(13)
        bot_state["operations_halted"] = False
        bot_state["initial_setup_pending"] = False
    else:
        await send_tg_message(context, f"⚠️ Failed to stake LP NFT {nft_id}. Remains in wallet. Bot HALTED.", menu_type="main")
        bot_state["operations_halted"] = True
    await save_state_async()
    await handle_status_action(context)

async def handle_startup_withdraw_unstaked_nft_action(context: CallbackContext):
    nft_id = bot_state.get("aerodrome_lp_nft_id")
    if not nft_id:
        await send_tg_message(context, "⚠️ No NFT ID found in state to withdraw. Please restart bot.", menu_type="main")
        bot_state["operations_halted"] = True
        await save_state_async()
        return
    
    wblt_decimals_val = await asyncio.to_thread(wblt_token_contract.functions.decimals().call)
    # usdc_decimals_val = await asyncio.to_thread(usdc_token_contract.functions.decimals().call) # Not strictly needed here if only converting to USDC

    # --- 1. Withdraw Liquidity from Aerodrome LP (NFT is already in wallet) ---
    position_details = await get_lp_position_details(context, nft_id)

    if position_details and position_details['liquidity'] > 0:
        await send_tg_message(context, f"ℹ️ Withdrawing liquidity `({position_details['liquidity']})` from LP `{nft_id}`...", menu_type=None)
        
        # Decrease Liquidity
        decrease_params = {
            'tokenId': nft_id,
            'liquidity': position_details['liquidity'],
            'amount0Min': 0,
            'amount1Min': 0,
            'deadline': int(time.time()) + 600 # 10 min deadline
        }
        decrease_tx_params = {
            'from': BOT_WALLET_ADDRESS, 
            'nonce': await asyncio.to_thread(get_nonce), 
            'chainId': await asyncio.to_thread(lambda: w3.eth.chain_id)
        }
        decrease_tx = aerodrome_nft_manager_contract.functions.decreaseLiquidity(decrease_params).build_transaction(decrease_tx_params)
        decrease_receipt = await asyncio.to_thread(_send_and_wait_for_transaction, decrease_tx, f"Decrease Liquidity NFT {nft_id} (Startup)")
        
        if decrease_receipt and decrease_receipt.status == 1:
            await send_tg_message(context, f"✅ Withdrawal successful! ", menu_type=None)
            await asyncio.sleep(13)
            # Collect Tokens
            await send_tg_message(context, f"ℹ️ Collecting tokens...", menu_type=None)
            collect_params = {
                'tokenId': nft_id,
                'recipient': BOT_WALLET_ADDRESS,
                'amount0Max': 2**128 - 1,
                'amount1Max': 2**128 - 1
            }
            collect_tx_params = {
                'from': BOT_WALLET_ADDRESS, 
                'nonce': await asyncio.to_thread(get_nonce), 
                'chainId': await asyncio.to_thread(lambda: w3.eth.chain_id)
            }
            collect_tx = aerodrome_nft_manager_contract.functions.collect(collect_params).build_transaction(collect_tx_params)
            collect_receipt_obj = await asyncio.to_thread(_send_and_wait_for_transaction, collect_tx, f"Collect Tokens NFT {nft_id} (Startup)")
            if collect_receipt_obj and collect_receipt_obj.status == 1:
                await send_tg_message(context, "✅ Tokens collected successfully.", menu_type=None)
                await asyncio.sleep(13)
            else:
                await send_tg_message(context, f"⚠️ Failed to collect tokens for {nft_id}. Proceeds might be stuck or already claimed. Continuing...", menu_type=None)
        else:
            await send_tg_message(context, f"⚠️ Failed to decrease liquidity for {nft_id}. Manual check advised. Bot HALTED.", menu_type="main")
            bot_state["operations_halted"] = True
            await save_state_async()
            return

        # Burn the now empty NFT
        # Re-check liquidity before burning, just in case.
        final_pos_details_for_burn = await get_lp_position_details(context, nft_id)
        if final_pos_details_for_burn and final_pos_details_for_burn['liquidity'] == 0:
            await send_tg_message(context, f"Burning empty LP `{nft_id}`...", menu_type=None)
            burn_tx_params = {
                'from': BOT_WALLET_ADDRESS, 
                'nonce': await asyncio.to_thread(get_nonce), 
                'chainId': await asyncio.to_thread(lambda: w3.eth.chain_id)
            }
            burn_tx = aerodrome_nft_manager_contract.functions.burn(nft_id).build_transaction(burn_tx_params)
            burn_receipt = await asyncio.to_thread(_send_and_wait_for_transaction, burn_tx, f"Burn LP NFT {nft_id} (Startup)")
            if burn_receipt and burn_receipt.status == 1:
                await send_tg_message(context, f"✅ LP `{nft_id}` burned!", menu_type=None)
                await asyncio.sleep(13)
            else:
                await send_tg_message(context, f"⚠️ Failed to burn {nft_id}, or it was already gone.", menu_type=None)
        else:
            logger.warning(f"NFT {nft_id} still shows liquidity {final_pos_details_for_burn['liquidity'] if final_pos_details_for_burn else 'unknown'} after collect attempt. Skipping burn.")
            await send_tg_message(context, f"⚠️ LP {nft_id} still has liquidity after collect attempt or details unavailable. Burn skipped.", menu_type=None)

    elif position_details and position_details['liquidity'] == 0:
        await send_tg_message(context, f"ℹ️ LP `{nft_id}` already has 0 liquidity. Burning...", menu_type=None)
        burn_tx_params = {
            'from': BOT_WALLET_ADDRESS, 
            'nonce': await asyncio.to_thread(get_nonce), 
            'chainId': await asyncio.to_thread(lambda: w3.eth.chain_id)
        }
        burn_tx = aerodrome_nft_manager_contract.functions.burn(nft_id).build_transaction(burn_tx_params)
        burn_receipt = await asyncio.to_thread(_send_and_wait_for_transaction, burn_tx, f"Burn 0-Liquidity LP NFT {nft_id} (Startup)")
        if burn_receipt and burn_receipt.status == 1:
            await send_tg_message(context, f"✅ LP `{nft_id}` (0 liquidity) burned!", menu_type=None)
            await asyncio.sleep(13)
        else:
            await send_tg_message(context, f"⚠️ Failed to burn 0-liquidity NFT {nft_id}, or it was already gone.", menu_type=None)
    else:
        await send_tg_message(context, f"Could not get position details for NFT {nft_id}, or it doesn't exist. Assuming no liquidity to withdraw from it.", menu_type=None)

    # --- 2. Sell all WBLT for USDC ---
    available_wblt = await get_token_balance(wblt_token_contract, BOT_WALLET_ADDRESS)
    if available_wblt > Decimal("0.1"):
        await send_tg_message(context, f"ℹ️  Selling `{available_wblt:.{wblt_decimals_val}f}` WBLT for USDC...", menu_type=None)
        swap_success, usdc_received = await execute_kyberswap_swap(context, wblt_token_contract, USDC_TOKEN_ADDRESS, available_wblt)
        if swap_success:
            await send_tg_message(context, f"✅ WBLT sold for ~{usdc_received:.2f} USDC.", menu_type=None)
        else:
            await send_tg_message(context, "⚠️ Failed to sell WBLT for USDC. WBLT remains in wallet.", menu_type=None)
    else:
        logger.info("No significant WBLT balance to sell after withdrawal.")

    # --- Finalize State ---
    bot_state["aerodrome_lp_nft_id"] = None
    bot_state["current_lp_principal_wblt_amount"] = Decimal("0")
    bot_state["current_lp_principal_usdc_amount"] = Decimal("0")
    bot_state["initial_setup_pending"] = True
    bot_state["operations_halted"] = True

    final_usdc_balance = await get_token_balance(usdc_token_contract, BOT_WALLET_ADDRESS)
    await send_tg_message(
        context,
        f"✅ Liquidity withdrawal process complete. Bot wallet has `{final_usdc_balance:.2f}` USDC. "
        f"Bot is HALTED. You can start a new LP via the main menu if funds are present.",
        menu_type="main"
    )
    await save_state_async()
    # Optionally, call handle_status_action(context) if you want a full status update here

async def handle_startup_continue_monitoring_action(context: CallbackContext):
    nft_id = bot_state.get("aerodrome_lp_nft_id")
    if not nft_id:
        await send_tg_message(context, "⚠️ No staked NFT ID found in state. Please restart bot.", menu_type="main")
        bot_state["operations_halted"] = True
        return

    await send_tg_message(context, f"✅ Resuming normal monitoring for STAKED LP NFT ID `{nft_id}`.")
    bot_state["operations_halted"] = False
    bot_state["initial_setup_pending"] = False
    await save_state_async()
    await handle_status_action(context)


async def handle_startup_unstake_and_manage_action(context: CallbackContext):
    nft_id = bot_state.get("aerodrome_lp_nft_id")
    if not nft_id:
        await send_tg_message(context, "⚠️ No staked NFT ID found in state. Please restart bot.", menu_type="main")
        bot_state["operations_halted"] = True
        await save_state_async()
        return

    await send_tg_message(context, f"Attempting to unstake, withdraw liquidity, and burn NFT ID `{nft_id}`...", menu_type=None)

    # --- 1. Unstake ---
    await send_tg_message(context, f"ℹ️ Unstaking LP `{nft_id}`...", menu_type=None)
    unstake_tx_params = {'from': BOT_WALLET_ADDRESS}
    unstake_tx = aerodrome_gauge_contract.functions.withdraw(nft_id).build_transaction(unstake_tx_params)
    unstake_receipt = await asyncio.to_thread(_send_and_wait_for_transaction, unstake_tx, f"Unstake LP NFT {nft_id} (Startup Manage)")

    if not (unstake_receipt and unstake_receipt.status == 1):
        await send_tg_message(context, f"⚠️ Failed to unstake LP NFT {nft_id}. Bot remains HALTED. Staked state may be unchanged. Manual check advised.", menu_type="main")
        bot_state["operations_halted"] = True
        await save_state_async()
        await handle_status_action(context)
        return
    
    await send_tg_message(context, f"✅ LP `{nft_id}` UNSTAKED!", menu_type=None)
    await asyncio.sleep(13)

    # --- 2. Withdraw Liquidity & Burn ---
    position_details = await get_lp_position_details(context, nft_id)

    if position_details and position_details['liquidity'] > 0:
        await send_tg_message(context, f"ℹ️ Withdrawing liquidity `({position_details['liquidity']})` from LP `{nft_id}`...", menu_type=None)
        decrease_params = {
            'tokenId': nft_id, 'liquidity': position_details['liquidity'],
            'amount0Min': 0, 'amount1Min': 0, 'deadline': int(time.time()) + 600
        }
        decrease_tx_params = {'from': BOT_WALLET_ADDRESS}
        decrease_tx = aerodrome_nft_manager_contract.functions.decreaseLiquidity(decrease_params).build_transaction(decrease_tx_params)
        decrease_receipt = await asyncio.to_thread(_send_and_wait_for_transaction, decrease_tx, f"Decrease Liquidity NFT {nft_id} (Startup Manage)")

        if decrease_receipt and decrease_receipt.status == 1:
            await send_tg_message(context, "✅ Withdrew liquidity!", menu_type=None)
            await asyncio.sleep(13)
            await send_tg_message(context, f"ℹ️ Collecting tokens...", menu_type=None)
            collect_params = {
                'tokenId': nft_id, 'recipient': BOT_WALLET_ADDRESS,
                'amount0Max': 2**128 - 1, 'amount1Max': 2**128 - 1
            }
            collect_tx_params = {'from': BOT_WALLET_ADDRESS}
            collect_tx = aerodrome_nft_manager_contract.functions.collect(collect_params).build_transaction(collect_tx_params)
            collect_receipt_obj = await asyncio.to_thread(_send_and_wait_for_transaction, collect_tx, f"Collect Tokens NFT {nft_id} (Startup Manage)")
            if collect_receipt_obj and collect_receipt_obj.status == 1:
                await send_tg_message(context, "✅ Tokens collected!", menu_type=None)
                await asyncio.sleep(13)
            else:
                await send_tg_message(context, "⚠️ Failed to collect tokens. Funds might be in wallet or require manual collection.", menu_type=None)
        else:
            await send_tg_message(context, f"⚠️ Failed to decrease liquidity for NFT {nft_id}. Manual check advised.", menu_type="main")

        # Burn
        final_pos_details_for_burn = await get_lp_position_details(context, nft_id)
        if final_pos_details_for_burn and final_pos_details_for_burn['liquidity'] == 0:
            await send_tg_message(context, f"ℹ️ Burning empty LP `{nft_id}`...", menu_type=None)
            burn_tx_params = {'from': BOT_WALLET_ADDRESS}
            burn_tx = aerodrome_nft_manager_contract.functions.burn(nft_id).build_transaction(burn_tx_params)
            burn_receipt = await asyncio.to_thread(_send_and_wait_for_transaction, burn_tx, f"Burn LP NFT {nft_id} (Startup Manage)")
            if burn_receipt and burn_receipt.status == 1:
                await send_tg_message(context, f"✅ NFT {nft_id} burned.", menu_type=None)
            else:
                await send_tg_message(context, f"⚠️ Failed to burn NFT {nft_id}.", menu_type=None)
        elif final_pos_details_for_burn:
             logger.warning(f"NFT {nft_id} still shows liquidity {final_pos_details_for_burn['liquidity']} after collect attempt. Skipping burn.")
             await send_tg_message(context, f"⚠️ NFT {nft_id} still has liquidity. Burn skipped. Manual check needed.", menu_type=None)
        else:
            logger.warning(f"Could not get final position details for NFT {nft_id} before burn. It might already be gone.")


    elif position_details and position_details['liquidity'] == 0:
        await send_tg_message(context, f"ℹ️ LP `{nft_id}` already has 0 liquidity. Burning...", menu_type=None)
        burn_tx_params = {'from': BOT_WALLET_ADDRESS}
        burn_tx = aerodrome_nft_manager_contract.functions.burn(nft_id).build_transaction(burn_tx_params)
        burn_receipt = await asyncio.to_thread(_send_and_wait_for_transaction, burn_tx, f"Burn 0-Liquidity LP NFT {nft_id} (Startup Manage)")
        await asyncio.sleep(13)
        if burn_receipt and burn_receipt.status == 1:
            await send_tg_message(context, f"✅ LP `{nft_id}` (0 liquidity) burned!", menu_type=None)
        else:
            await send_tg_message(context, f"⚠️ Failed to burn 0-liquidity NFT {nft_id}. It might already be gone or an error occurred.", menu_type=None)
            logger.warning(f"Failed to burn 0-liquidity NFT {nft_id} during startup manage action.")
    else:
        await send_tg_message(context, f"Could not find position details for NFT {nft_id}. It might have been burned already or does not belong to the WBLT/USDC pair.", menu_type=None)

    # --- Finalize State ---
    bot_state["aerodrome_lp_nft_id"] = None
    bot_state["current_lp_principal_wblt_amount"] = Decimal("0")
    bot_state["current_lp_principal_usdc_amount"] = Decimal("0")
    bot_state["initial_setup_pending"] = True
    bot_state["operations_halted"] = True

    await send_tg_message(
        context,
        f"✅ LP `{nft_id}` unstaked, liquidity withdrawn, and LP burned (or attempted). "
        f"Bot holds WBLT/USDC. Bot is HALTED. Use 'Force Rebalance' or 'Start Bot Operations' to create a new LP.",
        menu_type="main"
    )
    await save_state_async()
    await handle_status_action(context)

# --- Main Bot Loop ---
async def main_bot_loop(application: Application):
    while True:
        context = CallbackContext(application)
        try:
            if bot_state["operations_halted"] or bot_state.get("is_processing_action", False):
                await asyncio.sleep(15) 
                continue

            logger.info("Main loop iteration started...")
            bot_state["is_processing_action"] = True 

            if not check_connection():
                await send_tg_message(context, "Web3 connection lost! Bot pausing.", menu_type=None)
                await asyncio.sleep(MAIN_LOOP_INTERVAL_SECONDS)
                bot_state["is_processing_action"] = False
                continue
            
            current_time = time.time()

            if bot_state.get("initial_setup_pending", True):
                bot_usdc_balance = await get_token_balance(usdc_token_contract, BOT_WALLET_ADDRESS)
                bot_wblt_balance = await get_token_balance(wblt_token_contract, BOT_WALLET_ADDRESS)
                if bot_usdc_balance > Decimal("1") or bot_wblt_balance > Decimal("1"): 
                    logger.info("Initial setup pending and funds detected. Attempting first rebalance/deposit.")
                    await process_full_rebalance(context, triggered_by="initial_setup")
                else:
                    logger.info("Initial setup pending, but insufficient WBLT/USDC in bot wallet to create LP.")
                
                bot_state["is_processing_action"] = False
                await save_state_async()
                await asyncio.sleep(MAIN_LOOP_INTERVAL_SECONDS)
                continue

            # 1. Check for Rebalancing
            if bot_state["aerodrome_lp_nft_id"]:
                price_wblt_usdc, current_tick = await get_aerodrome_pool_price_and_tick()
                position_details = await get_lp_position_details(context, bot_state["aerodrome_lp_nft_id"])

                if position_details and price_wblt_usdc is not None and current_tick is not None:
                    tick_lower_lp = position_details['tickLower']
                    tick_upper_lp = position_details['tickUpper']
                    actual_tick_span_lp = tick_upper_lp - tick_lower_lp

                    if actual_tick_span_lp <= 0:
                        logger.warning(f"LP NFT {bot_state['aerodrome_lp_nft_id']} has zero or negative tick span. Skipping rebalance check.")
                    else:
                        pool_tick_spacing = await asyncio.to_thread(aerodrome_pool_contract.functions.tickSpacing().call)
                        if pool_tick_spacing == 0:
                            logger.error("Pool tickSpacing is 0! Cannot calculate buffer. Skipping rebalance check.")
                            bot_state["is_processing_action"] = False
                            await save_state_async()
                            await asyncio.sleep(MAIN_LOOP_INTERVAL_SECONDS)
                            continue


                        # 2. Calculate the raw buffer in ticks based on your percentage
                        raw_buffer_in_ticks = actual_tick_span_lp * (REBALANCE_TRIGGER_BUFFER_PERCENTAGE / Decimal(100))

                        # 3. Convert this raw buffer into a number of full tickSpacing units
                        num_tick_spacings_for_buffer = int(Decimal(raw_buffer_in_ticks) / Decimal(pool_tick_spacing))

                        # 4. Ensure the buffer is at least ONE tickSpacing if the percentage calculation
                        # resulted in a non-zero raw buffer but less than one full tickSpacing.
                        if num_tick_spacings_for_buffer == 0 and raw_buffer_in_ticks > 0:
                            num_tick_spacings_for_buffer = 1
                        
                        # 5. Calculate the final buffer_tick_amount as a multiple of tickSpacing
                        buffer_tick_amount_aligned = num_tick_spacings_for_buffer * pool_tick_spacing

                        # Safety: Ensure buffer_tick_amount_aligned doesn't make trigger points cross
                        if (2 * buffer_tick_amount_aligned) >= actual_tick_span_lp and actual_tick_span_lp > 0 :
                            if actual_tick_span_lp > pool_tick_spacing :
                                buffer_tick_amount_aligned = pool_tick_spacing
                                logger.warning(f"Calculated buffer ({REBALANCE_TRIGGER_BUFFER_PERCENTAGE}%) is too large for the LP span ({actual_tick_span_lp} ticks). Falling back to a buffer of one tickSpacing ({pool_tick_spacing} ticks).")
                            else:
                                buffer_tick_amount_aligned = 0
                                logger.warning(f"LP span ({actual_tick_span_lp} ticks) is too narrow for any meaningful buffer. Triggering on range exit.")


                        lower_trigger_tick = tick_lower_lp + buffer_tick_amount_aligned
                        upper_trigger_tick = tick_upper_lp - buffer_tick_amount_aligned

                        logger.info(f"Rebalance Check: LP Range [{tick_lower_lp}, {tick_upper_lp}], Span: {actual_tick_span_lp} ticks.")
                        logger.info(f"Buffer Config: {REBALANCE_TRIGGER_BUFFER_PERCENTAGE}%, TickSpacing: {pool_tick_spacing}.")
                        logger.info(f"Calculated Aligned Buffer: {buffer_tick_amount_aligned} ticks ({num_tick_spacings_for_buffer} tick spacings).")
                        logger.info(f"Trigger Ticks: Lower={lower_trigger_tick}, Upper={upper_trigger_tick}. Current Pool Tick: {current_tick}.")


                        needs_rebalance = False
                        if current_tick < lower_trigger_tick:
                            logger.info(f"Rebalance triggered: Current tick {current_tick} < Lower trigger tick {lower_trigger_tick}")
                            needs_rebalance = True
                        elif current_tick >= upper_trigger_tick:
                            logger.info(f"Rebalance triggered: Current tick {current_tick} >= Upper trigger tick {upper_trigger_tick}")
                            needs_rebalance = True

                        if needs_rebalance:
                            await process_full_rebalance(context, triggered_by="auto_buffer_trigger")
                            bot_state["is_processing_action"] = False
                            await save_state_async()
                            await asyncio.sleep(MAIN_LOOP_INTERVAL_SECONDS)
                            continue
            
            # 2. Check for AERO Claim & Sell
            if bot_state["aerodrome_lp_nft_id"]:
                pending_aero = await get_pending_aero_rewards(context, bot_state["aerodrome_lp_nft_id"])
                time_since_last_claim = current_time - bot_state.get("last_aero_claim_time", 0)

                if pending_aero >= AERO_CLAIM_THRESHOLD_AMOUNT or \
                   (    pending_aero > Decimal("1.0") and time_since_last_claim >= AERO_CLAIM_TIME_THRESHOLD_SECONDS) :
                    logger.info(f"AERO claim triggered. Pending: {pending_aero}, Time since last: {time_since_last_claim/3600:.2f} hrs")
                    await process_claim_sell_aero(context, triggered_by="auto_threshold")
                    bot_state["is_processing_action"] = False
                    await save_state_async()
                    await asyncio.sleep(MAIN_LOOP_INTERVAL_SECONDS)
                    continue
            
            if current_time - bot_state.get("last_telegram_status_update_time", 0) >= PERIODIC_STATUS_UPDATE_INTERVAL_SECONDS:
                logger.info(f"Sending periodic status update to Telegram (interval: {PERIODIC_STATUS_UPDATE_INTERVAL_SECONDS / 3600:.1f} hours).")
                temp_context = CallbackContext(application)
                await handle_status_action(temp_context)

            bot_state["is_processing_action"] = False
            await save_state_async()
            logger.info(f"Main loop iteration finished. Sleeping for {MAIN_LOOP_INTERVAL_SECONDS}s.")
            await asyncio.sleep(MAIN_LOOP_INTERVAL_SECONDS)

        except Exception as e:
            logger.error(f"Critical error in main_bot_loop: {e}", exc_info=True)
            bot_state["is_processing_action"] = False
            await save_state_async()
            try:
                temp_context = CallbackContext(application)
                await send_tg_message(temp_context, f"🚨 CRITICAL ERROR in main loop: {str(e)[:1000]}. Bot continues but check logs.", menu_type="main")
            except Exception as te:
                logger.error(f"Failed to send critical error message to Telegram: {te}")
            await asyncio.sleep(MAIN_LOOP_INTERVAL_SECONDS * 2)


async def approve_nft_for_spending(context: CallbackContext, nft_contract, spender_address, token_id_to_approve: int):
    try:
        # Get human-readable spender name
        spender_name = CONTRACT_NAME_MAP.get(Web3.to_checksum_address(spender_address), spender_address)

        current_approved_address_raw = await asyncio.to_thread(
            nft_contract.functions.getApproved(token_id_to_approve).call
        )
        current_approved_address = Web3.to_checksum_address(current_approved_address_raw) if current_approved_address_raw != '0x0000000000000000000000000000000000000000' else None

        is_operator_approved = await asyncio.to_thread(
            nft_contract.functions.isApprovedForAll(BOT_WALLET_ADDRESS, spender_address).call
        )

        if is_operator_approved:
            logger.info(f"Operator {spender_name} ({spender_address}) is already approved for all NFTs of {BOT_WALLET_ADDRESS}.")
            return True
        
        if current_approved_address != Web3.to_checksum_address(spender_address):
            await send_tg_message(
                context, 
                f"ℹ️ Approving NFT ID `{token_id_to_approve}` for spender: **{spender_name}**...", 
                menu_type=None
            )
            
            approve_tx_params = {'from': BOT_WALLET_ADDRESS}
            approve_tx = nft_contract.functions.approve(spender_address, token_id_to_approve).build_transaction(approve_tx_params)
            
            receipt = await asyncio.to_thread(
                _send_and_wait_for_transaction, 
                approve_tx, 
                f"Approve NFT {token_id_to_approve} for {spender_name}"
            )

            if receipt is not None and receipt.status == 1:
                await send_tg_message(context, f"✅ Approved LP `{token_id_to_approve}` for **{spender_name}**!", menu_type=None)
                return True
            else:
                await send_tg_message(context, f"❌ Failed to approve NFT ID `{token_id_to_approve}` for **{spender_name}**.", menu_type=None)
                return False
        else:
            logger.info(f"NFT ID {token_id_to_approve} already approved for {spender_name} ({spender_address}).")
            return True
    except Exception as e:
        spender_name_for_error = CONTRACT_NAME_MAP.get(Web3.to_checksum_address(spender_address), spender_address)
        logger.error(f"Error in approve_nft_for_spending for NFT {token_id_to_approve} to {spender_name_for_error}: {e}", exc_info=True)
        await send_tg_message(context, f"Error approving NFT {token_id_to_approve} for {spender_name_for_error}: {str(e)[:100]}", menu_type=None)
        return False

# --- Application ---
def main():
    if not all([TELEGRAM_BOT_TOKEN, str(TELEGRAM_ADMIN_USER_ID) != "0", ALCHEMY_RPC_URL, BOT_WALLET_ADDRESS, BOT_PRIVATE_KEY, USER_PROFIT_WITHDRAWAL_ADDRESS]):
        logger.critical("CRITICAL: One or more essential configuration variables are not set. Exiting.")
        print("CRITICAL: One or more essential configuration variables are not set in the script. Please edit the script and fill them.")
        return

    if not all([ERC20_ABI, AERODROME_NFT_MANAGER_ABI, AERODROME_GAUGE_ABI, AERODROME_POOL_ABI]):
        logger.critical("CRITICAL: Could not load one or more contract ABIs. Ensure ABI files are in the 'abis' directory. Exiting.")
        print("CRITICAL: Could not load one or more contract ABIs. Ensure ABI JSON files are in the 'abis' directory and named correctly.")
        return
        
    load_state_sync()

    # ALWAYS START IN A HALTED STATE on script (re)start, requiring manual initiation
    bot_state["operations_halted"] = True
    bot_state["is_processing_action"] = False 
    logger.info("Bot forced into HALTED state on script startup. Initializing...")

    # Set initial NFT ID from config if provided and not already in state
    if INITIAL_LP_NFT_ID_CONFIG is not None and bot_state.get("aerodrome_lp_nft_id") is None:
        logger.info(f"Using initial LP NFT ID from config: {INITIAL_LP_NFT_ID_CONFIG} (will be verified by discovery)")
        bot_state["aerodrome_lp_nft_id"] = INITIAL_LP_NFT_ID_CONFIG # Tentatively set

    # --- Application and Event Loop ---
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    loop = asyncio.get_event_loop()

    # --- LP State Discovery and User Interaction on Startup ---
    async def startup_sequence(app: Application):
        nonlocal loop
        startup_context = CallbackContext(app)
        await send_tg_message(startup_context, "🤖 Bot instance starting up. Discovering LP state and balances...", menu_type=None)

        discovered_nft_id, discovered_status = await discover_lp_state(startup_context, BOT_WALLET_ADDRESS)

        if discovered_nft_id is not None:
            logger.info(f"Discovered LP NFT ID: {discovered_nft_id}, Status: {discovered_status}")
            bot_state["aerodrome_lp_nft_id"] = discovered_nft_id
            bot_state["initial_setup_pending"] = False

            if discovered_status == "unstaked_in_wallet":
                await send_tg_message(
                    startup_context,
                    f"ℹ️ Discovered UNSTAKED WBLT/USDC LP `{discovered_nft_id}` in bot wallet with active liquidity. What would you like to do?",
                    menu_type="startup_unstaked_lp"
                )
            elif discovered_status == "staked":
                await send_tg_message(
                    startup_context,
                    f"ℹ️ Discovered STAKED WBLT/USDC LP `{discovered_nft_id}` in the gauge with active liquidity. What would you like to do?",
                    menu_type="startup_staked_lp"
                )
            else: 
                logger.error(f"Discovered NFT {discovered_nft_id} but with unknown status: {discovered_status}. Halting.")
                await send_tg_message(startup_context, f"⚠️ Error: Discovered NFT {discovered_nft_id} with unknown status. Manual check needed. Bot HALTED.", menu_type="main")
                bot_state["operations_halted"] = True
        else:
            logger.info("No active WBLT/USDC LP NFT discovered on-chain for the bot.")

            if bot_state.get("aerodrome_lp_nft_id") is not None:
                logger.warning(
                    f"Saved state had LP ID {bot_state.get('aerodrome_lp_nft_id')} but no active LP found on-chain. "
                    "Resetting saved LP ID."
                )
            bot_state["aerodrome_lp_nft_id"] = None
            bot_state["initial_setup_pending"] = True

            # --- Query balances if no LP is found ---
            await send_tg_message(startup_context, "No active LP found. Checking wallet balances for WBLT & USDC...", menu_type=None)
            try:
                
                if not wblt_token_contract or not usdc_token_contract:
                    logger.error("Token contracts not initialized. Cannot fetch balances for startup message.")
                    await send_tg_message(startup_context, "⚠️ Error: Token contracts not ready. Cannot display balances.", menu_type="main")
                else:
                    wallet_wblt_balance = await get_token_balance(wblt_token_contract, BOT_WALLET_ADDRESS)
                    wallet_usdc_balance = await get_token_balance(usdc_token_contract, BOT_WALLET_ADDRESS)
                    
                    wblt_decimals_val = await asyncio.to_thread(wblt_token_contract.functions.decimals().call)
                    usdc_decimals_val = await asyncio.to_thread(usdc_token_contract.functions.decimals().call)

                    logger.info(f"Startup balances: WBLT={wallet_wblt_balance}, USDC={wallet_usdc_balance}")

                    balance_message_part = (
                        f"Current wallet balances:\n"
                        f"  🌯 WBLT: `{wallet_wblt_balance:.{wblt_decimals_val}f}`\n"
                        f"  💰 USDC: `{wallet_usdc_balance:.{usdc_decimals_val}f}`\n\n"
                    )

                    # Check if there are sufficient funds to potentially start an LP
                    has_significant_wblt = wallet_wblt_balance > MIN_SWAP_THRESHOLD_WBLT
                    has_significant_usdc = wallet_usdc_balance > MIN_SWAP_THRESHOLD_USDC

                    if has_significant_wblt or has_significant_usdc:
                        startup_info_message = (
                            f"{balance_message_part}"
                            f"ℹ️ Initial setup needed. Funds detected. "
                            f"You can 'Start Bot Operations' to create a new LP with these funds, "
                            f"or manage principal/set an existing LP ID via the menu."
                        )
                    else:
                        startup_info_message = (
                            f"{balance_message_part}"
                            f"ℹ️ Initial setup needed. Insufficient WBLT/USDC detected in the bot wallet (`{BOT_WALLET_ADDRESS}`). "
                            f"Please send funds or set an existing LP ID via Manage Principal before starting operations."
                        )
                    await send_tg_message(startup_context, startup_info_message, menu_type="main")

            except Exception as e_bal:
                logger.error(f"Error fetching balances during startup: {e_bal}", exc_info=True)
                await send_tg_message(startup_context, f"ℹ️ Initial setup needed: No LP found. Send WBLT/USDC to `{BOT_WALLET_ADDRESS}` or set an existing LP ID via Manage Principal before starting operations.", menu_type="main")
            
            # Always show the main menu after these initial messages when no LP is found
            # await send_tg_message(startup_context, "Operations are PAUSED. Use menu to start/resume.", menu_type="main")

        save_state_sync() 

        loop.create_task(main_bot_loop(app))
        logger.info("Main bot loop scheduled. Startup sequence complete or awaiting user input.")

    # --- Handlers ---
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("menu", menu_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CallbackQueryHandler(button_handler))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_message_handler))

    # --- Run Startup Sequence ---
    try:
        loop.run_until_complete(startup_sequence(application))
        logger.info("Starting Telegram bot polling...")
        application.run_polling()

    except KeyboardInterrupt:
        logger.info("Bot shutdown requested via KeyboardInterrupt.")
    except Exception as e:
        logger.critical(f"Application failed: {e}", exc_info=True)
    finally:
        logger.info("Bot shutting down. Saving final state.")
        save_state_sync()
        tasks = [t for t in asyncio.all_tasks(loop) if t is not asyncio.current_task(loop)]
        if tasks:
            logger.info(f"Cancelling {len(tasks)} outstanding tasks...")
            for task in tasks:
                task.cancel()
            loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
        loop.close()
        logger.info("Asyncio loop closed.")

if __name__ == '__main__':
    if not BOT_PRIVATE_KEY or BOT_PRIVATE_KEY == "YOUR_BOT_PRIVATE_KEY_NO_0x_PREFIX":
        print("FATAL: BOT_PRIVATE_KEY is not set in the script. Exiting.")
        logger.critical("FATAL: BOT_PRIVATE_KEY is not set. Exiting.")
    elif not TELEGRAM_BOT_TOKEN or TELEGRAM_BOT_TOKEN == "YOUR_TELEGRAM_BOT_TOKEN":
        print("FATAL: TELEGRAM_BOT_TOKEN is not set in the script. Exiting.")
        logger.critical("FATAL: TELEGRAM_BOT_TOKEN is not set. Exiting.")
    elif not ALCHEMY_RPC_URL or "YOUR_ALCHEMY_API_KEY" in ALCHEMY_RPC_URL:
        print("FATAL: ALCHEMY_RPC_URL is not set correctly in the script. Exiting.")
        logger.critical("FATAL: ALCHEMY_RPC_URL is not set correctly. Exiting.")
    elif TELEGRAM_ADMIN_USER_ID == 0:
        print("FATAL: TELEGRAM_ADMIN_USER_ID is not set in the script. Exiting.")
        logger.critical("FATAL: TELEGRAM_ADMIN_USER_ID is not set. Exiting.")
    else:
        main()
