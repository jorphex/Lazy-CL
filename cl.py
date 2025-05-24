#
# Change sleep times to whatever works for you. 13s seems to be what's stable for Base, which is annoying given 2s block times
# Change settings and contracts as needed

import os
from dotenv import load_dotenv
import asyncio
import json
import time
import logging
from typing import Callable, Awaitable, Tuple, Optional, Union
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

# --- Logging ---
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - %(message)s',
    level=logging.INFO,
    handlers=[
        logging.FileHandler("aerodrome_bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
# use .env here or die!
load_dotenv()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_ADMIN_USER_ID_STR = os.getenv("TELEGRAM_ADMIN_USER_ID")
TELEGRAM_ADMIN_USER_ID = int(TELEGRAM_ADMIN_USER_ID_STR) if TELEGRAM_ADMIN_USER_ID_STR else 0
ALCHEMY_RPC_URL = os.getenv("ALCHEMY_RPC_URL")
BOT_WALLET_ADDRESS_STR = os.getenv("BOT_WALLET_ADDRESS")
BOT_WALLET_ADDRESS = Web3.to_checksum_address(BOT_WALLET_ADDRESS_STR) if BOT_WALLET_ADDRESS_STR else None
BOT_PRIVATE_KEY = os.getenv("BOT_PRIVATE_KEY")
USER_PROFIT_WITHDRAWAL_ADDRESS_STR = os.getenv("USER_PROFIT_WITHDRAWAL_ADDRESS")
USER_PROFIT_WITHDRAWAL_ADDRESS = Web3.to_checksum_address(USER_PROFIT_WITHDRAWAL_ADDRESS_STR) if USER_PROFIT_WITHDRAWAL_ADDRESS_STR else None

# --- Validate Essential Configuration ---
ESSENTIAL_CONFIG_VARS = {
    "TELEGRAM_BOT_TOKEN": TELEGRAM_BOT_TOKEN,
    "TELEGRAM_ADMIN_USER_ID": TELEGRAM_ADMIN_USER_ID,
    "ALCHEMY_RPC_URL": ALCHEMY_RPC_URL,
    "BOT_WALLET_ADDRESS": BOT_WALLET_ADDRESS,
    "BOT_PRIVATE_KEY": BOT_PRIVATE_KEY,
    "USER_PROFIT_WITHDRAWAL_ADDRESS": USER_PROFIT_WITHDRAWAL_ADDRESS
}

missing_configs = [key for key, value in ESSENTIAL_CONFIG_VARS.items() if not value or (isinstance(value, int) and value == 0)]

if missing_configs:
    error_message = f"CRITICAL: The following essential configuration variables are missing or invalid in .env or script: {', '.join(missing_configs)}. Exiting."
    logger.critical(error_message)
    print(error_message)
    exit()

# Aerodrome & Token Addresses
AERO_TOKEN_ADDRESS = Web3.to_checksum_address("0x940181a94A35A4569E4529A3CDfB74e38FD98631")
WBLT_TOKEN_ADDRESS = Web3.to_checksum_address("0x4E74D4Db6c0726ccded4656d0BCE448876BB4C7A")
USDC_TOKEN_ADDRESS = Web3.to_checksum_address("0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913")

AERODROME_CL_POOL_ADDRESS = Web3.to_checksum_address("0x7cE345561E1690445eEfA0dB04F59d64b65598A8")
AERODROME_SLIPSTREAM_NFT_MANAGER_ADDRESS = Web3.to_checksum_address("0x827922686190790b37229fd06084350E74485b72")
AERODROME_CL_GAUGE_ADDRESS = Web3.to_checksum_address("0xf00d67799Cd4E1A77D14671149b599A96DcD38eC")

# KyberSwap
KYBERSWAP_ROUTER_ADDRESS = Web3.to_checksum_address("0x6131B5fae19EA4f9D964eAc0408E4408b66337b5")
KYBERSWAP_API_BASE_URL = "https://aggregator-api.kyberswap.com/base/api/v1"
KYBERSWAP_X_CLIENT_ID = "AerodromeKyberBotV1" # Any name

# Dexscreener
DEXSCREENER_API_BASE_URL = "https://api.dexscreener.com/latest/dex/pairs/base/"
DEXSCREENER_PAIR_WBLT_USDC = "0x7cE345561E1690445eEfA0dB04F59d64b65598A8"
DEXSCREENER_PAIR_AERO_USDC = "0x6cdcb1c4a4d1c3c6d054b27ac5b77e89eafb971d"
DEXSCREENER_PAIR_USDC_STABLE = "0x98c7a2338336d2d354663246f64676009c7bda97"

# Bot Settings
SLIPPAGE_BPS = 100  # 1%
TARGET_RANGE_WIDTH_PERCENTAGE = Decimal("2.0")  # 2% total width for the LP position
REBALANCE_TRIGGER_BUFFER_PERCENTAGE = Decimal("5.0") # 5% buffer from each edge of the active range
AERO_CLAIM_THRESHOLD_AMOUNT = Decimal("50") # Claim AERO if pending > 50 AERO
AERO_CLAIM_TIME_THRESHOLD_SECONDS = 12 * 60 * 60 # Claim AERO every 12 hours
MAIN_LOOP_INTERVAL_SECONDS = 15 * 60 # Check conditions every 15 minutes
PERIODIC_STATUS_UPDATE_INTERVAL_SECONDS = 6 * 60 * 60 # every 6 hours
TRANSACTION_TIMEOUT_SECONDS = 360 # 6 minutes for transaction receipt
INITIAL_LP_NFT_ID_CONFIG = None
MIN_SWAP_THRESHOLD_WBLT = Decimal("0.1") # Don't swap less than 1.0 WBLT
MIN_SWAP_THRESHOLD_USDC = Decimal("1.0")  # Don't swap less than 1.0 USDC

OperationCoro = Callable[[], Awaitable[Tuple[bool, Optional[Decimal]]]]

# Gas
MAX_PRIORITY_FEE_PER_GAS_GWEI = Decimal("0.002")

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
    default_bot_state_keys = list(bot_state.keys())

    try:
        with open(STATE_FILE, 'r') as f:
            loaded_s = json.load(f)
            for key, loaded_value in loaded_s.items():
                if key in default_bot_state_keys:
                    decimal_keys = [
                        "target_range_width_percentage", 
                        "rebalance_buffer_percentage", 
                        "aero_claim_threshold_amount",
                        "accumulated_profit_usdc"
                    ]
                    if key in decimal_keys:
                        try:
                            bot_state[key] = Decimal(str(loaded_value))
                        except Exception as e_dec:
                            logger.error(f"Error converting loaded state value for '{key}' to Decimal: {loaded_value}. Error: {e_dec}. Using default or 0.")
                            if key == "accumulated_profit_usdc":
                                bot_state[key] = Decimal("0") 
                    else:
                        bot_state[key] = loaded_value
        logger.info(f"Bot state loaded from {STATE_FILE}")
    except FileNotFoundError:
        logger.warning(f"State file {STATE_FILE} not found. Using default state values.")
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from {STATE_FILE}. Using default state values.")
    except Exception as e:
        logger.error(f"Error loading state: {e}. Using default state values.")

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
    elif menu_type == "withdraw_funds_options":
        keyboard = await get_withdraw_options_menu(context)
    elif menu_type == "withdraw_wallet_usdc":
        keyboard = await get_withdraw_wallet_usdc_menu(context)
    elif menu_type == "profit_withdrawal":
        keyboard = await get_profit_withdrawal_keyboard()
    elif menu_type == "emergency_exit_confirm":
        keyboard = await get_emergency_exit_confirmation_keyboard()
    elif menu_type == "manage_lp_wallet":
        keyboard = await get_manage_lp_wallet_keyboard()
    elif menu_type == "startup_unstaked_lp":
        keyboard = await get_startup_unstaked_lp_menu()
    elif menu_type == "startup_staked_lp":
        keyboard = await get_startup_staked_lp_menu()

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

async def get_dexscreener_price_usd(pair_address: str) -> Optional[Decimal]:
    """
    Fetches the USD price of the base token in a given pair from DexScreener.
    Returns priceUsd as Decimal, or None if an error occurs or price not found.
    """
    url = f"{DEXSCREENER_API_BASE_URL}{pair_address}"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url)
        response.raise_for_status()
        data = response.json()

        if data and data.get("pairs") and isinstance(data["pairs"], list) and len(data["pairs"]) > 0:
            
            pair_data = data["pairs"][0]
            if 'priceUsd' in pair_data and pair_data['priceUsd'] is not None:
                price_str = pair_data['priceUsd']
                logger.debug(f"DexScreener priceUsd for pair {pair_address}: {price_str}")
                return Decimal(price_str)
            else:
                logger.warning(f"DexScreener: 'priceUsd' not found or is null for pair {pair_address}. Data: {pair_data}")
                return None
        else:
            logger.warning(f"DexScreener: No pairs data found for {pair_address}. Response: {data}")
            return None
    except httpx.HTTPStatusError as e:
        logger.error(f"DexScreener HTTP error for {pair_address}: {e.response.status_code} - {e.response.text}")
        return None
    except (httpx.RequestError, json.JSONDecodeError) as e:
        logger.error(f"DexScreener request/JSON error for {pair_address}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error in get_dexscreener_price_usd for {pair_address}: {e}", exc_info=True)
        return None
    
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

    # Calculate SqrtPrices
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

# --- LP Management ---
async def _claim_rewards_for_staked_nft(context: CallbackContext, nft_id_to_claim: int) -> bool:
    """Helper to claim AERO rewards for a specific staked NFT."""
    if not nft_id_to_claim: return False
    
    pending_aero_before_claim = await get_pending_aero_rewards(context, nft_id_to_claim)
    if pending_aero_before_claim < Decimal("1.0"):
        await send_tg_message(context, f"ℹ️ No significant AERO rewards ({pending_aero_before_claim:.6f}) to claim for LP `{nft_id_to_claim}`.", menu_type=None)
        bot_state["last_aero_claim_time"] = time.time()
        return True

    await send_tg_message(context, f"ℹ️ Attempting to claim `{pending_aero_before_claim:.6f}` AERO for LP `{nft_id_to_claim}`...", menu_type=None)
    claim_tx_params = {'from': BOT_WALLET_ADDRESS}
    claim_tx_obj = aerodrome_gauge_contract.functions.getReward(nft_id_to_claim) 
    
    try:
        claim_tx = claim_tx_obj.build_transaction(claim_tx_params)
    except Exception as e_build:
        logger.error(f"Error building claim transaction for NFT {nft_id_to_claim}: {e_build}", exc_info=True)
        await send_tg_message(context, f"❌ Error building claim transaction for LP {nft_id_to_claim}.", menu_type=None)
        return False

    claim_receipt = await asyncio.to_thread(_send_and_wait_for_transaction, claim_tx, f"Claim AERO for LP {nft_id_to_claim}")

    if claim_receipt and claim_receipt.status == 1:
        await send_tg_message(context, "✅ AERO rewards claimed successfully from gauge.", menu_type=None)
        bot_state["last_aero_claim_time"] = time.time()
        await asyncio.sleep(13)
        return True
    else:
        await send_tg_message(context, "❌ AERO claim transaction failed or not confirmed.", menu_type=None)
        return False
    
async def _execute_mint_lp_operation(
    context: CallbackContext,
    mint_params_as_tuple: tuple
) -> Tuple[bool, Union[dict, str, None]]:
    """
    Performs the static call and then the actual mint transaction.
    Meant to be wrapped by attempt_operation_with_retries.
    """
    logger.info("Attempting one cycle of LP mint operation (static call + send)...")
    
    if aerodrome_nft_manager_contract is None:
        logger.critical("CRITICAL during _execute_mint_lp_operation: aerodrome_nft_manager_contract is None!")
        return False, "NFT Manager contract not loaded"

    prepared_mint_function_call = aerodrome_nft_manager_contract.functions.mint(mint_params_as_tuple)
    encoded_mint_data = prepared_mint_function_call._encode_transaction_data()
    tx_for_call_and_send = {
        'from': BOT_WALLET_ADDRESS,
        'to': AERODROME_SLIPSTREAM_NFT_MANAGER_ADDRESS,
        'data': encoded_mint_data,
    }

    # 1. Static Call
    logger.warning("Attempting static call (w3.eth.call) for mint...")
    try:
        await asyncio.to_thread(w3.eth.call, tx_for_call_and_send, 'latest')
        logger.info("Static call for mint SUCCEEDED (or did not revert).")
    except ContractLogicError as cle_static:
        logger.error(f"Static call for mint FAILED: {getattr(cle_static, 'message', str(cle_static))} Data: {getattr(cle_static, 'data', 'N/A')}")
        raise cle_static
    except Exception as e_static:
        logger.error(f"Unexpected error during static call for mint: {e_static}", exc_info=True)
        raise e_static

    # 2. Send Transaction
    mint_tx_params = tx_for_call_and_send.copy()

    receipt = await asyncio.to_thread(
         _send_and_wait_for_transaction,
         mint_tx_params,
         "Mint New Aerodrome LP (Optimal Amts)"
    )

    if receipt and receipt.status == 1:
        return True, receipt
    elif receipt and receipt.status == 0:
        logger.error(f"Mint transaction confirmed but FAILED on-chain (status 0). Receipt: {receipt}")
        raise ContractLogicError(f"On-chain revert (status 0) for mint.", data=receipt)
    else:
        logger.error("Mint transaction failed after all internal retries in _send_and_wait_for_transaction (network/timeout).")
        raise Exception("Mint transaction failed due to network/timeout issues after internal retries.")    

async def _execute_stake_lp_nft(context: CallbackContext, nft_id_to_stake: int) -> bool:
    """
    Approves and stakes the given LP NFT ID.
    Returns True on successful staking, False otherwise.
    """
    if not nft_id_to_stake:
        logger.error("_execute_stake_lp_nft: No NFT ID provided.")
        return False

    await send_tg_message(context, f"ℹ️ Preparing to stake LP `{nft_id_to_stake}`...", menu_type=None)

    # 1. Approve NFT for Gauge
    approved_nft_for_gauge = await approve_nft_for_spending(
        context,
        aerodrome_nft_manager_contract,
        AERODROME_CL_GAUGE_ADDRESS,
        nft_id_to_stake
    )

    if not approved_nft_for_gauge:
        logger.error(f"Failed to approve LP {nft_id_to_stake} for staking. Staking aborted.")
        return False
    
    await send_tg_message(context, f"ℹ️ Staking LP...", menu_type=None)

    # 2. Stake NFT
    stake_tx_params = {'from': BOT_WALLET_ADDRESS}
    stake_tx_obj = aerodrome_gauge_contract.functions.deposit(nft_id_to_stake)
    
    try:
        stake_tx = stake_tx_obj.build_transaction(stake_tx_params)
    except Exception as e_build:
        logger.error(f"Error building stake transaction for LP {nft_id_to_stake}: {e_build}", exc_info=True)
        await send_tg_message(context, f"❌ Error building stake transaction for LP {nft_id_to_stake}.", menu_type=None)
        return False

    stake_receipt = await asyncio.to_thread(
        _send_and_wait_for_transaction, 
        stake_tx, 
        f"Stake LP {nft_id_to_stake}"
    )

    if stake_receipt and stake_receipt.status == 1:
        await send_tg_message(context, f"✅ LP {nft_id_to_stake} successfully STAKED!", menu_type=None)
        bot_state["last_aero_claim_time"] = time.time()
        return True
    else:
        await send_tg_message(context, f"⚠️ Failed to stake LP {nft_id_to_stake}. Receipt: {stake_receipt}", menu_type=None)
        return False

async def _sell_all_available_aero_in_wallet(context: CallbackContext) -> tuple[bool, Decimal]:
    """Checks wallet AERO balance and sells if above threshold. Returns (success, usdc_received)."""
    try:
        aero_balance = await get_token_balance(aero_token_contract, BOT_WALLET_ADDRESS)
        logger.info(f"Current wallet AERO balance: {aero_balance}")

        MIN_AERO_TO_SELL_FOR_SWAP = Decimal("1.0")

        if aero_balance < MIN_AERO_TO_SELL_FOR_SWAP:
            logger.info(f"AERO balance {aero_balance} is below sell threshold {MIN_AERO_TO_SELL_FOR_SWAP}. No sale.")
            return True, Decimal(0)

        await send_tg_message(context, f"ℹ️ Attempting to sell `{aero_balance:.6f}` AERO from wallet for USDC...", menu_type=None)
        swap_aero_op = functools.partial(execute_kyberswap_swap, context, aero_token_contract, USDC_TOKEN_ADDRESS, aero_balance)
        swap_success, usdc_received = await attempt_operation_with_retries(
            swap_aero_op, "Sell All Wallet AERO via KyberSwap", context
        )

        if swap_success:
            usdc_amount = usdc_received if usdc_received else Decimal("0")
            return True, usdc_amount
        else:
            await send_tg_message(context, "⚠️ Failed to sell AERO from wallet after retries.", menu_type=None)
            return False, Decimal(0)
    except Exception as e:
        logger.error(f"Error in _sell_all_available_aero_in_wallet: {e}", exc_info=True)
        await send_tg_message(context, "❌ Critical error selling AERO from wallet.", menu_type=None)
        return False, Decimal(0)

async def _unstake_lp_from_gauge(context: CallbackContext, nft_id: int) -> bool:
    """Helper to unstake a given LP NFT ID from the gauge."""
    if not nft_id:
        logger.warning("_unstake_lp_from_gauge: No NFT ID provided.")
        return False
    
    logger.info(f"Attempting to unstake LP {nft_id} from gauge...")
    await send_tg_message(context, f"ℹ️ Unstaking LP `{nft_id}` from gauge...", menu_type=None)

    try:
        is_staked_check = await asyncio.to_thread(
            aerodrome_gauge_contract.functions.stakedContains(BOT_WALLET_ADDRESS, nft_id).call
        )
        if not is_staked_check:
            logger.info(f"LP {nft_id} is not currently staked in gauge. Skipping unstake transaction.")
            await send_tg_message(context, f"ℹ️ LP `{nft_id}` already in wallet (not staked).", menu_type=None)
            return True
    except Exception as e_staked_check:
        logger.warning(f"Could not check if LP {nft_id} is staked before unstaking: {e_staked_check}. Proceeding with unstake attempt.")

    unstake_tx_params = {'from': BOT_WALLET_ADDRESS}
    try:
        unstake_tx = aerodrome_gauge_contract.functions.withdraw(nft_id).build_transaction(unstake_tx_params)
        unstake_receipt = await asyncio.to_thread(_send_and_wait_for_transaction, unstake_tx, f"Unstake LP LP {nft_id}")
        
        if unstake_receipt and unstake_receipt.status == 1:
            await send_tg_message(context, f"✅ LP `{nft_id}` unstaked successfully!", menu_type=None)
            bot_state["last_aero_claim_time"] = time.time()
            await asyncio.sleep(13)
            return True
        else:
            await send_tg_message(context, f"⚠️ Failed to unstake LP `{nft_id}` (tx status 0 or no receipt).", menu_type=None)
            return False
    except ContractLogicError as cle:
        if "NA" in str(cle.message):
            logger.warning(f"Unstake attempt for LP {nft_id} reverted with 'NA'. Likely already unstaked or not associated with this gauge/wallet.")
            await send_tg_message(context, f"ℹ️ Unstake for LP `{nft_id}` reverted (likely already unstaked).", menu_type=None)
            return True
        else:
            logger.error(f"ContractLogicError during unstake for LP {nft_id}: {cle}", exc_info=True)
            await send_tg_message(context, f"❌ Contract error during unstake of LP `{nft_id}`: {str(cle.message)[:50]}", menu_type=None)
            return False
    except Exception as e:
        logger.error(f"Unexpected error during unstake for LP {nft_id}: {e}", exc_info=True)
        await send_tg_message(context, f"❌ Unexpected error during unstake of LP `{nft_id}`.", menu_type=None)
        return False


async def _withdraw_collect_from_lp(context: CallbackContext, nft_id: int) -> bool:
    """Helper to decrease all liquidity and collect tokens from an LP LP."""
    if not nft_id:
        logger.warning("_withdraw_collect_from_lp: No LP ID provided.")
        return False

    position_details = await get_lp_position_details(context, nft_id)
    if not position_details:
        await send_tg_message(context, f"⚠️ Cannot get position details for LP `{nft_id}`. Cannot withdraw/collect.", menu_type=None)
        return False

    if position_details['liquidity'] == 0:
        await send_tg_message(context, f"ℹ️ LP `{nft_id}` already has 0 liquidity. No withdrawal needed.", menu_type=None)
        return True

    await send_tg_message(context, f"ℹ️ Withdrawing liquidity ({position_details['liquidity']}) from LP `{nft_id}`...", menu_type=None)
    
    decrease_params = {
        'tokenId': nft_id, 'liquidity': position_details['liquidity'],
        'amount0Min': 0, 'amount1Min': 0, 'deadline': int(time.time()) + 600 
    }
    decrease_tx_params = {'from': BOT_WALLET_ADDRESS}
    try:
        decrease_tx = aerodrome_nft_manager_contract.functions.decreaseLiquidity(decrease_params).build_transaction(decrease_tx_params)
        decrease_receipt = await asyncio.to_thread(_send_and_wait_for_transaction, decrease_tx, f"Decrease Liquidity LP {nft_id}")
    except Exception as e_dec:
        logger.error(f"Error during decreaseLiquidity for LP {nft_id}: {e_dec}", exc_info=True)
        await send_tg_message(context, f"❌ Error decreasing liquidity for LP `{nft_id}`.", menu_type=None)
        return False

    if not (decrease_receipt and decrease_receipt.status == 1):
        await send_tg_message(context, f"⚠️ Failed to decrease liquidity for LP `{nft_id}`.", menu_type=None)
        return False

    await send_tg_message(context, f"✅ Withdraw successful for LP `{nft_id}`!", menu_type=None)
    await asyncio.sleep(13) 
    await send_tg_message(context, f"ℹ️ Collecting tokens...", menu_type=None)

    collect_params = {
        'tokenId': nft_id, 'recipient': BOT_WALLET_ADDRESS,
        'amount0Max': 2**128 - 1, 'amount1Max': 2**128 - 1
    }
    collect_tx_params = {'from': BOT_WALLET_ADDRESS}
    try:
        collect_tx = aerodrome_nft_manager_contract.functions.collect(collect_params).build_transaction(collect_tx_params)
        collect_receipt_obj = await asyncio.to_thread(_send_and_wait_for_transaction, collect_tx, f"Collect Tokens NFT {nft_id}")
    except Exception as e_col:
        logger.error(f"Error during collect for LP {nft_id}: {e_col}", exc_info=True)
        await send_tg_message(context, f"❌ Error collecting tokens for LP `{nft_id}`.", menu_type=None)
        return False

    if collect_receipt_obj and collect_receipt_obj.status == 1:
        await send_tg_message(context, f"✅ Tokens collected successfully for LP `{nft_id}`!", menu_type=None)
        await asyncio.sleep(13)
        return True
    else:
        await send_tg_message(context, f"⚠️ Failed to collect tokens for LP `{nft_id}`. Funds may require manual collection or are already in wallet.", menu_type=None)
        return False


async def _burn_lp_nft(context: CallbackContext, nft_id: int) -> bool:
    """Helper to burn an LP NFT, typically after liquidity is zero."""
    if not nft_id:
        logger.warning("_burn_lp_nft: No NFT ID provided.")
        return False

    final_pos_details = await get_lp_position_details(context, nft_id)
    if final_pos_details and final_pos_details['liquidity'] > 0:
        logger.warning(f"Attempting to burn LP {nft_id} but it still has liquidity {final_pos_details['liquidity']}. This is unusual.")
        await send_tg_message(context, f"⚠️ LP `{nft_id}` still has liquidity. Burn might fail or is not advised.", menu_type=None)
    elif not final_pos_details:
        logger.info(f"Could not get position details for LP {nft_id} before burning. It might already be gone.")

    await send_tg_message(context, f"ℹ️ Attempting to burn LP `{nft_id}`...", menu_type=None)
    burn_tx_params = {'from': BOT_WALLET_ADDRESS}
    try:
        burn_tx = aerodrome_nft_manager_contract.functions.burn(nft_id).build_transaction(burn_tx_params)
        burn_receipt = await asyncio.to_thread(_send_and_wait_for_transaction, burn_tx, f"Burn LP {nft_id}")
        
        if burn_receipt and burn_receipt.status == 1:
            await send_tg_message(context, f"✅ LP `{nft_id}` burned successfully!", menu_type=None)
            await asyncio.sleep(13)
            return True
        else:
            logger.warning(f"Burn transaction for LP {nft_id} failed or was not confirmed. It might have been burned already.")
            await send_tg_message(context, f"⚠️ Failed to confirm burn for LP `{nft_id}` (may already be gone).", menu_type=None)
            return False
    except ContractLogicError as cle:
        if "nonexistent token" in str(cle.message).lower() or "invalid token id" in str(cle.message).lower():
            logger.info(f"LP {nft_id} likely already burned (reverted with: {str(cle.message)}). Considering burn successful.")
            await send_tg_message(context, f"✅ LP `{nft_id}` appears to be already burned.", menu_type=None)
            return True
        else:
            logger.error(f"ContractLogicError during burn for LP {nft_id}: {cle}", exc_info=True)
            await send_tg_message(context, f"❌ Contract error during burn of LP `{nft_id}`: {str(cle.message)[:50]}", menu_type=None)
            return False
    except Exception as e:
        logger.error(f"Unexpected error during burn for LP {nft_id}: {e}", exc_info=True)
        await send_tg_message(context, f"❌ Unexpected error during burn of LP `{nft_id}`.", menu_type=None)
        return False

async def _fully_dismantle_lp(context: CallbackContext, nft_id_to_dismantle: int, initially_staked: bool) -> bool:
    """
    Orchestrates unstaking (if needed), withdrawing liquidity, collecting, and burning an LP.
    Returns True if all critical steps seem successful, False otherwise.
    """
    if not nft_id_to_dismantle:
        logger.error("_fully_dismantle_lp: No NFT ID provided.")
        return False

    success_overall = True

    if initially_staked:
        unstake_ok = await _unstake_lp_from_gauge(context, nft_id_to_dismantle)
        if not unstake_ok:
            logger.error(f"Dismantling LP {nft_id_to_dismantle}: Unstaking failed. Aborting further dismantling.")
            return False

    withdraw_collect_ok = await _withdraw_collect_from_lp(context, nft_id_to_dismantle)
    if not withdraw_collect_ok:
        logger.warning(f"Dismantling LP {nft_id_to_dismantle}: Withdraw/collect step was not fully successful. Will still attempt burn.")
        success_overall = False

    burn_ok = await _burn_lp_nft(context, nft_id_to_dismantle)
    if not burn_ok:
        logger.warning(f"Dismantling LP {nft_id_to_dismantle}: Burn step was not confirmed successful (might be okay if already gone).")
        if not success_overall:
             pass
        else:
             success_overall = False


    if success_overall:
        logger.info(f"Successfully dismantled LP {nft_id_to_dismantle}.")
    else:
        logger.warning(f"LP {nft_id_to_dismantle} dismantling process had issues. Check logs.")
    
    return success_overall

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
                await asyncio.sleep(13)
                return True
            else:
                await send_tg_message(context, f"❌ Failed to approve {token_symbol_for_log} for **{spender_name}**.", menu_type=None)
                return False
        else:
            logger.info(f"Sufficient allowance for {token_symbol_for_log} by {spender_name} ({spender_address}) already exists.")
            await send_tg_message(context, f"ℹ️ Sufficient allowance for {token_symbol_for_log} by **{spender_name}** already exists.", menu_type=None)
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
        
        wblt_decimals_val = await asyncio.to_thread(wblt_token_contract.functions.decimals().call)
        usdc_decimals_val = await asyncio.to_thread(usdc_token_contract.functions.decimals().call)
        
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
        logger.info(f"Wallet {wallet_address} owns {balance} LP(s) from this manager.")

        if balance == 0:
            logger.info("No NFTs owned by wallet in this manager. No unstaked LP found.")
            return None, None

        for i in range(balance):
            token_id = None
            try:
                token_id = await asyncio.to_thread(
                    aerodrome_nft_manager_contract.functions.tokenOfOwnerByIndex(wallet_address, i).call
                )
                logger.debug(f"Checking wallet LP at index {i}: Token ID {token_id}")

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
                        logger.info(f"Wallet token ID {token_id} (WBLT/USDC) has 0 liquidity. Ignoring (likely a burned or empty LP).")

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
            await send_tg_message(context, f"Warning: LP {nft_id} is not for WBLT/USDC pair.", menu_type=None)
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
        logger.error(f"Error getting LP position details for LP {nft_id}: {e}")
        return None

async def get_pending_aero_rewards(context: CallbackContext, nft_id_to_check: int):
    if not nft_id_to_check: return Decimal("0")
    try:
        is_still_staked_by_bot = await asyncio.to_thread(
            aerodrome_gauge_contract.functions.stakedContains(BOT_WALLET_ADDRESS, nft_id_to_check).call
        )

        if not is_still_staked_by_bot:
            logger.info(f"LP {nft_id_to_check} is not currently staked by {BOT_WALLET_ADDRESS} in gauge {AERODROME_CL_GAUGE_ADDRESS}. Pending AERO assumed to be 0 or claimed.")
            return Decimal("0")

        earned_wei = await asyncio.to_thread(
            aerodrome_gauge_contract.functions.earned(BOT_WALLET_ADDRESS, nft_id_to_check).call
        )
        aero_decimals = await asyncio.to_thread(aero_token_contract.functions.decimals().call)
        return from_wei(earned_wei, aero_decimals)
    except ContractLogicError as cle:
        if cle.message and "NA" in cle.message:
             logger.warning(f"Gauge reverted with 'NA' for LP {nft_id_to_check} even after stakedContains check (or if check was bypassed). Assuming 0 rewards. Error: {cle}")
             return Decimal("0")
        logger.error(f"ContractLogicError getting pending AERO rewards for LP {nft_id_to_check}: {cle}", exc_info=True)
        return Decimal("0")
    except Exception as e:
        logger.error(f"Error getting pending AERO rewards for LP {nft_id_to_check}: {e}", exc_info=True)
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
            
            fmt_actual_input = ("{:" + input_format_str.strip(':') + "}").format(actual_input_used_by_api)
            fmt_amount_out = ("{:" + output_format_str.strip(':') + "}").format(amount_out_decimal)

            success_message = (
                f"✅ KyberSwap successful!\n"
                f"Swapped `{fmt_actual_input}` {token_in_symbol} "
                f"for `{fmt_amount_out}` {token_out_symbol}."
            )
            await send_tg_message(context, success_message, menu_type=None)
            await asyncio.sleep(13)
            return True, amount_out_decimal
        else:
            await send_tg_message(context, f"⚠️ KyberSwap swap for {token_in_symbol} failed (tx status 0 or not confirmed).", menu_type=None)
            return False, Decimal("0")

    except Exception as e:
        logger.error(f"Error in execute_kyberswap_swap for {token_in_contract.address} to {token_out_address}: {e}", exc_info=True)
        await send_tg_message(context, f"❌ Critical error during KyberSwap operation: {e}", menu_type=None)
        return False, Decimal("0")


async def attempt_operation_with_retries(
    operation_coro: OperationCoro,
    operation_name: str,
    context: CallbackContext,
    max_retries: int = 3,
    delay_seconds: int = 13
) -> Tuple[bool, Optional[Decimal]]:

    """
    Attempts an operation, retrying on failure.
    operation_coro should be an awaitable that returns a tuple: (success_bool, result_value_or_None).
    Example: (True, amount_out_decimal) or (False, None) or (False, Decimal(0))
    """
    last_exception = None
    for attempt in range(max_retries + 1):
        try:
            logger.info(f"Attempting {operation_name}, attempt {attempt + 1}/{max_retries + 1}...")

            op_success, *op_results = await operation_coro()

            returned_value = op_results[0] if op_results else None

            if op_success:
                logger.info(f"{operation_name} successful on attempt {attempt + 1}.")
                return True, returned_value
            else:
                logger.warning(f"{operation_name} failed on attempt {attempt + 1} (operation returned False). Result: {returned_value}")
                last_exception = RuntimeError(f"{operation_name} returned False.")
                if attempt < max_retries:
                    await send_tg_message(context, f"⚠️ {operation_name} failed (attempt {attempt+1}). Retrying in {delay_seconds}s...", menu_type=None)
                    await asyncio.sleep(delay_seconds)

        except ContractLogicError as cle:
            logger.error(f"ContractLogicError during {operation_name} (attempt {attempt + 1}): {getattr(cle, 'message', str(cle))} Data: {getattr(cle, 'data', 'N/A')}", exc_info=True)
            last_exception = cle
            if attempt < max_retries:
                await send_tg_message(context, f"⚠️ {operation_name} failed with ContractLogicError (attempt {attempt+1}): {str(getattr(cle, 'message', str(cle)))[:50]}. Retrying in {delay_seconds}s...", menu_type=None)
                await asyncio.sleep(delay_seconds)
        except Exception as e:
            logger.error(f"Unexpected error during {operation_name} (attempt {attempt + 1}): {e}", exc_info=True)
            last_exception = e
            if attempt < max_retries:
                await send_tg_message(context, f"⚠️ {operation_name} failed with error (attempt {attempt+1}): {str(e)[:50]}. Retrying in {delay_seconds}s...", menu_type=None)
                await asyncio.sleep(delay_seconds)
        
        if attempt >= max_retries:
            break

    logger.error(f"{operation_name} FAILED after {max_retries + 1} attempts. Last error: {type(last_exception).__name__ if last_exception else 'N/A'}: {str(last_exception)[:100]}")
    await send_tg_message(context, f"❌ {operation_name} FAILED definitively after {max_retries + 1} attempts. Please check logs.", menu_type=None)
    if last_exception and isinstance(last_exception, RuntimeError) and "returned False" in str(last_exception):
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

    sqrt_P_current = Decimal(current_pool_sqrt_price_x96) / (Decimal(2)**96)    
    sqrt_P_lower   = Decimal("1.0001") ** (Decimal(tick_lower) / Decimal(2))
    sqrt_P_upper   = Decimal("1.0001") ** (Decimal(tick_upper) / Decimal(2))

    logger.info(f"Targeted Swap SqrtPs: Current={sqrt_P_current:.30f}, Lower={sqrt_P_lower:.30f}, Upper={sqrt_P_upper:.30f}")

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
         InlineKeyboardButton(f"⚙️ Strat: {bot_state['current_strategy'][:12]}", callback_data="toggle_strategy")]
    )

    # Row 3: Manual Actions / Management
    keyboard_rows.append(
        [InlineKeyboardButton("💰 Claim & Sell AERO", callback_data="claim_sell_aero"),
         InlineKeyboardButton("🔄 Force Rebalance", callback_data="force_rebalance")]
    )
    
    # Row 4: Financial Management
    keyboard_rows.append(
        [InlineKeyboardButton("💸 Withdraw Funds", callback_data="withdraw_funds_options_menu"),
        InlineKeyboardButton("🛠️ LP & Wallet Actions", callback_data="manage_lp_wallet_menu")]
    )
    
    return InlineKeyboardMarkup(keyboard_rows)

async def get_startup_unstaked_lp_menu():
    keyboard = [
        [InlineKeyboardButton("✅ Stake this LP", callback_data=CB_STARTUP_STAKE_NFT)],
        [InlineKeyboardButton("🛑 Withdraw Liquidity from this LP", callback_data=CB_STARTUP_WITHDRAW_UNSTAKED_NFT)]
    ]
    return InlineKeyboardMarkup(keyboard)

async def get_startup_staked_lp_menu():
    keyboard = [
        [InlineKeyboardButton("▶️ Continue Monitoring (Normal Loop)", callback_data=CB_STARTUP_CONTINUE_MONITORING_STAKED)],
        [InlineKeyboardButton("🛠️ Unstake & Manage (Withdraw/Rebalance)", callback_data=CB_STARTUP_UNSTAKE_AND_MANAGE_STAKED)]
    ]
    return InlineKeyboardMarkup(keyboard)

async def get_withdraw_options_menu(context: CallbackContext):
    keyboard = [
        [InlineKeyboardButton("🏧 Withdraw Available Wallet USDC", callback_data="withdraw_wallet_usdc_menu")],
        [InlineKeyboardButton("🤑 Withdraw Tracked AERO Sale Profit", callback_data="withdraw_profit_menu")],
        [InlineKeyboardButton("⬅️ Back to Main Menu", callback_data="main_menu")]
    ]
    return InlineKeyboardMarkup(keyboard)

async def get_withdraw_wallet_usdc_menu(context: CallbackContext):
    bot_usdc_balance_live = await get_token_balance(usdc_token_contract, BOT_WALLET_ADDRESS)
    usdc_decimals = await asyncio.to_thread(usdc_token_contract.functions.decimals().call)

    balance_str = f"{bot_usdc_balance_live:.{usdc_decimals}f} USDC"
    keyboard = [
        [InlineKeyboardButton(f"Withdraw ALL Wallet USDC ({balance_str})", callback_data="withdraw_wallet_usdc_all")],
        [InlineKeyboardButton("Withdraw Custom Wallet USDC", callback_data="withdraw_wallet_usdc_custom")],
        [InlineKeyboardButton("⬅️ Back to Withdraw Options", callback_data="withdraw_funds_options_menu")]
    ]
    return InlineKeyboardMarkup(keyboard)

async def get_profit_withdrawal_keyboard():
    profit_available = bot_state.get('accumulated_profit_usdc', Decimal(0))
    profit_str = f"{profit_available:.2f} USDC"
    keyboard = [
        [InlineKeyboardButton(f"Withdraw ALL Profit ({profit_str})", callback_data="withdraw_profit_all")],
        [InlineKeyboardButton("Withdraw Custom Profit Amount", callback_data="withdraw_profit_custom")],
        [InlineKeyboardButton("⬅️ Back to Withdraw Options", callback_data="withdraw_funds_options_menu")]
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

async def get_manage_lp_wallet_keyboard():
    keyboard = [
        [InlineKeyboardButton("➕ Add Funds Info", callback_data="add_funds_info")],
        [InlineKeyboardButton("🆔 Set Initial LP NFT ID", callback_data="set_initial_lp_nft_id_prompt")],
        [InlineKeyboardButton("▶️ Stake Unstaked LP", callback_data="manual_stake_lp")],
        [InlineKeyboardButton("⬅️ Back to Main Menu", callback_data="main_menu")]
    ]
    return InlineKeyboardMarkup(keyboard)


# --- Telegram Command Handlers --- # probably not needed. slash commands suck
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

        elif action == "withdraw_funds_options_menu":
            await send_tg_message(context, "Select withdrawal type:", menu_type="withdraw_funds_options")

        # Wallet USDC Withdrawal
        elif action == "withdraw_wallet_usdc_menu":
            await handle_withdraw_wallet_usdc_menu_action(context)
        elif action == "withdraw_wallet_usdc_all":
            await handle_withdraw_wallet_usdc_all_action(context)
        elif action == "withdraw_wallet_usdc_custom":
            context.user_data['awaiting_wallet_usdc_withdrawal_amount'] = True
            await send_tg_message(context, "Enter amount of WALLET USDC to withdraw (e.g., 10.75):", menu_type=None)

        # Profit USDC Withdrawal
        elif action == "withdraw_profit_menu":
            await handle_withdraw_profit_menu_action(context)
        elif action == "withdraw_profit_all":
            await handle_withdraw_profit_all_action(context)
        elif action == "withdraw_profit_custom":
            context.user_data['awaiting_profit_withdrawal_amount'] = True
            await send_tg_message(context, "Enter amount of TRACKED PROFIT to withdraw (e.g., 10.75):", menu_type=None)

        elif action == "manage_lp_wallet_menu":
            await send_tg_message(context, "LP & Wallet Options:", menu_type="manage_lp_wallet")
        elif action == "add_funds_info":
            await send_tg_message(context, f"To add funds, send WBLT or USDC to the bot's address: `{BOT_WALLET_ADDRESS}`. Funds will be included in the next rebalance.", menu_type="main")
        elif action == "set_initial_lp_nft_id_prompt":
            context.user_data['awaiting_initial_lp_nft_id'] = True
            await send_tg_message(context, "Please type the Token ID of your existing WBLT-USDC LP owned by the bot:", menu_type=None)
        elif action == "manual_stake_lp":
            await handle_manual_stake_lp_action(context)

        elif action == "emergency_exit_confirm":
            await send_tg_message(context, "⚠️ **WARNING!** This will unstake, withdraw all LP funds, convert WBLT and AERO to USDC, and halt bot operations. Are you sure?", menu_type="emergency_exit_confirm")
        elif action == "emergency_exit_execute":
            await handle_emergency_exit_action(context)

        elif action == CB_STARTUP_STAKE_NFT:
            await handle_startup_stake_nft_action(context)
        elif action == CB_STARTUP_WITHDRAW_UNSTAKED_NFT:
            await handle_startup_withdraw_unstaked_nft_action(context)
        elif action == CB_STARTUP_CONTINUE_MONITORING_STAKED:
            await handle_startup_continue_monitoring_action(context)
        elif action == CB_STARTUP_UNSTAKE_AND_MANAGE_STAKED:
            await handle_startup_unstake_and_manage_action(context)

        elif action == "start_bot_operations":
            await handle_start_bot_operations_action(context)
        elif action == "pause_bot_operations":
            await handle_pause_bot_operations_action(context)
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

        elif context.user_data.get('awaiting_wallet_usdc_withdrawal_amount'):
            del context.user_data['awaiting_wallet_usdc_withdrawal_amount']
            try:
                amount = Decimal(user_text)
                if amount <= 0:
                    await send_tg_message(context, "Withdrawal amount must be positive.")
                else:
                    await _execute_fund_withdrawal(context, amount, usdc_token_contract, "Wallet USDC")
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

    price_usdc_in_usd = await get_dexscreener_price_usd(DEXSCREENER_PAIR_USDC_STABLE)
    if price_usdc_in_usd is None:
        logger.warning("Failed to get USDC/USD price from DexScreener, defaulting to 1.0 for USDC value.")
        price_usdc_in_usd = Decimal(1) 

    onchain_wblt_price_vs_usdc, current_tick = await get_aerodrome_pool_price_and_tick() 

    price_wblt_in_usd_ds = await get_dexscreener_price_usd(DEXSCREENER_PAIR_WBLT_USDC)
    if price_wblt_in_usd_ds is None:
        logger.warning("Failed to get WBLT/USD price from DexScreener. Status may use on-chain WBLT/USDC price.")
        if onchain_wblt_price_vs_usdc:
            price_wblt_in_usd_ds = onchain_wblt_price_vs_usdc * price_usdc_in_usd 
        else:
            price_wblt_in_usd_ds = Decimal(0) 

    price_aero_in_usd_ds = await get_dexscreener_price_usd(DEXSCREENER_PAIR_AERO_USDC)
    if price_aero_in_usd_ds is None:
        logger.warning("Failed to get AERO/USD price from DexScreener, defaulting to 0 for AERO value.")
        price_aero_in_usd_ds = Decimal(0)

    logger.info(f"DexScreener Prices for Status: USDC=${price_usdc_in_usd:.4f}, WBLT=${price_wblt_in_usd_ds:.4f}, AERO=${price_aero_in_usd_ds:.4f}")
    
    # Wallet Balances
    eth_balance_wei = await asyncio.to_thread(w3.eth.get_balance, BOT_WALLET_ADDRESS)
    status_lines.append(f"🔷 `{from_wei(eth_balance_wei, 18):.6f} ETH`") 

    usdc_bal = await get_token_balance(usdc_token_contract, BOT_WALLET_ADDRESS)
    wblt_bal = await get_token_balance(wblt_token_contract, BOT_WALLET_ADDRESS)
    aero_bal = await get_token_balance(aero_token_contract, BOT_WALLET_ADDRESS)

    usdc_bal_usd_value = usdc_bal * price_usdc_in_usd
    wblt_bal_usd_value = wblt_bal * price_wblt_in_usd_ds
    aero_bal_usd_value = aero_bal * price_aero_in_usd_ds

    status_lines.append(f"💰 `{usdc_bal:.2f} USDC` (`${usdc_bal_usd_value:.2f}`)")
    status_lines.append(f"🌯 `{wblt_bal:.4f} WBLT` (`${wblt_bal_usd_value:.2f}`)")
    status_lines.append(f"✈️ `{aero_bal:.4f} AERO` (`${aero_bal_usd_value:.2f}`)")
    status_lines.append("---")

    current_nft_id_in_state = bot_state.get("aerodrome_lp_nft_id")

    if current_nft_id_in_state:
        status_lines.append(f"💰 LP `{current_nft_id_in_state}`")
        position_details = await get_lp_position_details(context, current_nft_id_in_state)
                
        if position_details and onchain_wblt_price_vs_usdc is not None and current_tick is not None:
            tick_lower_lp = position_details['tickLower']
            tick_upper_lp = position_details['tickUpper']
            lp_liquidity = position_details['liquidity']

            staked_status_str = "(Unknown)"
            try:
                is_staked = await asyncio.to_thread(
                    aerodrome_gauge_contract.functions.stakedContains(BOT_WALLET_ADDRESS, current_nft_id_in_state).call
                )
                staked_status_str = "(Staked)" if is_staked else "(Unstaked)"
            except Exception as e_staked_check:
                logger.warning(f"Could not check staked status for LP {current_nft_id_in_state} in status: {e_staked_check}")

            is_in_actual_range = current_tick >= tick_lower_lp and current_tick < tick_upper_lp
            range_status_emoji = "✅ **In Range**" if is_in_actual_range else "❌ **Out of Range**"
            
            actual_tick_span_lp = tick_upper_lp - tick_lower_lp
            lower_trigger_tick_for_status = tick_lower_lp 
            upper_trigger_tick_for_status = tick_upper_lp
            buffer_tick_amount_aligned = Decimal(0)
            can_show_buffer_info = False

            if actual_tick_span_lp > 0:
                try:
                    pool_tick_spacing = await asyncio.to_thread(aerodrome_pool_contract.functions.tickSpacing().call)
                    if pool_tick_spacing > 0:
                        raw_buffer_in_ticks = actual_tick_span_lp * (REBALANCE_TRIGGER_BUFFER_PERCENTAGE / Decimal(100))
                        num_tick_spacings_for_buffer = int(Decimal(raw_buffer_in_ticks) / Decimal(pool_tick_spacing))
                        if num_tick_spacings_for_buffer == 0 and raw_buffer_in_ticks > 0:
                            num_tick_spacings_for_buffer = 1
                        buffer_tick_amount_aligned = Decimal(num_tick_spacings_for_buffer * pool_tick_spacing)
                        if (2 * buffer_tick_amount_aligned) >= actual_tick_span_lp:
                            buffer_tick_amount_aligned = Decimal(pool_tick_spacing) if actual_tick_span_lp > pool_tick_spacing else Decimal(0)
                        
                        lower_trigger_tick_for_status = tick_lower_lp + int(buffer_tick_amount_aligned)
                        upper_trigger_tick_for_status = tick_upper_lp - int(buffer_tick_amount_aligned)
                        can_show_buffer_info = True
                except Exception as e_buffer_calc_status:
                    logger.warning(f"Could not calculate buffer details for status: {e_buffer_calc_status}")
            
            status_line_text = f"{range_status_emoji} {staked_status_str}"
            is_within_buffer_zone = current_tick >= lower_trigger_tick_for_status and current_tick < upper_trigger_tick_for_status
            
            if is_in_actual_range:
                if not is_within_buffer_zone and actual_tick_span_lp > 0 and buffer_tick_amount_aligned > 0 : 
                     status_line_text += " (Near Edge)"
            status_lines.append(status_line_text)

            status_lines.append(f"📈 Tick: `{current_tick}`")

            wblt_decimals_val = await asyncio.to_thread(wblt_token_contract.functions.decimals().call)
            usdc_decimals_val = await asyncio.to_thread(usdc_token_contract.functions.decimals().call)
            decimal_adj_factor_for_price = Decimal(10)**(wblt_decimals_val - usdc_decimals_val)
            
            price_at_tick_lower_lp = (Decimal("1.0001")**Decimal(tick_lower_lp)) * decimal_adj_factor_for_price
            price_at_tick_upper_lp = (Decimal("1.0001")**Decimal(tick_upper_lp)) * decimal_adj_factor_for_price
                        
            status_lines.append(f"📐 Tick Range: `{tick_lower_lp}` to `{tick_upper_lp}`")
            status_lines.append(f"💲 Price Range: `{price_at_tick_lower_lp:.4f}` - `{price_at_tick_upper_lp:.4f}` `({TARGET_RANGE_WIDTH_PERCENTAGE}%)`") 

            if price_wblt_in_usd_ds > 0:
                 status_lines.append(f"💵 WBLT: `{onchain_wblt_price_vs_usdc:.4f} USDC` (`${price_wblt_in_usd_ds:.4f}`)")
            else:
                status_lines.append(f"💵 WBLT: `{onchain_wblt_price_vs_usdc:.4f} USDC`")

            if can_show_buffer_info:
                price_at_lower_trigger = (Decimal("1.0001")**Decimal(lower_trigger_tick_for_status)) * decimal_adj_factor_for_price
                price_at_upper_trigger = (Decimal("1.0001")**Decimal(upper_trigger_tick_for_status)) * decimal_adj_factor_for_price
                status_lines.append(
                    f"🔔 Rebalance Triggers: `<{price_at_lower_trigger:.4f}`, `>{price_at_upper_trigger:.4f}` "
                    f"`({REBALANCE_TRIGGER_BUFFER_PERCENTAGE}%)`"
                )
            elif actual_tick_span_lp <= 0:
                 status_lines.append("  ❗ LP range span is zero or negative, cannot calculate buffer.")
            else:
                 status_lines.append("  ❗ Could not determine buffer trigger prices.")
            
            status_lines.append(f"💧 Liquidity: `{lp_liquidity}`")

            actual_wblt_in_lp, actual_usdc_in_lp = await get_amounts_for_liquidity(
                lp_liquidity, current_tick, tick_lower_lp, tick_upper_lp,
                wblt_decimals_val, usdc_decimals_val
            )
            status_lines.append(f"💼 Principal: `{actual_wblt_in_lp:.4f} WBLT` & `{actual_usdc_in_lp:.2f} USDC`")
            
            est_value_wblt_usd = actual_wblt_in_lp * price_wblt_in_usd_ds
            est_value_usdc_usd = actual_usdc_in_lp * price_usdc_in_usd
            total_est_value_lp_usd = est_value_wblt_usd + est_value_usdc_usd
            status_lines.append(f"💲 Est. Value: `${total_est_value_lp_usd:.2f}`")
            
            uncollected_wblt_human = Decimal(0)
            uncollected_usdc_human = Decimal(0)
            pos_token0_addr = Web3.to_checksum_address(position_details['token0'])

            if pos_token0_addr == WBLT_TOKEN_ADDRESS:
                uncollected_wblt_human = from_wei(position_details['tokensOwed0_wei'], wblt_decimals_val)
                uncollected_usdc_human = from_wei(position_details['tokensOwed1_wei'], usdc_decimals_val)
            elif pos_token0_addr == USDC_TOKEN_ADDRESS:
                uncollected_usdc_human = from_wei(position_details['tokensOwed0_wei'], usdc_decimals_val)
                uncollected_wblt_human = from_wei(position_details['tokensOwed1_wei'], wblt_decimals_val)
            
            if uncollected_wblt_human > Decimal(0) or uncollected_usdc_human > Decimal(0):
                fees_wblt_usd_value = uncollected_wblt_human * price_wblt_in_usd_ds
                fees_usdc_usd_value = uncollected_usdc_human * price_usdc_in_usd
                total_fees_usd_value = fees_wblt_usd_value + fees_usdc_usd_value
                status_lines.append(f"🤑 Fees: `{uncollected_wblt_human:.4f} WBLT` & `{uncollected_usdc_human:.2f} USDC` (`~${total_fees_usd_value:.2f}`)")
            else:
                status_lines.append("🤑 Fees: `None`")

            pending_aero = await get_pending_aero_rewards(context, current_nft_id_in_state)
            pending_aero_usd_value = pending_aero * price_aero_in_usd_ds
            status_lines.append(f"🎁 AERO: `{pending_aero:.4f}` (`~${pending_aero_usd_value:.2f}`)")
        else:
            status_lines.append("🤷‍♀️ Could not fetch full LP details or price data.")
    else:
        status_lines.append("❌ No active Aerodrome LP position.")
    status_lines.append("---")

    status_lines.append(f"🧠 Strategy: `{bot_state['current_strategy']}`")
    profit_value_str = f"{bot_state.get('accumulated_profit_usdc', Decimal(0)):.2f}"
    status_lines.append(f"💸 Tracked Profit: `${profit_value_str} USDC`")
    status_lines.append(f"🛑 Halted: `{'YES' if bot_state['operations_halted'] else 'NO'}`")
    status_lines.append(f"🔒 Lock: `{'ENGAGED' if bot_state.get('is_processing_action', False) else 'FREE'}`")
    status_lines.append(f"🛠️ Setup: `{'YES' if bot_state.get('initial_setup_pending', True) else 'NO'}`")

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


async def handle_withdraw_funds_menu_action(context: CallbackContext):
    if not USER_PROFIT_WITHDRAWAL_ADDRESS or "YOUR_PERSONAL_WALLET_ADDRESS" in USER_PROFIT_WITHDRAWAL_ADDRESS:
        await send_tg_message(context, "⚠️ Withdrawal address is not configured. Please set it in the script.")
        return

    bot_usdc_balance_live = await get_token_balance(usdc_token_contract, BOT_WALLET_ADDRESS)
    usdc_decimals = await asyncio.to_thread(usdc_token_contract.functions.decimals().call)
    balance_str = f"{bot_usdc_balance_live:.{usdc_decimals}f} USDC"
    
    message = (
        f"Available liquid funds in bot wallet for withdrawal:\n"
        f"  💰 {balance_str}\n"
        f"Withdrawals will be sent to: `{USER_PROFIT_WITHDRAWAL_ADDRESS}`\n\n"
        "Choose an option:"
    )
    await send_tg_message(context, message, menu_type="withdraw_funds")

async def _execute_fund_withdrawal(context: CallbackContext, amount_decimal: Decimal, token_contract, token_symbol_for_log: str):
    if amount_decimal <= 0:
        await send_tg_message(context, "Withdrawal amount must be positive.")
        return

    bot_token_balance = await get_token_balance(token_contract, BOT_WALLET_ADDRESS)
    token_decimals = await asyncio.to_thread(token_contract.functions.decimals().call)

    if bot_token_balance < amount_decimal:
        await send_tg_message(context, f"Insufficient {token_symbol_for_log} in wallet. Requested: {amount_decimal:.{token_decimals}f}, Available: {bot_token_balance:.{token_decimals}f}.")
        return

    await send_tg_message(context, f"Attempting to withdraw {amount_decimal:.{token_decimals}f} {token_symbol_for_log} to {USER_PROFIT_WITHDRAWAL_ADDRESS}...", menu_type=None)
    
    amount_wei = to_wei(amount_decimal, token_decimals)

    tx_params = {'from': BOT_WALLET_ADDRESS}
    transfer_tx = token_contract.functions.transfer(USER_PROFIT_WITHDRAWAL_ADDRESS, amount_wei).build_transaction(tx_params)
    
    receipt = await asyncio.to_thread(_send_and_wait_for_transaction, transfer_tx, f"Withdraw {amount_decimal:.{token_decimals}f} {token_symbol_for_log}")

    if receipt and receipt.status == 1:
        new_bot_balance = await get_token_balance(token_contract, BOT_WALLET_ADDRESS)
        await send_tg_message(context, 
            f"✅ Successfully withdrew {amount_decimal:.{token_decimals}f} {token_symbol_for_log}.\n"
            f"New bot wallet balance: {new_bot_balance:.{token_decimals}f} {token_symbol_for_log}."
        )
    else:
        await send_tg_message(context, f"❌ {token_symbol_for_log} withdrawal of {amount_decimal:.{token_decimals}f} FAILED.")
    await save_state_async()

async def handle_withdraw_wallet_usdc_all_action(context: CallbackContext):
    bot_usdc_balance_live = await get_token_balance(usdc_token_contract, BOT_WALLET_ADDRESS)
    if bot_usdc_balance_live > Decimal("1.0"):
        await _execute_fund_withdrawal(context, bot_usdc_balance_live, usdc_token_contract, "Wallet USDC")
    else:
        await send_tg_message(context, f"ℹ️ No considerable USDC available in wallet to withdraw. `{bot_usdc_balance_live:.2f} USDC`")

async def handle_set_initial_lp_nft_id_action(context: CallbackContext, nft_id: int):
    await send_tg_message(context, f"Attempting to load LP: {nft_id}...", menu_type=None)
    try:
        owner = await asyncio.to_thread(aerodrome_nft_manager_contract.functions.ownerOf(nft_id).call)
        if owner != BOT_WALLET_ADDRESS:
            await send_tg_message(context, f"⚠️ Bot does not own LP {nft_id}. Current owner: {owner}")
            return

        position_details = await get_lp_position_details(context, nft_id)
        if not position_details:
            await send_tg_message(context, f"⚠️ Could not fetch details for LP {nft_id} or it's not a WBLT/USDC pair.")
            return
                
        bot_state["aerodrome_lp_nft_id"] = nft_id
        bot_state["initial_setup_pending"] = False
        
        await send_tg_message(context, f"✅ LP ID set to {nft_id}. Principal amounts will be accurately set after the next rebalance. Current tracked principals might be approximate.")
        
        await save_state_async()
        await handle_status_action(context)

    except Exception as e:
        logger.error(f"Error setting initial LP {nft_id}: {e}")
        await send_tg_message(context, f"❌ Error setting LP: {e}")

async def handle_withdraw_profit_menu_action(context: CallbackContext):
    if not USER_PROFIT_WITHDRAWAL_ADDRESS or "YOUR_PERSONAL_WALLET_ADDRESS" in USER_PROFIT_WITHDRAWAL_ADDRESS:
        await send_tg_message(context, "⚠️ Profit withdrawal address is not configured.")
        return

    profit_available = bot_state.get('accumulated_profit_usdc', Decimal(0))
    profit_str = f"{profit_available:.2f} USDC"
    message = (
        f"Available profit for withdrawal: {profit_str}\n"
        f"Withdrawals will be sent to: `{USER_PROFIT_WITHDRAWAL_ADDRESS}`\n\n"
        "Choose an option:"
    )
    await send_tg_message(context, message, menu_type="profit_withdrawal")

async def handle_withdraw_wallet_usdc_menu_action(context: CallbackContext):
    if not USER_PROFIT_WITHDRAWAL_ADDRESS or "YOUR_PERSONAL_WALLET_ADDRESS" in USER_PROFIT_WITHDRAWAL_ADDRESS:
        await send_tg_message(context, "⚠️ Withdrawal address is not configured.")
        return
        
    bot_usdc_balance_live = await get_token_balance(usdc_token_contract, BOT_WALLET_ADDRESS)
    usdc_decimals = await asyncio.to_thread(usdc_token_contract.functions.decimals().call)
    balance_str = f"{bot_usdc_balance_live:.{usdc_decimals}f} USDC"
    
    message = (
        f"Available liquid USDC in bot wallet: {balance_str}\n"
        f"Withdrawals sent to: `{USER_PROFIT_WITHDRAWAL_ADDRESS}`\n\n"
        "Choose an option for WALLET USDC:"
    )
    await send_tg_message(context, message, menu_type="withdraw_wallet_usdc")

async def _execute_profit_withdrawal(context: CallbackContext, amount_decimal: Decimal):
    if amount_decimal <= 0:
        await send_tg_message(context, "Withdrawal amount must be positive.")
        return

    current_accumulated_profit = bot_state.get('accumulated_profit_usdc', Decimal(0))
    if amount_decimal > current_accumulated_profit:
        await send_tg_message(context, f"ℹ️ Insufficient profit. Requested: {amount_decimal:.2f}, Available: {current_accumulated_profit:.2f} USDC.")
        return

    bot_usdc_wallet_balance = await get_token_balance(usdc_token_contract, BOT_WALLET_ADDRESS)
    if bot_usdc_wallet_balance < amount_decimal:
        await send_tg_message(context, 
            f"⚠️ Bot's USDC wallet balance ({bot_usdc_wallet_balance:.2f}) is less than requested profit withdrawal ({amount_decimal:.2f}). "
            f"This shouldn't happen if profit tracking is correct. Manual check needed."
        )
        return

    usdc_decimals = await asyncio.to_thread(usdc_token_contract.functions.decimals().call)
    await send_tg_message(context, f"Attempting to withdraw {amount_decimal:.2f} USDC profit to {USER_PROFIT_WITHDRAWAL_ADDRESS}...", menu_type=None)
    
    amount_wei = to_wei(amount_decimal, usdc_decimals)
    tx_params = {'from': BOT_WALLET_ADDRESS}
    transfer_tx = usdc_token_contract.functions.transfer(USER_PROFIT_WITHDRAWAL_ADDRESS, amount_wei).build_transaction(tx_params)
    
    receipt = await asyncio.to_thread(_send_and_wait_for_transaction, transfer_tx, f"Withdraw {amount_decimal:.2f} USDC Profit")

    if receipt and receipt.status == 1:
        bot_state["accumulated_profit_usdc"] = current_accumulated_profit - amount_decimal # Decrement tracked profit
        await save_state_async()
        await send_tg_message(context, f"✅ Successfully withdrew {amount_decimal:.2f} USDC profit! Remaining tracked profit: {bot_state['accumulated_profit_usdc']:.2f} USDC.")
    else:
        await send_tg_message(context, f"❌ Profit withdrawal of {amount_decimal:.2f} USDC FAILED.")

async def handle_withdraw_profit_all_action(context: CallbackContext):
    profit_to_withdraw = bot_state.get('accumulated_profit_usdc', Decimal(0))
    if profit_to_withdraw > Decimal("1.0"):
        await _execute_profit_withdrawal(context, profit_to_withdraw)
    else:
        await send_tg_message(context, f"ℹ️ No considerable profit available to withdraw. `{profit_to_withdraw:.2f} USDC`")

async def handle_withdraw_profit_custom_action(context: CallbackContext, amount: Decimal):
    await _execute_profit_withdrawal(context, amount)

async def handle_manual_stake_lp_action(context: CallbackContext):
    nft_id_to_stake = bot_state.get("aerodrome_lp_nft_id")
    action_description = "Manual Stake LP"

    if not nft_id_to_stake:
        await send_tg_message(context, "ℹ️ No active LP found in bot state to stake.", menu_type="main")
        return

    # Check if operations were halted, and if so, offer to resume for this action
    if bot_state.get("operations_halted", True):
        await send_tg_message(context, f"Operations are paused. Resuming temporarily for {action_description}...", menu_type=None)
    
    await send_tg_message(context, f"Attempting to manually stake LP `{nft_id_to_stake}`...", menu_type=None)

    # Verify it's not already staked
    try:
        is_already_staked = await asyncio.to_thread(
            aerodrome_gauge_contract.functions.stakedContains(BOT_WALLET_ADDRESS, nft_id_to_stake).call
        )
        if is_already_staked:
            await send_tg_message(context, f"✅ LP `{nft_id_to_stake}` is already staked.", menu_type="main")
            await handle_status_action(context)
            return
    except Exception as e_check:
        logger.error(f"Error checking if NFT {nft_id_to_stake} is already staked: {e_check}")
        await send_tg_message(context, f"⚠️ Could not verify if LP {nft_id_to_stake} is already staked. Proceed with caution or check manually.", menu_type="main")
        # Optionally, you could ask for confirmation here to proceed anyway.

    # Verify it has liquidity (optional but good)
    position_details = await get_lp_position_details(context, nft_id_to_stake)
    if not (position_details and position_details['liquidity'] > 0):
        await send_tg_message(context, f"⚠️ LP `{nft_id_to_stake}` has no liquidity or details are unavailable. Cannot stake.", menu_type="main")
        return

    stake_successful = await _execute_stake_lp_nft(context, nft_id_to_stake)

    if stake_successful:
        await send_tg_message(context, f"✅ Manual staking of LP `{nft_id_to_stake}` SUCCEEDED.", menu_type="main")
    else:
        await send_tg_message(context, f"❌ Manual staking of LP `{nft_id_to_stake}` FAILED. Check logs.", menu_type="main")
    
    await save_state_async()
    await handle_status_action(context)

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
        # Steps 1 & 2 combined: Unstake (if staked) and Withdraw/Collect/Burn
        if original_nft_id:
            logger.info(f"Processing existing LP: {original_nft_id} for rebalance.")
            is_staked_initially = False
            try:
                is_staked_initially = await asyncio.to_thread(
                    aerodrome_gauge_contract.functions.stakedContains(BOT_WALLET_ADDRESS, original_nft_id).call
                )
            except Exception:
                logger.warning(f"Could not determine staked status for {original_nft_id} before dismantling.")

            dismantle_success = await _fully_dismantle_lp(context, original_nft_id, is_staked_initially)
            
            if dismantle_success:
                logger.info(f"Successfully dismantled old LP {original_nft_id}.")
                bot_state["aerodrome_lp_nft_id"] = None
            else:
                logger.error(f"Failed to fully dismantle old LP {original_nft_id}. Aborting rebalance.")
                await send_tg_message(context, f"⚠️ Failed to fully dismantle old LP {original_nft_id}. Rebalance aborted. Manual check needed.\n\n Press **Force Rebalance** to try again.", menu_type="main")
                # Potentially set bot_state["operations_halted"] = True
                return
        else:
            logger.info("No existing LP in state. Proceeding with available wallet funds.")
            await send_tg_message(context, "ℹ️ No existing LP found. Will use wallet funds for new position.", menu_type=None)

        # 3. Consolidate funds
        available_wblt = await get_token_balance(wblt_token_contract, BOT_WALLET_ADDRESS)
        available_usdc = await get_token_balance(usdc_token_contract, BOT_WALLET_ADDRESS)
        available_aero = await get_token_balance(aero_token_contract, BOT_WALLET_ADDRESS)
        logger.info(f"Consolidated Funds - WBLT: {available_wblt}, USDC: {available_usdc}, AERO: {available_aero}")

        # 4. Sell ALL AERO (from wallet, which includes any just auto-claimed from unstake)
        logger.info("Attempting to sell any AERO present in the wallet...")
        sell_success, usdc_from_aero_sale = await _sell_all_available_aero_in_wallet(context)
        if sell_success and usdc_from_aero_sale > 0:
            available_usdc += usdc_from_aero_sale
            logger.info(f"Added {usdc_from_aero_sale:.2f} USDC from AERO sale to available funds.")
        logger.info(f"Funds after AERO sale attempt - WBLT: {available_wblt}, USDC: {available_usdc}")
        
        # 5. Determine New Optimal LP Range
        price_wblt_human_for_range, pool_current_tick = await get_aerodrome_pool_price_and_tick()
        slot0_data = await asyncio.to_thread(aerodrome_pool_contract.functions.slot0().call)
        current_pool_sqrt_price_x96_raw = slot0_data[0]
        if price_wblt_human_for_range is None or pool_current_tick is None:
            await send_tg_message(context, "❌ Cannot get current pool price/tick. Rebalance aborted.\n  Press **Force Rebalance** to try again.", menu_type=None)
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

        # 7 & 8. Approve and Deposit Liquidity (Mint)
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

        if desired_usdc_wei > 0:
            approve_usdc_amount_dec_buffered = final_usdc_for_mint * Decimal("1.001")
            approved_usdc = await approve_token_spending(context, usdc_token_contract, AERODROME_SLIPSTREAM_NFT_MANAGER_ADDRESS, approve_usdc_amount_dec_buffered)
            if not approved_usdc:
                logger.error("USDC approval failed. Minting aborted.")
                return
            logger.info("USDC approval for optimal amount successful or already sufficient.")

        # Mint parameters
        amount0_min_wei = 0 
        amount1_min_wei = 0
        sqrt_price_x96_limit_for_mint = 0

        mint_params_as_tuple = (
            Web3.to_checksum_address(WBLT_TOKEN_ADDRESS), Web3.to_checksum_address(USDC_TOKEN_ADDRESS),
            int(tick_spacing), int(new_tick_lower), int(new_tick_upper),
            int(desired_wblt_wei), int(desired_usdc_wei),
            int(amount0_min_wei), int(amount1_min_wei),
            Web3.to_checksum_address(BOT_WALLET_ADDRESS), int(time.time()) + 600,
            int(sqrt_price_x96_limit_for_mint)
        )
        logger.info(f"Prepared 12-ELEMENT params tuple for mint: {mint_params_as_tuple}")

         # Wrap the mint operation
        mint_op_callable = functools.partial(_execute_mint_lp_operation, context, mint_params_as_tuple)
    
        mint_op_success, mint_op_result_data = await attempt_operation_with_retries(
            mint_op_callable,
            "Full Mint LP Operation",
            context,
            max_retries=2,
            delay_seconds=45 
        )

        mint_receipt = None

        if mint_op_success:
            mint_receipt = mint_op_result_data
            logger.info("Mint LP operation fully successful.")
        else:
            error_detail = str(mint_op_result_data)[:150] if mint_op_result_data else "details unavailable"
            await send_tg_message(context, f"❌ Minting new LP FAILED after all attempts. Error context: {error_detail}", menu_type=None)
            return

        # --- Event Parsing & State Update (was Step 9) ---
        if not (mint_receipt and mint_receipt.status == 1):
            logger.error("Inconsistent state: Mint operation reported success, but receipt is invalid. Aborting.")
            await send_tg_message(context, "❌ Internal error after mint attempt. Aborting.", menu_type=None)
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
                        logger.info(f"Found Transfer event for new LP: {new_nft_id} to bot wallet.")
                        break
        if new_nft_id is None:
            await send_tg_message(context, "❌ Could not find new LP from mint events. Manual check needed.", menu_type=None)
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
                    logger.info(f"Decoded from IncreaseLiquidity for LP {new_nft_id}: WBLT_wei={actual_wblt_deposited_wei}, USDC_wei={actual_usdc_deposited_wei}")
                    break
        if not found_increase_liquidity_event:
            await send_tg_message(context, f"⚠️ Could not parse IncreaseLiquidity event for LP {new_nft_id}. Principals may be inaccurate.", menu_type=None)

        # 9. Update Principal & State
        bot_state["aerodrome_lp_nft_id"] = new_nft_id
        bot_state["initial_setup_pending"] = False 
        logger.info(f"LP Minted with Actual: {from_wei(actual_wblt_deposited_wei, wblt_decimals_val):.{wblt_decimals_val}f} WBLT, "
            f"{from_wei(actual_usdc_deposited_wei, usdc_decimals_val):.{usdc_decimals_val}f} USDC for NFT {new_nft_id}.")

        # 10. Stake New LP
        if new_nft_id:
            stake_op_callable = functools.partial(_execute_stake_lp_nft, context, new_nft_id)
        
            stake_op_success, _ = await attempt_operation_with_retries(
                stake_op_callable,
                f"Stake LP NFT {new_nft_id}",
                context,
                max_retries=2,
                delay_seconds=30
            )

            if not stake_op_success:
                logger.warning(f"Automated staking of new LP NFT {new_nft_id} FAILED after all attempts. It remains in the wallet.")
                await send_tg_message(context, f"⚠️ Automated staking for NFT {new_nft_id} FAILED after retries. Please check status and stake manually if needed.", menu_type="main")
        else:
            logger.error("No new_nft_id available to stake after minting (should have been caught earlier).")

    except Exception as e:
        logger.error(f"Critical error during process_full_rebalance: {e}", exc_info=True)
        await send_tg_message(context, f"⚠️ Critical error during rebalance: {str(e)[:200]}\n\n Press **Force Rebalance** to try again.", menu_type=None)
    finally:
        await save_state_async()
        await send_tg_message(context, "🤖 Full Rebalance Process Attempt Finished.", menu_type=None)
        await asyncio.sleep(1) 
        await handle_status_action(context)


async def process_claim_sell_aero(context: CallbackContext, triggered_by="auto"):
    if bot_state["operations_halted"] and triggered_by == "auto":
        logger.info("AERO claim/sell skipped (auto): Operations halted.")
        return
    if not bot_state.get("aerodrome_lp_nft_id"):
        logger.info("AERO claim/sell skipped: No active LP NFT ID.")
        await send_tg_message(context, "ℹ️ No active LP to claim AERO from.", menu_type=None)
        return

    await send_tg_message(context, f"ℹ️ Initiating AERO Claim & Sell (Trigger: `{triggered_by}`)...", menu_type=None)
    
    nft_id_to_claim = bot_state["aerodrome_lp_nft_id"]
    claim_successful = await _claim_rewards_for_staked_nft(context, nft_id_to_claim)
    
    if claim_successful:
        sell_successful, usdc_from_sale = await _sell_all_available_aero_in_wallet(context)
        
        if sell_successful and usdc_from_sale > 0:
            if bot_state["current_strategy"] == "take_profit":
                bot_state["accumulated_profit_usdc"] = bot_state.get("accumulated_profit_usdc", Decimal(0)) + usdc_from_sale
                logger.info(f"Added {usdc_from_sale:.2f} USDC from AERO sale to accumulated profit. New total: {bot_state['accumulated_profit_usdc']:.2f}")
                await send_tg_message(context, f"✅ AERO sold. Profit of `{usdc_from_sale:.2f}` USDC added to profit pool.", menu_type=None)
            else:
                logger.info(f"Sold AERO for {usdc_from_sale:.2f} USDC (compound strategy). Funds in wallet.")
                await send_tg_message(context, f"✅ Sold AERO for `{usdc_from_sale:.2f}` USDC. Funds are in bot wallet for future compounding.", menu_type=None)
        elif sell_successful and usdc_from_sale == 0:
            logger.info("No AERO was sold (either balance too low after claim or already sold).")
    else:
        logger.warning(f"AERO claim step failed for LP {nft_id_to_claim}. Skipping AERO sale.")
        await send_tg_message(context, f"⚠️ AERO claim failed for NFT {nft_id_to_claim}. Sale skipped.", menu_type=None)
    
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
    
    nft_id_to_exit = bot_state.get("aerodrome_lp_nft_id")
    dismantle_needed_and_possible = False
    initial_staked_status_for_exit = False

    if nft_id_to_exit:
        pos_details = await get_lp_position_details(context, nft_id_to_exit)
        if pos_details:
            dismantle_needed_and_possible = True
            try:
                initial_staked_status_for_exit = await asyncio.to_thread(
                    aerodrome_gauge_contract.functions.stakedContains(BOT_WALLET_ADDRESS, nft_id_to_exit).call
                )
            except Exception:
                logger.warning(f"Emergency Exit: Could not determine staked status for {nft_id_to_exit}.")
        else:
            logger.info(f"Emergency Exit: LP {nft_id_to_exit} in state, but no position details found. Assuming already gone.")
            bot_state["aerodrome_lp_nft_id"] = None

    if dismantle_needed_and_possible and nft_id_to_exit:
        await _fully_dismantle_lp(context, nft_id_to_exit, initial_staked_status_for_exit)
        bot_state["aerodrome_lp_nft_id"] = None

    # 3. Claim any AERO from gauge (might have been auto-claimed on unstake, but try again)
    await send_tg_message(context, "ℹ️ Checking for AERO to sell...", menu_type=None)
    await _sell_all_available_aero_in_wallet(context)


    # 4. Sell all WBLT for USDC
    wblt_balance = await get_token_balance(wblt_token_contract, BOT_WALLET_ADDRESS)
    if wblt_balance > Decimal("0.1"):
        await send_tg_message(context, f"ℹ️ Selling `{wblt_balance:.4f}` WBLT for USDC...", menu_type=None)
        success, _ = await execute_kyberswap_swap(context, wblt_token_contract, USDC_TOKEN_ADDRESS, wblt_balance)
        if success:
            await send_tg_message(context, "✅ WBLT sold for USDC!", menu_type=None)
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
        await send_tg_message(context, "⚠️ No NFT ID found in state to stake (startup). Please restart bot.", menu_type="main")
        return

    await send_tg_message(context, f"Attempting to stake discovered (unstaked) LP `{nft_id}` as per startup choice...", menu_type=None)
    
    stake_successful = await _execute_stake_lp_nft(context, nft_id)

    if stake_successful:
        await send_tg_message(context, f"✅ LP {nft_id} successfully STAKED! Resuming normal operations.", menu_type="main")
        bot_state["operations_halted"] = False 
        bot_state["initial_setup_pending"] = False
    else:
        await send_tg_message(context, f"⚠️ Failed to stake LP {nft_id} during startup. Remains in wallet. Bot HALTED.", menu_type="main")
        bot_state["operations_halted"] = True 
    await save_state_async()
    await handle_status_action(context)

async def handle_startup_withdraw_unstaked_nft_action(context: CallbackContext):
    nft_id = bot_state.get("aerodrome_lp_nft_id")
    if not nft_id:
        await send_tg_message(context, "⚠️ No LP found in state to withdraw. Please restart bot.", menu_type="main")
        bot_state["operations_halted"] = True
        await save_state_async()
        return
    
    wblt_decimals_val = await asyncio.to_thread(wblt_token_contract.functions.decimals().call)
    # usdc_decimals_val = await asyncio.to_thread(usdc_token_contract.functions.decimals().call) # Not strictly needed here if only converting to USDC

    withdraw_collect_ok = await _withdraw_collect_from_lp(context, nft_id)
    if withdraw_collect_ok:
        await _burn_lp_nft(context, nft_id)

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
        await send_tg_message(context, "⚠️ No staked LP found in state. Please restart bot.", menu_type="main")
        bot_state["operations_halted"] = True
        return

    await send_tg_message(context, f"✅ Resuming normal monitoring for stakted LP `{nft_id}`...", menu_type="none")
    bot_state["operations_halted"] = False
    bot_state["initial_setup_pending"] = False
    await save_state_async()
    await handle_status_action(context)


async def handle_startup_unstake_and_manage_action(context: CallbackContext):
    nft_id = bot_state.get("aerodrome_lp_nft_id")
    if not nft_id:
        await send_tg_message(context, "⚠️ No staked LP found in state. Please restart bot.", menu_type="main")
        bot_state["operations_halted"] = True
        await save_state_async()
        return

    await send_tg_message(context, f"ℹ️ Attempting to unstake, withdraw liquidity, and burn LP `{nft_id}`...", menu_type=None)

    # Since this is called on a "staked" LP from startup discovery:
    dismantle_success = await _fully_dismantle_lp(context, nft_id, initially_staked=True)

    if dismantle_success:
        await send_tg_message(context, f"✅ LP {nft_id} fully dismantled. Bot holds liquid WBLT/USDC.", menu_type="main")
    else:
        await send_tg_message(context, f"⚠️ LP {nft_id} dismantling had issues. Please check logs. Bot holds liquid assets as best as possible.", menu_type="main")

    bot_state["aerodrome_lp_nft_id"] = None
    bot_state["initial_setup_pending"] = True
    bot_state["operations_halted"] = True
    
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
                        logger.warning(f"LP {bot_state['aerodrome_lp_nft_id']} has zero or negative tick span. Skipping rebalance check.")
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

                        # Ensure buffer_tick_amount_aligned doesn't make trigger points cross
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
                    await handle_status_action(context)
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
                f"ℹ️ Approving LP `{token_id_to_approve}` for spender: **{spender_name}**...", 
                menu_type=None
            )
            
            approve_tx_params = {'from': BOT_WALLET_ADDRESS}
            approve_tx = nft_contract.functions.approve(spender_address, token_id_to_approve).build_transaction(approve_tx_params)
            
            receipt = await asyncio.to_thread(
                _send_and_wait_for_transaction, 
                approve_tx, 
                f"Approve LP {token_id_to_approve} for {spender_name}"
            )

            if receipt is not None and receipt.status == 1:
                await send_tg_message(context, f"✅ Approved LP `{token_id_to_approve}` for **{spender_name}**!", menu_type=None)
                await asyncio.sleep(13)
                return True
            else:
                await send_tg_message(context, f"❌ Failed to approve LP `{token_id_to_approve}` for **{spender_name}**.", menu_type=None)
                return False
        else:
            logger.info(f"ℹ️ LP {token_id_to_approve} already approved for {spender_name} ({spender_address}).")
            return True
    except Exception as e:
        spender_name_for_error = CONTRACT_NAME_MAP.get(Web3.to_checksum_address(spender_address), spender_address)
        logger.error(f"Error in approve_nft_for_spending for LP {token_id_to_approve} to {spender_name_for_error}: {e}", exc_info=True)
        await send_tg_message(context, f"Error approving LP {token_id_to_approve} for {spender_name_for_error}: {str(e)[:100]}", menu_type=None)
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
        logger.info(f"Using initial LP from config: {INITIAL_LP_NFT_ID_CONFIG} (will be verified by discovery)")
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
            logger.info(f"Discovered LP: {discovered_nft_id}, Status: {discovered_status}")
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
                logger.error(f"Discovered LP {discovered_nft_id} but with unknown status: {discovered_status}. Halting.")
                await send_tg_message(startup_context, f"⚠️ Error: Discovered LP {discovered_nft_id} with unknown status. Manual check needed. Bot HALTED.", menu_type="main")
                bot_state["operations_halted"] = True
        else:
            logger.info("No active WBLT/USDC LP LP discovered on-chain for the bot.")

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
