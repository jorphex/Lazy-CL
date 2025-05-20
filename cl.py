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

# Bot Settings (adjustablel to whatever you want)
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

# Gas
MAX_FEE_PER_GAS_GWEI = Decimal("0.005")
MAX_PRIORITY_FEE_PER_GAS_GWEI = Decimal("0.005")

STATE_FILE = "aerodrome_bot_state.json"
ABI_DIR = "abis"

# Callback data for startup LP discovery choices
CB_STARTUP_STAKE_NFT = "startup_stake_nft"
CB_STARTUP_WITHDRAW_UNSTAKED_NFT = "startup_withdraw_unstaked_nft"
CB_STARTUP_CONTINUE_MONITORING_STAKED = "startup_continue_monitoring_staked"
CB_STARTUP_UNSTAKE_AND_MANAGE_STAKED = "startup_unstake_and_manage_staked"

# --- Precision for Decimal calculations ---
getcontext().prec = 60

# --- Logging Setup ---
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - %(message)s',
    level=logging.INFO,
    handlers=[
        logging.FileHandler("aerodrome_bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# noise
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
    "is_processing_action": False, # to to prevent overlapping operations
    "initial_setup_pending": True
}

# --- Web3 ---
w3 = Web3(Web3.HTTPProvider(ALCHEMY_RPC_URL))

# --- ABI ---
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

# Load ABIs
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
    # defaults
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
    elif menu_type == "manage_principal": # Add this case
        keyboard = await get_manage_principal_keyboard()
    elif menu_type == "startup_unstaked_lp": # New
        keyboard = await get_startup_unstaked_lp_menu()
    elif menu_type == "startup_staked_lp": # New
        keyboard = await get_startup_staked_lp_menu()

    try:
        safe_message = telegramify_markdown.markdownify(message) # Assuming you've added telegramify

        await context.bot.send_message(
            chat_id=TELEGRAM_ADMIN_USER_ID,
            text=safe_message, 
            reply_markup=keyboard,
            parse_mode=ParseMode.MARKDOWN_V2 
        )
    except Exception as e:
        logger.error(f"Error sending Telegram message: {e} - Original Message: {message[:200]} - Safe Message: {safe_message[:200] if 'safe_message' in locals() else 'N/A'}")


# --- Web3 ---
def check_connection():
    if not w3.is_connected():
        logger.error("Web3 not connected!")
        return False
    return True

def get_nonce():
    return w3.eth.get_transaction_count(BOT_WALLET_ADDRESS, 'pending')

def _send_and_wait_for_transaction(tx_params_dict_to_sign, description="Transaction"):
    logger.debug(f"Attempting to sign and send for {description}. Input type: {type(tx_params_dict_to_sign)}. Value: {str(tx_params_dict_to_sign)[:300]}")
    
    signed_tx_object = None

    try:
        if not isinstance(tx_params_dict_to_sign, dict):
            logger.error(f"CRITICAL DEBUG: _send_and_wait_for_transaction expected a dict for {description}, but got {type(tx_params_dict_to_sign)}")
            return None

        tx_params = tx_params_dict_to_sign
        
        if 'gasPrice' not in tx_params and ('maxFeePerGas' not in tx_params or 'maxPriorityFeePerGas' not in tx_params):
            base_fee = w3.eth.get_block('latest')['baseFeePerGas']
            tx_params['maxPriorityFeePerGas'] = w3.to_wei(MAX_PRIORITY_FEE_PER_GAS_GWEI, 'gwei')
            calculated_max_fee = base_fee + tx_params['maxPriorityFeePerGas']
            buffer = w3.to_wei('0.001', 'gwei') 
            tx_params['maxFeePerGas'] = calculated_max_fee + buffer
            if tx_params['maxFeePerGas'] < tx_params['maxPriorityFeePerGas']:
                tx_params['maxFeePerGas'] = tx_params['maxPriorityFeePerGas'] + buffer

        if 'gas' not in tx_params:
            try:
                if 'chainId' not in tx_params:
                     tx_params['chainId'] = w3.eth.chain_id
                tx_params['gas'] = w3.eth.estimate_gas(tx_params)
            except Exception as e:
                logger.warning(f"Could not estimate gas for {description}: {e}. Using default 500,000. Params: {tx_params}")
                tx_params['gas'] = 500000
        
        if 'chainId' not in tx_params:
            tx_params['chainId'] = w3.eth.chain_id
            logger.debug(f"Added chainId {tx_params['chainId']} for {description}")

        logger.debug(f"Transaction parameters for signing ({description}): {tx_params}")
        try:
            signed_tx_object = w3.eth.account.sign_transaction(tx_params, BOT_PRIVATE_KEY)
            logger.debug(f"Signing successful for {description}. Signed object type: {type(signed_tx_object)}")
        except Exception as signing_error:
            logger.error(f"Error DURING SIGNING transaction for {description}: {signing_error}", exc_info=True)
            return None

        if not hasattr(signed_tx_object, 'raw_transaction'):
            logger.error(f"Signed object for {description} LACKS raw_transaction attribute immediately after signing! Type: {type(signed_tx_object)}")
            if signed_tx_object:
                logger.error(f"Attributes of signed_tx_object: {dir(signed_tx_object)}")
            return None
        if signed_tx_object.raw_transaction is None:
            logger.error(f"Signed object for {description} has raw_transaction attribute but it is None!")
            return None

        tx_hash = w3.eth.send_raw_transaction(signed_tx_object.raw_transaction)
        logger.info(f"{description} sent. Tx Hash: {tx_hash.hex()}")

        receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=TRANSACTION_TIMEOUT_SECONDS)

        if receipt.status == 1:
            logger.info(f"{description} successful. Tx: {tx_hash.hex()}, Gas used: {receipt.gasUsed}")
            time.sleep(15)
            return receipt
        else:
            logger.error(f"{description} FAILED. Tx: {tx_hash.hex()}, Status: {receipt.status}, Gas used: {receipt.gasUsed}")
            return None

    except TransactionNotFound:
        current_tx_hash = 'N/A'
        if 'tx_hash' in locals() and tx_hash:
            current_tx_hash = tx_hash.hex()
        logger.error(f"{description} timed out (not found after {TRANSACTION_TIMEOUT_SECONDS}s). Tx Hash: {current_tx_hash}")
        return None
    except ContractLogicError as cle:
        current_tx_hash = 'N/A'
        if 'tx_hash' in locals() and tx_hash:
            current_tx_hash = tx_hash.hex()
        logger.error(f"{description} ContractLogicError: {cle}. Tx Hash: {current_tx_hash}")
        return None
    except ValueError as ve: 
        logger.error(f"{description} ValueError: {ve}. This might indicate a revert reason or issue with tx params.")
        return None
    except Exception as e:
        current_tx_hash = 'N/A'
        if 'tx_hash' in locals() and tx_hash:
            current_tx_hash = tx_hash.hex()
        logger.error(f"An unexpected error occurred during {description}: {e}. Tx Hash: {current_tx_hash}", exc_info=True)
        return None


async def approve_token_spending(context: CallbackContext, token_contract, spender_address, amount_decimal):
    try:
        decimals = await asyncio.to_thread(token_contract.functions.decimals().call)
        amount_wei = to_wei(amount_decimal, decimals)
        token_symbol_for_log = await asyncio.to_thread(token_contract.functions.symbol().call)

        current_allowance = await asyncio.to_thread(
            token_contract.functions.allowance(BOT_WALLET_ADDRESS, spender_address).call
        )

        if current_allowance < amount_wei:
            await send_tg_message(context, f"👍 Approving `{from_wei(amount_wei, decimals)}` {token_symbol_for_log} for `{spender_address}`...", menu_type=None)
            
            current_nonce = await asyncio.to_thread(get_nonce)
            chain_id = await asyncio.to_thread(lambda: w3.eth.chain_id)

            approve_tx_params_for_build = {
                'from': BOT_WALLET_ADDRESS,
                'nonce': current_nonce,
                'chainId': chain_id 
                # 'gas': some_estimated_gas, # Optional: _send_and_wait_for_transaction can estimate
                # 'maxFeePerGas': ...,      # Optional: _send_and_wait_for_transaction can set
                # 'maxPriorityFeePerGas': ... # Optional: _send_and_wait_for_transaction can set
            }
            
            approve_tx_dict = token_contract.functions.approve(spender_address, amount_wei).build_transaction(approve_tx_params_for_build)
            logger.debug(f"Built approve_tx_dict for {token_symbol_for_log}: {approve_tx_dict}")

            receipt = await asyncio.to_thread(_send_and_wait_for_transaction, approve_tx_dict, f"Approve {token_symbol_for_log}")
            return receipt is not None and receipt.status == 1
        else:
            logger.info(f"Sufficient allowance for {token_symbol_for_log} by {spender_address} already exists.")
            return True
    except Exception as e:
        logger.error(f"Error in approve_token_spending for {token_contract.address}: {e}", exc_info=True)
        await send_tg_message(context, f"Error approving token {token_contract.address}: {e}", menu_type=None)
        return False

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
        tick_lower_raw = center_tick_raw - Decimal(tick_spacing * 5) # Arbitrary offset
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
        raw_unit_ratio_wblt_per_usdc = Decimal("1.0001") ** Decimal(current_tick)
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
                # else:
                #     logger.debug(f"Staked token ID {token_id} is not for WBLT/USDC. Token0: {pos_token0}, Token1: {pos_token1}")

            except Exception as e_pos:
                logger.warning(f"Could not get position details for staked token ID {token_id}: {e_pos}")
                continue # Try next staked token

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
                # else:
                #    logger.debug(f"Wallet token ID {token_id} is not for WBLT/USDC. Token0: {pos_token0}, Token1: {pos_token1}")

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

async def get_pending_aero_rewards(context: CallbackContext, nft_id):
    if not nft_id: return Decimal("0")
    try:
        earned_wei = await asyncio.to_thread(aerodrome_gauge_contract.functions.earned(BOT_WALLET_ADDRESS, nft_id).call)
        aero_decimals = await asyncio.to_thread(aero_token_contract.functions.decimals().call)
        return from_wei(earned_wei, aero_decimals)
    except Exception as e:
        logger.error(f"Error getting pending AERO rewards for NFT {nft_id}: {e}")
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
    try:
        token_in_decimals = await asyncio.to_thread(token_in_contract.functions.decimals().call)
        token_in_symbol = await asyncio.to_thread(token_in_contract.functions.symbol().call)
        amount_in_wei_for_route_and_approval = to_wei(amount_in_decimal, token_in_decimals)

        await send_tg_message(context, f"🏗 Fetching KyberSwap route to sell `{amount_in_decimal}` {token_in_symbol} for token `{token_out_address}`...", menu_type=None)
        route_data = await get_kyberswap_swap_route(token_in_contract.address, token_out_address, amount_in_wei_for_route_and_approval)

        if not route_data or not route_data.get("routeSummary"):
            await send_tg_message(context, f"Could not get swap route from KyberSwap for {token_in_symbol}.", menu_type=None)
            return False, Decimal("0")

        route_summary = route_data["routeSummary"]
        
        await send_tg_message(context, f"🏗 Building KyberSwap transaction data...", menu_type=None)
        swap_build_data = await build_kyberswap_swap_data(route_summary, BOT_WALLET_ADDRESS, BOT_WALLET_ADDRESS, SLIPPAGE_BPS)

        if not swap_build_data or not swap_build_data.get("data"):
            await send_tg_message(context, f"Could not build swap data from KyberSwap for {token_in_symbol}.", menu_type=None)
            return False, Decimal("0")

        # --- DEBUG BLOCK ---
        api_requested_amount_in_wei_str = swap_build_data.get("amountIn", "0")
        api_requested_amount_in_wei = int(api_requested_amount_in_wei_str) # amountIn from API is usually a string

        logger.info(f"KyberSwap Swap Debug: Intended swap amount (for approval): {amount_in_wei_for_route_and_approval} {token_in_symbol}_wei.")
        logger.info(f"KyberSwap Swap Debug: API route's actual amountIn to be used in tx: {api_requested_amount_in_wei} {token_in_symbol}_wei.")

        if api_requested_amount_in_wei > amount_in_wei_for_route_and_approval:
            logger.warning(f"KyberSwap API amountIn ({api_requested_amount_in_wei}) is GREATER than our initial intended/approval amount ({amount_in_wei_for_route_and_approval}). This could cause TRANSFER_FROM_FAILED.")
        elif api_requested_amount_in_wei < amount_in_wei_for_route_and_approval:
            logger.warning(f"KyberSwap API amountIn ({api_requested_amount_in_wei}) is LESS than our initial intended amount ({amount_in_wei_for_route_and_approval}). Swap will use the smaller API amount.")
        # --- END DEBUG BLOCK ---

        actual_router_for_tx = Web3.to_checksum_address(swap_build_data["routerAddress"])

        approved = await approve_token_spending(context, token_in_contract, actual_router_for_tx, amount_in_decimal)
        if not approved:
            await send_tg_message(context, f"Failed to approve {token_in_symbol} for KyberSwap (amount: {amount_in_decimal}).", menu_type=None)
            return False, Decimal("0")

        # swap
        tx_calldata = swap_build_data["data"]
        tx_value = int(swap_build_data.get("value", "0")) 

        swap_tx_params = {
            'to': actual_router_for_tx,
            'from': BOT_WALLET_ADDRESS,
            'value': tx_value,
            'data': tx_calldata,
            'nonce': get_nonce(),
        }
        
        await send_tg_message(context, f"💸 Executing KyberSwap swap (API expects to use `{from_wei(api_requested_amount_in_wei, token_in_decimals):.{token_in_decimals}f}` {token_in_symbol})...", menu_type=None)
        receipt = await asyncio.to_thread(_send_and_wait_for_transaction, swap_tx_params, f"KyberSwap {token_in_symbol} Swap")

        if receipt and receipt.status == 1:
            token_out_decimals = 18 
            if token_out_address == USDC_TOKEN_ADDRESS:
                token_out_decimals = await asyncio.to_thread(usdc_token_contract.functions.decimals().call)
            elif token_out_address == WBLT_TOKEN_ADDRESS:
                 token_out_decimals = await asyncio.to_thread(wblt_token_contract.functions.decimals().call)
            
            amount_out_wei_str = swap_build_data.get("amountOut", "0")
            amount_out_decimal = from_wei(amount_out_wei_str, token_out_decimals)
            
            actual_input_used_by_api = from_wei(api_requested_amount_in_wei, token_in_decimals)
            await send_tg_message(context, f"✅ KyberSwap successful! Swapped ~`{actual_input_used_by_api:.{token_in_decimals}f}` {token_in_symbol} for approx. `{amount_out_decimal:.6f}` of token `{token_out_address}`.", menu_type=None)
            return True, amount_out_decimal
        else:
            await send_tg_message(context, f"KyberSwap swap failed for {token_in_symbol}.", menu_type=None)
            return False, Decimal("0")

    except Exception as e:
        logger.error(f"Error in execute_kyberswap_swap: {e}", exc_info=True)
        await send_tg_message(context, f"Critical error during KyberSwap operation: {e}", menu_type=None)
        return False, Decimal("0")


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


# --- Telegram Command Handlers (probably not necessary, never actually used these commands in prod) ---
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

# --- Telegram Button Handlers ---
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

# --- Action Handler (button_handler) ---
async def handle_status_action(context: CallbackContext):
    if not check_connection():
        await send_tg_message(context, "Web3 not connected. Cannot fetch status.")
        return

    status_lines = ["📊 **Bot Status**"]
    
    # Wallet Balances
    eth_balance_wei = await asyncio.to_thread(w3.eth.get_balance, BOT_WALLET_ADDRESS)
    status_lines.append(f"🔷 ETH Balance: {from_wei(eth_balance_wei, 18):.6f} ETH")
    usdc_bal = await get_token_balance(usdc_token_contract, BOT_WALLET_ADDRESS)
    wblt_bal = await get_token_balance(wblt_token_contract, BOT_WALLET_ADDRESS)
    aero_bal = await get_token_balance(aero_token_contract, BOT_WALLET_ADDRESS)
    status_lines.append(f"💰 USDC Balance: {usdc_bal:.2f}")
    status_lines.append(f"🌯 WBLT Balance: {wblt_bal:.4f}")
    status_lines.append(f"✈️ AERO Balance: {aero_bal:.4f}")
    status_lines.append("---")

    # LP Position
    if bot_state["aerodrome_lp_nft_id"]:
        status_lines.append(f"Aerodrome LP NFT ID: `{bot_state['aerodrome_lp_nft_id']}`")
        position_details = await get_lp_position_details(context, bot_state["aerodrome_lp_nft_id"])
        price_wblt_usdc, current_tick = await get_aerodrome_pool_price_and_tick()

        if position_details and price_wblt_usdc is not None:
            tick_lower_lp = position_details['tickLower']
            tick_upper_lp = position_details['tickUpper']

            # Convert LP range ticks to human-readable prices
            wblt_decimals_val = await asyncio.to_thread(wblt_token_contract.functions.decimals().call)
            usdc_decimals_val = await asyncio.to_thread(usdc_token_contract.functions.decimals().call)
            
            # Price of WBLT in USDC
            decimal_adj_factor_for_price = Decimal(10)**(wblt_decimals_val - usdc_decimals_val)
            
            price_at_tick_lower = (Decimal("1.0001")**Decimal(tick_lower_lp)) * decimal_adj_factor_for_price
            price_at_tick_upper = (Decimal("1.0001")**Decimal(tick_upper_lp)) * decimal_adj_factor_for_price
                        
            status_lines.append(f"  LP Tick Range: `{tick_lower_lp}` to `{tick_upper_lp}`")
            status_lines.append(f"  LP Price Range (WBLT/USDC): `${price_at_tick_lower:.4f}` - `${price_at_tick_upper:.4f}`")
            status_lines.append(f"  Current Pool Tick: `{current_tick}`")
            status_lines.append(f"  Current WBLT Price: `{price_wblt_usdc:.4f} USDC`")

            if current_tick >= tick_lower_lp and current_tick < tick_upper_lp:
                status_lines.append("  Status: ✅ **In Range**")
            else:
                status_lines.append("  Status: ❌ **Out of Range**")
            
            status_lines.append(f"  LP Liquidity (Abstract): `{position_details['liquidity']}`")
            
            # Principal Value
            principal_wblt_usd = bot_state["current_lp_principal_wblt_amount"] * price_wblt_usdc
            total_principal_usd = principal_wblt_usd + bot_state["current_lp_principal_usdc_amount"]
            status_lines.append(f"  Principal in LP: `{bot_state['current_lp_principal_wblt_amount']:.4f} WBLT` & `{bot_state['current_lp_principal_usdc_amount']:.2f} USDC`")
            status_lines.append(f"  Est. Principal Value: `${total_principal_usd:.2f}`")

            pending_aero = await get_pending_aero_rewards(context, bot_state["aerodrome_lp_nft_id"])
            status_lines.append(f"  Pending AERO: `{pending_aero:.4f} AERO`")
        else:
            status_lines.append("  🤷‍♀️ Could not fetch LP position details or pool price.")
    else:
        status_lines.append("No active Aerodrome LP position\.")
    status_lines.append("---")

    status_lines.append(f"Strategy: `{bot_state['current_strategy']}`")
    profit_value_str = f"{bot_state['accumulated_profit_usdc']:.2f}"
    status_lines.append(f"Accumulated Profit (Withdrawable): `${profit_value_str}` USDC")
    status_lines.append(f"Operations Halted: {'YES' if bot_state['operations_halted'] else 'NO'}")
    status_lines.append(f"Action Lock: {'ENGAGED' if bot_state['is_processing_action'] else 'FREE'}")
    status_lines.append(f"Initial Setup Pending: {'YES' if bot_state['initial_setup_pending'] else 'NO'}")

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

    # Check bot's actual USDC balance
    bot_usdc_balance = await asyncio.to_thread(get_token_balance, usdc_token_contract, BOT_WALLET_ADDRESS)
    if bot_usdc_balance < amount_decimal:
        await send_tg_message(context, f"⚠️ Bot's USDC wallet balance ({bot_usdc_balance:.2f}) is less than requested profit withdrawal ({amount_decimal:.2f}). Manual check needed.")
        return

    await send_tg_message(context, f"Attempting to withdraw {amount_decimal:.2f} USDC to {USER_PROFIT_WITHDRAWAL_ADDRESS}...", menu_type=None)
    
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
        await send_tg_message(context, f"✅ Successfully withdrew {amount_decimal:.2f} USDC. Remaining profit: {bot_state['accumulated_profit_usdc']:.2f} USDC.")
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
        
        # Now placeholder. potentially replaced by NFT and gauge queries
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
            await send_tg_message(context, f"Unstaking LP NFT `{original_nft_id}` from gauge...", menu_type=None)
            unstake_tx_params = {'from': BOT_WALLET_ADDRESS, 'nonce': get_nonce()}
            unstake_tx_params['chainId'] = await asyncio.to_thread(lambda: w3.eth.chain_id)
            unstake_tx = aerodrome_gauge_contract.functions.withdraw(original_nft_id).build_transaction(unstake_tx_params)
            unstake_receipt = await asyncio.to_thread(_send_and_wait_for_transaction, unstake_tx, f"Unstake LP NFT {original_nft_id}")
            if not (unstake_receipt and unstake_receipt.status == 1):
                await send_tg_message(context, f"⚠️ Failed to unstake LP NFT {original_nft_id}. Rebalance aborted.", menu_type=None)
                return 
            await send_tg_message(context, f"✅ LP NFT {original_nft_id} unstaked.", menu_type=None)
            bot_state["last_aero_claim_time"] = time.time()
        else:
            logger.info("No existing LP NFT to unstake. Proceeding with available wallet funds.")
            await send_tg_message(context, "No existing LP NFT found. Will use wallet funds for new position.", menu_type=None)

        # 2. Withdraw Liquidity from Aerodrome LP (if an LP was unstaked)
        if original_nft_id:
            position_details = await get_lp_position_details(context, original_nft_id)
            if position_details and position_details['liquidity'] > 0:
                await send_tg_message(context, f"Withdrawing liquidity ({position_details['liquidity']}) from LP NFT {original_nft_id}...", menu_type=None)
                decrease_params = {
                    'tokenId': original_nft_id, 'liquidity': position_details['liquidity'],
                    'amount0Min': 0, 'amount1Min': 0, 'deadline': int(time.time()) + 600 
                }
                decrease_tx_params = {'from': BOT_WALLET_ADDRESS, 'nonce': get_nonce(), 'chainId': await asyncio.to_thread(lambda: w3.eth.chain_id)}
                decrease_tx = aerodrome_nft_manager_contract.functions.decreaseLiquidity(decrease_params).build_transaction(decrease_tx_params)
                decrease_receipt = await asyncio.to_thread(_send_and_wait_for_transaction, decrease_tx, f"Decrease Liquidity NFT {original_nft_id}")

                if decrease_receipt and decrease_receipt.status == 1:
                    collect_params = {
                        'tokenId': original_nft_id, 'recipient': BOT_WALLET_ADDRESS,
                        'amount0Max': 2**128 - 1, 'amount1Max': 2**128 - 1
                    }
                    collect_tx_params = {'from': BOT_WALLET_ADDRESS, 'nonce': get_nonce(), 'chainId': await asyncio.to_thread(lambda: w3.eth.chain_id)}
                    collect_tx = aerodrome_nft_manager_contract.functions.collect(collect_params).build_transaction(collect_tx_params)
                    collect_receipt_obj = await asyncio.to_thread(_send_and_wait_for_transaction, collect_tx, f"Collect Tokens NFT {original_nft_id}")
                    if collect_receipt_obj and collect_receipt_obj.status == 1:
                        await send_tg_message(context, "✅ Liquidity withdrawn and tokens collected.", menu_type=None)
                    else:
                        await send_tg_message(context, "⚠️ Failed to collect tokens. Will rely on wallet balance.", menu_type=None)
                else:
                    await send_tg_message(context, "⚠️ Failed to decrease liquidity. Rebalance aborted.", menu_type=None)
                    return
                
                final_pos_details = await get_lp_position_details(context, original_nft_id)
                if final_pos_details and final_pos_details['liquidity'] == 0:
                    await send_tg_message(context, f"Burning empty LP NFT {original_nft_id}...", menu_type=None)
                    burn_tx_params = {'from': BOT_WALLET_ADDRESS, 'nonce': get_nonce(), 'chainId': await asyncio.to_thread(lambda: w3.eth.chain_id)}
                    burn_tx = aerodrome_nft_manager_contract.functions.burn(original_nft_id).build_transaction(burn_tx_params)
                    await asyncio.to_thread(_send_and_wait_for_transaction, burn_tx, f"Burn LP NFT {original_nft_id}")
            else:
                await send_tg_message(context, f"No active liquidity in LP NFT {original_nft_id} to withdraw.", menu_type=None)
            bot_state["aerodrome_lp_nft_id"] = None

        # 3. Consolidate funds
        available_wblt = await get_token_balance(wblt_token_contract, BOT_WALLET_ADDRESS)
        available_usdc = await get_token_balance(usdc_token_contract, BOT_WALLET_ADDRESS)
        available_aero = await get_token_balance(aero_token_contract, BOT_WALLET_ADDRESS)
        logger.info(f"Consolidated Funds - WBLT: {available_wblt}, USDC: {available_usdc}, AERO: {available_aero}")

        # 4. Sell ALL AERO
        usdc_from_aero_sale = Decimal("0")
        if available_aero > Decimal("0.00001"):
            await send_tg_message(context, f"Selling {available_aero:.6f} AERO for USDC...", menu_type=None)
            swap_success, usdc_received = await execute_kyberswap_swap(context, aero_token_contract, USDC_TOKEN_ADDRESS, available_aero)
            if swap_success:
                usdc_from_aero_sale = usdc_received
                available_usdc += usdc_from_aero_sale
                if bot_state["current_strategy"] == "take_profit":
                    bot_state["accumulated_profit_usdc"] += usdc_from_aero_sale
                    await send_tg_message(context, f"✅ AERO sold. Profit of {usdc_from_aero_sale:.2f} USDC added.", menu_type=None)
                else:
                    await send_tg_message(context, f"✅ AERO sold. {usdc_from_aero_sale:.2f} USDC to be compounded.", menu_type=None)
            else:
                await send_tg_message(context, "⚠️ Failed to sell AERO.", menu_type=None)
        
        # 5. Determine New Optimal LP Range
        price_from_this_pool, pool_current_tick_for_logging = await get_aerodrome_pool_price_and_tick()
        if price_from_this_pool is None:
            await send_tg_message(context, "❌ Cannot get current pool price. Rebalance aborted.", menu_type=None)
            return
        center_price_for_range_calc = price_from_this_pool
        logger.info(f"Centering new LP range around pool price: ${center_price_for_range_calc:.6f} (tick {pool_current_tick_for_logging})")

        tick_spacing = await asyncio.to_thread(aerodrome_pool_contract.functions.tickSpacing().call)
        logger.info(f"Pool {AERODROME_CL_POOL_ADDRESS} uses tickSpacing: {tick_spacing}")

        current_target_range_width = TARGET_RANGE_WIDTH_PERCENTAGE
        logger.info(f"Using target range width: {current_target_range_width}% for this mint attempt.")

        wblt_decimals_val = await asyncio.to_thread(wblt_token_contract.functions.decimals().call)
        usdc_decimals_val = await asyncio.to_thread(usdc_token_contract.functions.decimals().call)

        new_tick_lower, new_tick_upper = await asyncio.to_thread(
            calculate_ticks_for_range,
            center_price_for_range_calc,
            current_target_range_width,
            tick_spacing,
            wblt_decimals_val,
            usdc_decimals_val
        )
        logger.info(f"Calculated new LP ticks using calculate_ticks_for_range: Lower={new_tick_lower}, Upper={new_tick_upper}")
        await send_tg_message(context, f"New target LP range: Ticks `[{new_tick_lower}, {new_tick_upper}]`.", menu_type=None)

        # 6. & 7. Rebalance WBLT/USDC for ~50/50 USD Value
        price_to_use_for_rebalance = center_price_for_range_calc 
        total_value_usd = (available_wblt * price_to_use_for_rebalance) + available_usdc
        if total_value_usd > Decimal("0"):
            target_usdc_value_for_each_token = total_value_usd / Decimal("2")
            if price_to_use_for_rebalance == Decimal(0):
                await send_tg_message(context, "❌ Price for rebalancing is zero. Aborting.", menu_type=None)
                return
            wblt_target_amount = target_usdc_value_for_each_token / price_to_use_for_rebalance
            amount_to_swap = Decimal("0")
            swap_wblt_to_usdc = False
            swap_usdc_to_wblt = False

            if available_wblt > wblt_target_amount: 
                amount_to_swap = available_wblt - wblt_target_amount
                swap_wblt_to_usdc = True
            elif available_wblt < wblt_target_amount: 
                usdc_to_spend_on_wblt = (wblt_target_amount - available_wblt) * price_to_use_for_rebalance
                if usdc_to_spend_on_wblt > Decimal("0") and usdc_to_spend_on_wblt <= available_usdc : 
                    amount_to_swap = usdc_to_spend_on_wblt 
                    swap_usdc_to_wblt = True
            
            if swap_wblt_to_usdc and amount_to_swap > Decimal("0.00001"):
                await send_tg_message(context, f"🌀 Rebalancing: Selling ~**{amount_to_swap:.4f} WBLT**.", menu_type=None)
                success, usdc_gained = await execute_kyberswap_swap(context, wblt_token_contract, USDC_TOKEN_ADDRESS, amount_to_swap)
                if success: available_wblt -= amount_to_swap; available_usdc += usdc_gained
            elif swap_usdc_to_wblt and amount_to_swap > Decimal("0.01"):
                await send_tg_message(context, f"🌀 Rebalancing: Buying WBLT with ~**{amount_to_swap:.2f} USDC**.", menu_type=None)
                success, wblt_gained = await execute_kyberswap_swap(context, usdc_token_contract, WBLT_TOKEN_ADDRESS, amount_to_swap)
                if success: available_usdc -= amount_to_swap; available_wblt += wblt_gained
        logger.info(f"Funds after rebalancing swaps - WBLT: {available_wblt}, USDC: {available_usdc}")

        # 8. Deposit Liquidity into New Range on Aerodrome
        if available_wblt <= Decimal("0.01") and available_usdc <= Decimal("0.01"):
            await send_tg_message(context, "❌ Insufficient WBLT or USDC available to deposit after rebalancing. Aborting mint.", menu_type=None)
            logger.warning("Mint aborted: available WBLT and USDC are zero or dust.")
            return

        await send_tg_message(context, f"🏗 Preparing to mint new LP with up to **{available_wblt:.{wblt_decimals_val}f} WBLT** and **{available_usdc:.{usdc_decimals_val}f} USDC**...", menu_type=None)
        logger.info(f"Preparing to mint. Max Available WBLT: {available_wblt}, Max Available USDC: {available_usdc}")

        # Approvals
        if available_wblt > Decimal("0"):
            approved_wblt = await approve_token_spending(context, wblt_token_contract, AERODROME_SLIPSTREAM_NFT_MANAGER_ADDRESS, available_wblt)
            if approved_wblt: 
                logger.info("WBLT approval successful or already sufficient.")
            else:
                await send_tg_message(context, "❌ Failed to approve WBLT for NFT Manager. Minting aborted.", menu_type=None)
                return
        
        if available_usdc > Decimal("0"):
            approved_usdc = await approve_token_spending(context, usdc_token_contract, AERODROME_SLIPSTREAM_NFT_MANAGER_ADDRESS, available_usdc)
            if approved_usdc:
                logger.info("USDC approval successful or already sufficient.")
            else:
                await send_tg_message(context, "❌ Failed to approve USDC for NFT Manager. Minting aborted.", menu_type=None)
                return
        
        slot0_data = await asyncio.to_thread(aerodrome_pool_contract.functions.slot0().call)
        current_pool_sqrt_price_x96_for_context = slot0_data[0] # For logging or other logic

        # --- Amounts for Mint ---
        # Provide the MAX available amounts. The contract will take what's needed for the range.
        desired_wblt_wei = to_wei(available_wblt, wblt_decimals_val)
        desired_usdc_wei = to_wei(available_usdc, usdc_decimals_val)

        current_slippage_bps_for_min = 200 # Or SLIPPAGE_BPS from your config
        slippage_factor = Decimal(1) - (Decimal(current_slippage_bps_for_min) / Decimal(10000))
        
        amount0_min_wei = 0 # Or 1 if you want to ensure at least a symbolic amount if token0 is used
        amount1_min_wei = 0 # Or 1 if you want to ensure at least a symbolic amount if token1 is used
        logger.info(f"Using amount0Desired={desired_wblt_wei}, amount1Desired={desired_usdc_wei}")
        logger.info(f"REVISED: Using amount0Min={amount0_min_wei}, amount1Min={amount1_min_wei} for mint flexibility.")

        sqrt_price_x96_limit_for_mint = 0
        # current_pool_sqrt_price_x96_for_context is still useful for logging

        mint_params_dict_for_log = {
            "token0": WBLT_TOKEN_ADDRESS, "token1": USDC_TOKEN_ADDRESS,
            "tickSpacing_actual_param": tick_spacing,
            "tickLower": new_tick_lower, "tickUpper": new_tick_upper,
            "amount0Desired": desired_wblt_wei,
            "amount1Desired": desired_usdc_wei,
            "amount0Min": amount0_min_wei,
            "amount1Min": amount1_min_wei,
            "recipient": BOT_WALLET_ADDRESS, "deadline": int(time.time()) + 600,
            "sqrtPriceX96Limit_actual_param": sqrt_price_x96_limit_for_mint,
        }
        logger.info(f"Attempting to mint with contextual params (for logging): {mint_params_dict_for_log}")

        mint_receipt = None
        estimated_gas_for_mint = None
        encoded_mint_data = None

        try:
            if aerodrome_nft_manager_contract is None:
                logger.critical("CRITICAL: aerodrome_nft_manager_contract is None! ABI likely failed to load.")
                await send_tg_message(context, "❌ CRITICAL ERROR: NFT Manager contract not loaded. Aborting mint.", menu_type=None)
                return

            # Construct the 12-element tuple matching the ABI's MintParams struct
            mint_params_as_tuple = (
                Web3.to_checksum_address(WBLT_TOKEN_ADDRESS),
                Web3.to_checksum_address(USDC_TOKEN_ADDRESS),
                int(tick_spacing),
                int(new_tick_lower),
                int(new_tick_upper),
                int(desired_wblt_wei),
                int(desired_usdc_wei),
                int(amount0_min_wei),
                int(amount1_min_wei),
                Web3.to_checksum_address(BOT_WALLET_ADDRESS),
                int(time.time()) + 600,
                int(sqrt_price_x96_limit_for_mint)
            )
            logger.info(f"Attempting to mint with 12-ELEMENT params tuple (for encoding): {mint_params_as_tuple}")

            prepared_mint_function_call = aerodrome_nft_manager_contract.functions.mint(
                mint_params_as_tuple 
            )
            encoded_mint_data = prepared_mint_function_call._encode_transaction_data()

            logger.info(f"Encoded mint data (using 12-element tuple): {encoded_mint_data[:120]}...")

            tx_for_estimation_or_call = {
                'from': BOT_WALLET_ADDRESS,
                'to': AERODROME_SLIPSTREAM_NFT_MANAGER_ADDRESS,
                'data': encoded_mint_data,
            }
            logger.info(f"Transaction dict for eth_call/estimate_gas: {tx_for_estimation_or_call}")

            # --- STATIC CALL TEST (w3.eth.call) ---
            try:
                logger.warning("Attempting static call (w3.eth.call) for mint...")
                call_result_raw = await asyncio.to_thread(w3.eth.call, tx_for_estimation_or_call, 'latest')
                logger.info(f"Static call for mint SUCCEEDED. Result (hex): {call_result_raw.hex() if isinstance(call_result_raw, bytes) else call_result_raw}")
            except ContractLogicError as cle_call:
                logger.error(f"Static call (w3.eth.call) for mint FAILED. Message: {getattr(cle_call, 'message', str(cle_call))} Data: {getattr(cle_call, 'data', 'N/A')}", exc_info=True)
                await send_tg_message(context, f"❌ Minting failed (Static Call Reverted): {getattr(cle_call, 'message', str(cle_call))}", menu_type=None)
                return
            except Exception as e_call:
                logger.error(f"Unexpected error during static call (w3.eth.call) for mint: {e_call}", exc_info=True)
                await send_tg_message(context, f"❌ Minting failed (Static Call Unexpected Error): {e_call}", menu_type=None)
                return
            # --- END STATIC CALL TEST ---

            logger.info(f"Estimating gas for mint with tx_dict: {tx_for_estimation_or_call}")
            estimated_gas_for_mint = await asyncio.to_thread(w3.eth.estimate_gas, tx_for_estimation_or_call)
            logger.info(f"Successfully estimated gas for mint: {estimated_gas_for_mint}")

            final_mint_tx_params_to_send = {
                'from': BOT_WALLET_ADDRESS,
                'to': AERODROME_SLIPSTREAM_NFT_MANAGER_ADDRESS,
                'data': encoded_mint_data,
                'nonce': await asyncio.to_thread(get_nonce),
                'gas': estimated_gas_for_mint,
                'chainId': await asyncio.to_thread(lambda: w3.eth.chain_id)
            }

            logger.info(f"Transaction ready to be sent for mint: {final_mint_tx_params_to_send}")
            mint_receipt = await asyncio.to_thread(_send_and_wait_for_transaction, final_mint_tx_params_to_send, "Mint New Aerodrome LP")

        except AttributeError as ae: 
            logger.error(f"AttributeError during mint prep: {ae}", exc_info=True)
            await send_tg_message(context, f"❌ Minting failed (AttributeError during prep): {ae}", menu_type=None)
            return
        except ContractLogicError as cle: 
            logger.error(f"ContractLogicError during mint prep (estimate_gas likely): Message: {getattr(cle, 'message', str(cle))} Data: {getattr(cle, 'data', 'N/A')}", exc_info=True)
            await send_tg_message(context, f"❌ Minting failed (ContractLogicError during prep): {getattr(cle, 'message', str(cle))}", menu_type=None)
            return 
        except Exception as e:
            logger.error(f"Unexpected error during mint prep: {e}", exc_info=True)
            await send_tg_message(context, f"❌ Minting failed unexpectedly during prep: {e}", menu_type=None)
            return
        
        # --- Check mint_receipt and proceed with event parsing, state update, staking ---
        if not (mint_receipt and mint_receipt.status == 1):
            await send_tg_message(context, "❌ Minting new LP position transaction failed or was not confirmed.", menu_type=None)
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
        await send_tg_message(context, f"✅ New LP position minted! NFT ID: `{new_nft_id}`", menu_type=None)

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
        await send_tg_message(context, f"ℹ️ Staking new LP NFT `{new_nft_id}`...", menu_type=None)
        approved_nft_for_gauge = await approve_nft_for_spending(context, aerodrome_nft_manager_contract, AERODROME_CL_GAUGE_ADDRESS, new_nft_id)
        if not approved_nft_for_gauge:
            await send_tg_message(context, f"❌ Failed to approve NFT {new_nft_id} for staking. Staking aborted.", menu_type=None)
        else:
            await send_tg_message(context, f"✅ NFT {new_nft_id} approved. Staking...", menu_type=None)
            stake_receipt = None 
            deposit_function_call = aerodrome_gauge_contract.functions.deposit(new_nft_id)
            base_stake_tx_params = {'from': BOT_WALLET_ADDRESS, 'chainId': await asyncio.to_thread(lambda: w3.eth.chain_id), 'nonce': get_nonce()}
            try:
                stake_tx_dict = deposit_function_call.build_transaction(base_stake_tx_params)
                stake_receipt = await asyncio.to_thread(_send_and_wait_for_transaction, stake_tx_dict, f"Stake LP NFT {new_nft_id}")
            except Exception as e: logger.error(f"Error during STAKE of NFT {new_nft_id}: {e}", exc_info=True)
            if not (stake_receipt and stake_receipt.status == 1):
                await send_tg_message(context, f"⚠️ Failed to stake LP NFT {new_nft_id}. Remains in wallet.", menu_type=None)
            else:
                await send_tg_message(context, f"✅ LP NFT {new_nft_id} successfully staked!", menu_type=None)

    except Exception as e:
        logger.error(f"Critical error during process_full_rebalance: {e}", exc_info=True)
        await send_tg_message(context, f"⚠️ Critical error during rebalance: {str(e)[:200]}", menu_type=None)
    finally:
        await save_state_async()
        await send_tg_message(context, "🤖 Full Rebalance Process Attempt Finished.")
        await asyncio.sleep(1) 
        await handle_status_action(context)


async def process_claim_sell_aero(context: CallbackContext, triggered_by="auto"):
    if bot_state["operations_halted"] or not bot_state["aerodrome_lp_nft_id"]:
        logger.info("AERO claim/sell skipped: Operations halted or no LP NFT.")
        if bot_state["operations_halted"]: await send_tg_message(context, "AERO claim/sell skipped: Operations halted.", menu_type=None)
        return

    await send_tg_message(context, f"💰 Initiating AERO Claim & Sell (Trigger: `{triggered_by}`)...", menu_type=None)
    
    try:
        # 1. Claim AERO from Gauge
        nft_id_to_claim = bot_state["aerodrome_lp_nft_id"]
        pending_aero_before_claim = await get_pending_aero_rewards(context, nft_id_to_claim)
        
        if pending_aero_before_claim < Decimal("0.1"): # Small threshold, effectively zero, USD 0.06 at current prices
            await send_tg_message(context, "No significant AERO rewards to claim.", menu_type=None)
            bot_state["last_aero_claim_time"] = time.time()
            await save_state_async()
            return

        await send_tg_message(context, f"🛄 Attempting to claim `{pending_aero_before_claim:.6f}` AERO for NFT `{nft_id_to_claim}`...", menu_type=None)
        claim_tx_params = {'from': BOT_WALLET_ADDRESS, 'nonce': get_nonce()}
        claim_tx = aerodrome_gauge_contract.functions.getReward(nft_id_to_claim).build_transaction(claim_tx_params)
        claim_receipt = await asyncio.to_thread(_send_and_wait_for_transaction, claim_tx, f"Claim AERO for NFT {nft_id_to_claim}")

        if not (claim_receipt and claim_receipt.status == 1):
            await send_tg_message(context, "❌ AERO claim transaction failed or not confirmed.", menu_type=None)
            return
        
        await send_tg_message(context, "✅ AERO rewards claimed successfully.", menu_type=None)
        bot_state["last_aero_claim_time"] = time.time()

        # 2. Get bot's AERO balance (should include newly claimed AERO)
        aero_balance = await get_token_balance(aero_token_contract, BOT_WALLET_ADDRESS)
        
        if aero_balance < Decimal("0.01"): # Small threshold, effectively zero, USD 0.06 at current prices
            await send_tg_message(context, "No AERO in wallet to sell after claim attempt.", menu_type=None)
            await save_state_async()
            return

        # 3. Sell ALL AERO for USDC (KyberSwap)
        await send_tg_message(context, f"🤑 Attempting to sell `{aero_balance:.6f}` AERO for USDC via KyberSwap...", menu_type=None)
        swap_success, usdc_received = await execute_kyberswap_swap(context, aero_token_contract, USDC_TOKEN_ADDRESS, aero_balance)

        if swap_success:
            bot_state["accumulated_profit_usdc"] += usdc_received
            await send_tg_message(context, f"✅ Sold `{aero_balance:.6f}` AERO for `{usdc_received:.2f}` USDC. Total profit available: `${bot_state['accumulated_profit_usdc']:.2f}` USDC.")
        else:
            await send_tg_message(context, f"❌ Failed to sell AERO for USDC. AERO remains in bot wallet.", menu_type=None)
            
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
        # Add a small delay to allow the message to send and avoid immediate re-lock if process is quick
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
        await send_tg_message(context, "Unstaking LP NFT from gauge...", menu_type=None)
        unstake_tx_params = { 'from': BOT_WALLET_ADDRESS, 'nonce': get_nonce() }
        unstake_tx = aerodrome_gauge_contract.functions.withdraw(bot_state["aerodrome_lp_nft_id"]).build_transaction(unstake_tx_params)
        unstake_receipt = await asyncio.to_thread(_send_and_wait_for_transaction, unstake_tx, "Emergency Unstake")
        if not unstake_receipt or unstake_receipt.status != 1:
            await send_tg_message(context, "⚠️ Failed to unstake LP NFT. Manual intervention may be required.", menu_type=None)
        else:
            await send_tg_message(context, "✅ LP NFT unstaked.", menu_type=None)

    # 2. Withdraw liquidity if an LP NFT ID exists
    if bot_state["aerodrome_lp_nft_id"]:
        position_details = await get_lp_position_details(context, bot_state["aerodrome_lp_nft_id"])
        if position_details and position_details['liquidity'] > 0:
            await send_tg_message(context, f"Withdrawing liquidity ({position_details['liquidity']}) from LP NFT {bot_state['aerodrome_lp_nft_id']}...", menu_type=None)
            
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
                # Collect Tokens
                collect_params = {
                    'tokenId': bot_state["aerodrome_lp_nft_id"],
                    'recipient': BOT_WALLET_ADDRESS,
                    'amount0Max': 2**128 -1,
                    'amount1Max': 2**128 -1
                }
                collect_tx_params = {'from': BOT_WALLET_ADDRESS, 'nonce': get_nonce()}
                collect_tx = aerodrome_nft_manager_contract.functions.collect(collect_params).build_transaction(collect_tx_params)
                collect_receipt = await asyncio.to_thread(_send_and_wait_for_transaction, collect_tx, "Emergency Collect Tokens")
                if collect_receipt and collect_receipt.status == 1:
                    await send_tg_message(context, "✅ Liquidity withdrawn and tokens collected.", menu_type=None)
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
            if pending_aero_rewards > Decimal("0.000001"): # Small threshold
                await send_tg_message(context, f"Attempting to claim {pending_aero_rewards:.4f} AERO...", menu_type=None)
                claim_tx_params = {'from': BOT_WALLET_ADDRESS, 'nonce': get_nonce()}
                claim_tx = aerodrome_gauge_contract.functions.getReward(bot_state["aerodrome_lp_nft_id"]).build_transaction(claim_tx_params)
                claim_receipt = await asyncio.to_thread(_send_and_wait_for_transaction, claim_tx, "Emergency Claim AERO")
                if claim_receipt and claim_receipt.status == 1:
                     await send_tg_message(context, "✅ AERO claimed.", menu_type=None)
                else:
                     await send_tg_message(context, "⚠️ Failed to claim AERO or no AERO to claim.", menu_type=None)
        except Exception as e:
            logger.warning(f"Could not attempt emergency AERO claim for NFT {bot_state['aerodrome_lp_nft_id']}: {e}")


    # 4. Sell all WBLT for USDC
    wblt_balance = await get_token_balance(wblt_token_contract, BOT_WALLET_ADDRESS)
    if wblt_balance > Decimal("0.01"): # Small threshold
        await send_tg_message(context, f"Selling {wblt_balance:.4f} WBLT for USDC...", menu_type=None)
        success, _ = await execute_kyberswap_swap(context, wblt_token_contract, USDC_TOKEN_ADDRESS, wblt_balance)
        if success: await send_tg_message(context, "✅ WBLT sold for USDC.", menu_type=None)
        else: await send_tg_message(context, "⚠️ Failed to sell WBLT.", menu_type=None)

    # 5. Sell all AERO for USDC
    aero_balance = await get_token_balance(aero_token_contract, BOT_WALLET_ADDRESS)
    if aero_balance > Decimal("0.01"): # Small threshold
        await send_tg_message(context, f"Selling {aero_balance:.4f} AERO for USDC...", menu_type=None)
        success, _ = await execute_kyberswap_swap(context, aero_token_contract, USDC_TOKEN_ADDRESS, aero_balance)
        if success: await send_tg_message(context, "✅ AERO sold for USDC.", menu_type=None)
        else: await send_tg_message(context, "⚠️ Failed to sell AERO.", menu_type=None)

    # Finalize Exit
    bot_state["operations_halted"] = True
    bot_state["aerodrome_lp_nft_id"] = None
    bot_state["current_lp_principal_wblt_amount"] = Decimal("0")
    bot_state["current_lp_principal_usdc_amount"] = Decimal("0")
    bot_state["initial_setup_pending"] = True # Requires setup again to restart

    final_usdc_balance = await get_token_balance(usdc_token_contract, BOT_WALLET_ADDRESS)
    await save_state_async()
    await send_tg_message(context, f"🚨 **EMERGENCY EXIT COMPLETE!** Operations halted. Bot wallet has approx. {final_usdc_balance:.2f} USDC. Please verify all transactions.", menu_type="main")


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

    await send_tg_message(context, f"Attempting to stake discovered NFT ID `{nft_id}`...", menu_type=None)
    
    approved_nft_for_gauge = await approve_nft_for_spending(context, aerodrome_nft_manager_contract, AERODROME_CL_GAUGE_ADDRESS, nft_id)
    if not approved_nft_for_gauge:
        await send_tg_message(context, f"❌ Failed to approve NFT {nft_id} for staking. Staking aborted. Bot remains HALTED.", menu_type="main")
        bot_state["operations_halted"] = True
        return

    await send_tg_message(context, f"✅ NFT {nft_id} approved. Staking...", menu_type=None)

    stake_tx_params = {'from': BOT_WALLET_ADDRESS, 'nonce': await asyncio.to_thread(get_nonce), 'chainId': await asyncio.to_thread(lambda: w3.eth.chain_id)}
    stake_tx = aerodrome_gauge_contract.functions.deposit(nft_id).build_transaction(stake_tx_params)
    stake_receipt = await asyncio.to_thread(_send_and_wait_for_transaction, stake_tx, f"Stake LP NFT {nft_id}")

    if stake_receipt and stake_receipt.status == 1:
        await send_tg_message(context, f"✅ LP NFT {nft_id} successfully STAKED! Resuming normal operations.", menu_type="main")
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

    await send_tg_message(context, f"Attempting to withdraw liquidity from UNSTAKED NFT ID `{nft_id}` and convert assets to USDC...", menu_type=None)
    
    wblt_decimals_val = await asyncio.to_thread(wblt_token_contract.functions.decimals().call)
    # usdc_decimals_val = await asyncio.to_thread(usdc_token_contract.functions.decimals().call) # Not strictly needed here if only converting to USDC

    # --- 1. Withdraw Liquidity from Aerodrome LP (NFT is already in wallet) ---
    position_details = await get_lp_position_details(context, nft_id)

    if position_details and position_details['liquidity'] > 0:
        await send_tg_message(context, f"Withdrawing liquidity ({position_details['liquidity']}) from LP NFT {nft_id}...", menu_type=None)
        
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
            await send_tg_message(context, f"✅ Liquidity decrease successful for NFT {nft_id}. Collecting tokens...", menu_type=None)
            await asyncio.sleep(10) # Give some time for state to settle before collect

            # Collect Tokens
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
            else:
                await send_tg_message(context, f"⚠️ Failed to collect tokens for NFT {nft_id}. Proceeds might be stuck or already claimed. Continuing...", menu_type=None)
        else:
            await send_tg_message(context, f"⚠️ Failed to decrease liquidity for NFT {nft_id}. Manual check advised. Bot HALTED.", menu_type="main")
            bot_state["operations_halted"] = True
            await save_state_async()
            return

        # Burn the now empty NFT
        # Re-check liquidity before burning, just in case.
        final_pos_details_for_burn = await get_lp_position_details(context, nft_id)
        if final_pos_details_for_burn and final_pos_details_for_burn['liquidity'] == 0:
            await send_tg_message(context, f"Burning empty LP NFT {nft_id}...", menu_type=None)
            burn_tx_params = {
                'from': BOT_WALLET_ADDRESS, 
                'nonce': await asyncio.to_thread(get_nonce), 
                'chainId': await asyncio.to_thread(lambda: w3.eth.chain_id)
            }
            burn_tx = aerodrome_nft_manager_contract.functions.burn(nft_id).build_transaction(burn_tx_params)
            burn_receipt = await asyncio.to_thread(_send_and_wait_for_transaction, burn_tx, f"Burn LP NFT {nft_id} (Startup)")
            if burn_receipt and burn_receipt.status == 1:
                await send_tg_message(context, f"✅ NFT {nft_id} burned.", menu_type=None)
            else:
                await send_tg_message(context, f"⚠️ Failed to burn NFT {nft_id}, or it was already gone.", menu_type=None)
        else:
            logger.warning(f"NFT {nft_id} still shows liquidity {final_pos_details_for_burn['liquidity'] if final_pos_details_for_burn else 'unknown'} after collect attempt. Skipping burn.")
            await send_tg_message(context, f"⚠️ NFT {nft_id} still has liquidity after collect attempt or details unavailable. Burn skipped.", menu_type=None)

    elif position_details and position_details['liquidity'] == 0:
        await send_tg_message(context, f"NFT ID `{nft_id}` already has 0 liquidity. Attempting to burn...", menu_type=None)
        burn_tx_params = {
            'from': BOT_WALLET_ADDRESS, 
            'nonce': await asyncio.to_thread(get_nonce), 
            'chainId': await asyncio.to_thread(lambda: w3.eth.chain_id)
        }
        burn_tx = aerodrome_nft_manager_contract.functions.burn(nft_id).build_transaction(burn_tx_params)
        burn_receipt = await asyncio.to_thread(_send_and_wait_for_transaction, burn_tx, f"Burn 0-Liquidity LP NFT {nft_id} (Startup)")
        if burn_receipt and burn_receipt.status == 1:
            await send_tg_message(context, f"✅ NFT {nft_id} (0 liquidity) burned.", menu_type=None)
        else:
            await send_tg_message(context, f"⚠️ Failed to burn 0-liquidity NFT {nft_id}, or it was already gone.", menu_type=None)
    else:
        await send_tg_message(context, f"Could not get position details for NFT {nft_id}, or it doesn't exist. Assuming no liquidity to withdraw from it.", menu_type=None)

    # --- 2. Sell all WBLT for USDC ---
    available_wblt = await get_token_balance(wblt_token_contract, BOT_WALLET_ADDRESS)
    if available_wblt > Decimal("0.01"):
        await send_tg_message(context, f"Selling {available_wblt:.{wblt_decimals_val}f} WBLT for USDC...", menu_type=None)
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

    await asyncio.sleep(5) # Allow RPC to catch up for final balance check
    final_usdc_balance = await get_token_balance(usdc_token_contract, BOT_WALLET_ADDRESS)
    await send_tg_message(
        context,
        f"✅ Liquidity withdrawal process complete. Bot wallet has approx. `{final_usdc_balance:.2f}` USDC. "
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

    await send_tg_message(context, f"✅ Resuming normal monitoring for STAKED LP NFT ID `{nft_id}`.", menu_type="main")
    bot_state["operations_halted"] = False
    bot_state["initial_setup_pending"] = False
    await save_state_async()
    await handle_status_action(context)


async def handle_startup_unstake_and_manage_action(context: CallbackContext):
    nft_id = bot_state.get("aerodrome_lp_nft_id")
    if not nft_id:
        await send_tg_message(context, "⚠️ No staked NFT ID found in state. Please restart bot.", menu_type="main")
        bot_state["operations_halted"] = True
        return

    await send_tg_message(context, f"Attempting to unstake NFT ID `{nft_id}` to allow manual management (rebalance/withdraw)...", menu_type=None)

    # Unstake logic (from emergency_exit or rebalance)
    unstake_tx_params = {'from': BOT_WALLET_ADDRESS, 'nonce': await asyncio.to_thread(get_nonce), 'chainId': await asyncio.to_thread(lambda: w3.eth.chain_id)}
    unstake_tx = aerodrome_gauge_contract.functions.withdraw(nft_id).build_transaction(unstake_tx_params)
    unstake_receipt = await asyncio.to_thread(_send_and_wait_for_transaction, unstake_tx, f"Unstake LP NFT {nft_id} (Startup Choice)")

    if unstake_receipt and unstake_receipt.status == 1:
        await send_tg_message(context, f"✅ LP NFT {nft_id} UNSTAKED. It's now in the bot's wallet. You can now use 'Force Rebalance' or 'Emergency Exit'. Bot remains HALTED.", menu_type="main")
    else:
        await send_tg_message(context, f"⚠️ Failed to unstake LP NFT {nft_id}. Bot remains HALTED. Staked state unchanged.", menu_type="main")

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
                        logger.warning(f"LP NFT {bot_state['aerodrome_lp_nft_id']} has zero or negative tick span. Skipping rebalance check.")
                    else:
                        # 1. Get the pool's tickSpacing
                        pool_tick_spacing = await asyncio.to_thread(aerodrome_pool_contract.functions.tickSpacing().call)
                        if pool_tick_spacing == 0:
                            logger.error("Pool tickSpacing is 0! Cannot calculate buffer. Skipping rebalance check.")
                            bot_state["is_processing_action"] = False
                            await save_state_async()
                            await asyncio.sleep(MAIN_LOOP_INTERVAL_SECONDS) # Or a shorter error retry interval
                            continue


                        # 2. Calculate the raw buffer in ticks based on your percentage
                        raw_buffer_in_ticks = actual_tick_span_lp * (REBALANCE_TRIGGER_BUFFER_PERCENTAGE / Decimal(100))

                        # 3. Convert this raw buffer into a number of full tickSpacing units
                        num_tick_spacings_for_buffer = int(Decimal(raw_buffer_in_ticks) / Decimal(pool_tick_spacing))

                        # 4. Ensure the buffer is at least ONE tickSpacing if the percentage calculation resulted in a non-zero raw buffer but less than one full tickSpacing.
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
                   (    pending_aero > Decimal("0.000001") and time_since_last_claim >= AERO_CLAIM_TIME_THRESHOLD_SECONDS) :
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
        current_approved_address = await asyncio.to_thread(
            nft_contract.functions.getApproved(token_id_to_approve).call
        )

        if current_approved_address != spender_address:
            await send_tg_message(context, f"👍 Approving NFT ID `{token_id_to_approve}` for spender {spender_address}...", menu_type=None)
            
            tx_params = {
                'from': BOT_WALLET_ADDRESS,
                'nonce': get_nonce(),
            }

            approve_tx = nft_contract.functions.approve(spender_address, token_id_to_approve).build_transaction(tx_params)
            receipt = await asyncio.to_thread(_send_and_wait_for_transaction, approve_tx, f"Approve NFT {token_id_to_approve}")
            return receipt is not None and receipt.status == 1
        else:
            logger.info(f"NFT ID {token_id_to_approve} already approved for {spender_address}.")
            return True
    except Exception as e:
        logger.error(f"Error in approve_nft_for_spending for NFT {token_id_to_approve}: {e}")
        await send_tg_message(context, f"Error approving NFT {token_id_to_approve}: {e}", menu_type=None)
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

    # Set initial NFT ID from config if provided and not already in state (this is a fallback)
    if INITIAL_LP_NFT_ID_CONFIG is not None and bot_state.get("aerodrome_lp_nft_id") is None:
        logger.info(f"Using initial LP NFT ID from config: {INITIAL_LP_NFT_ID_CONFIG} (will be verified by discovery)")
        bot_state["aerodrome_lp_nft_id"] = INITIAL_LP_NFT_ID_CONFIG # Tentatively set

    # --- Application and Event Loop ---
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    loop = asyncio.get_event_loop()

    # --- LP State Discovery and User Interaction on Startup ---
    async def startup_sequence(app: Application):
        nonlocal loop # Allow modification of the outer loop variable if needed, though not strictly here
        startup_context = CallbackContext(app)
        await send_tg_message(startup_context, "🤖 Bot instance starting up. Discovering LP state...", menu_type=None)

        discovered_nft_id, discovered_status = await discover_lp_state(startup_context, BOT_WALLET_ADDRESS)

        if discovered_nft_id is not None:
            logger.info(f"Discovered LP NFT ID: {discovered_nft_id}, Status: {discovered_status}")
            bot_state["aerodrome_lp_nft_id"] = discovered_nft_id
            bot_state["initial_setup_pending"] = False

            if discovered_status == "unstaked_in_wallet":
                await send_tg_message(
                    startup_context,
                    f"ℹ️ Discovered UNSTAKED WBLT/USDC LP NFT ID `{discovered_nft_id}` in bot wallet with active liquidity. What would you like to do?",
                    menu_type="startup_unstaked_lp"
                )
            elif discovered_status == "staked":
                await send_tg_message(
                    startup_context,
                    f"ℹ️ Discovered STAKED WBLT/USDC LP NFT ID `{discovered_nft_id}` in the gauge with active liquidity. What would you like to do?",
                    menu_type="startup_staked_lp"
                )
            else:
                logger.error(f"Discovered NFT {discovered_nft_id} but with unknown status: {discovered_status}. Halting.")
                await send_tg_message(startup_context, f"⚠️ Error: Discovered NFT {discovered_nft_id} with unknown status. Manual check needed. Bot HALTED.", menu_type="main")
                bot_state["operations_halted"] = True # Ensure halted

        else:
            logger.info("No active WBLT/USDC LP NFT discovered on-chain for the bot.")
            await send_tg_message(startup_context, "ℹ️ No active WBLT/USDC LP NFT found for the bot.", menu_type=None)
            if bot_state.get("aerodrome_lp_nft_id") is not None:
                logger.warning(
                    f"Saved state had LP ID {bot_state.get('aerodrome_lp_nft_id')} but no active LP found on-chain. "
                    "Resetting saved LP ID."
                )
            bot_state["aerodrome_lp_nft_id"] = None
            bot_state["initial_setup_pending"] = True
            await send_tg_message(startup_context, "Operations are PAUSED. Use menu to start.", menu_type="main")
            if bot_state.get("initial_setup_pending", True):
                 await send_tg_message(startup_context, f"ℹ️ Initial setup needed: No LP NFT found. Send WBLT/USDC to `{BOT_WALLET_ADDRESS}` or set an existing NFT ID via Manage Principal before starting operations.", menu_type="main")

        save_state_sync()

        # Schedule the main bot loop AFTER startup sequence is done or user has made a choice
        # The main loop will only run if operations_halted becomes false.
        loop.create_task(main_bot_loop(app))
        logger.info("Main bot loop scheduled. Startup sequence complete or awaiting user input.")

    # --- Add Handlers ---
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("menu", menu_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CallbackQueryHandler(button_handler))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_message_handler))

    # --- Startup ---
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
