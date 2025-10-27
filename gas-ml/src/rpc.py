"""
RPC Client untuk koneksi ke Ethereum Sepolia Testnet.

Modul ini menyediakan abstraksi untuk:
- Koneksi ke node Ethereum via HTTP/WebSocket
- Fetching data blok dengan retry mechanism
- Error handling dan logging
- Support untuk multiple RPC endpoints (Infura, Alchemy, public nodes)
- Rate limiting untuk menghindari API throttling
"""

import logging
import time
from typing import Optional, Dict, Any, List
from collections import deque
from web3 import Web3
from web3.exceptions import BlockNotFound, TimeExhausted
from web3.types import BlockData
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EthereumRPCClient:
    """
    Client untuk berinteraksi dengan Ethereum node.
    
    Mendukung:
    - Sepolia testnet
    - Retry mechanism untuk koneksi yang tidak stabil
    - Multiple fallback RPC endpoints
    - Rate limiting untuk Alchemy (25 req/s) dan Etherscan (5 req/s)
    - Caching untuk mengurangi API calls
    """
    
    # Rate limits untuk different providers
    RATE_LIMITS = {
        'alchemy': 25,      # 25 requests per second
        'etherscan': 5,     # 5 requests per second
        'infura': 10,       # 10 requests per second (free tier)
        'public': 5,        # Conservative untuk public endpoints
    }
    
    # Default RPC endpoints untuk Sepolia
    SEPOLIA_ENDPOINTS = [
        os.getenv('SEPOLIA_RPC_URL', 'https://rpc.sepolia.org'),
        'https://ethereum-sepolia.publicnode.com',
        'https://rpc2.sepolia.org',
        'https://sepolia.gateway.tenderly.co',
    ]
    
    def __init__(
        self,
        network: str = 'sepolia',
        rpc_url: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        timeout: int = 30,
        rate_limit: Optional[int] = None
    ):
        """
        Inisialisasi RPC client.
        
        Args:
            network: Network name ('sepolia', 'mainnet', dll)
            rpc_url: Custom RPC URL (optional)
            max_retries: Maximum retry attempts untuk failed requests
            retry_delay: Delay antar retry (seconds)
            timeout: Request timeout (seconds)
            rate_limit: Custom rate limit (req/s). Auto-detect jika None
        """
        self.network = network
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout
        
        # Setup RPC endpoints
        self.endpoints = self._setup_endpoints(rpc_url)
        self.current_endpoint_idx = 0
        
        # Detect provider dan setup rate limiting
        self.provider_type = self._detect_provider(self.endpoints[0])
        self.rate_limit = rate_limit or self.RATE_LIMITS.get(self.provider_type, 10)
        self.min_interval = 1.0 / self.rate_limit  # Minimum interval antar requests
        
        # Request tracking untuk rate limiting
        self.request_times = deque(maxlen=self.rate_limit)
        
        # Initialize Web3 connection
        self.w3 = self._connect()
        
        # Verify connection
        if not self.w3.is_connected():
            raise ConnectionError(f"Failed to connect to {network} network")
        
        logger.info(
            f"Connected to {network} - Chain ID: {self.w3.eth.chain_id} "
            f"(Provider: {self.provider_type}, Rate Limit: {self.rate_limit} req/s)"
        )
    
    def _detect_provider(self, endpoint: str) -> str:
        """
        Deteksi provider type dari RPC URL.
        
        Args:
            endpoint: RPC endpoint URL
            
        Returns:
            Provider type ('alchemy', 'etherscan', 'infura', 'public')
        """
        endpoint_lower = endpoint.lower()
        
        if 'alchemy' in endpoint_lower:
            return 'alchemy'
        elif 'etherscan' in endpoint_lower:
            return 'etherscan'
        elif 'infura' in endpoint_lower:
            return 'infura'
        else:
            return 'public'
    
    def _wait_for_rate_limit(self):
        """
        Enforce rate limiting dengan sliding window.
        
        Tunggu jika perlu untuk menghindari exceeding rate limit.
        """
        now = time.time()
        
        # Jika kita sudah mencapai limit, tunggu
        if len(self.request_times) >= self.rate_limit:
            oldest_request = self.request_times[0]
            time_since_oldest = now - oldest_request
            
            # Jika oldest request masih dalam 1 detik, tunggu
            if time_since_oldest < 1.0:
                sleep_time = 1.0 - time_since_oldest + 0.01  # +10ms buffer
                logger.debug(f"Rate limit: sleeping {sleep_time:.3f}s")
                time.sleep(sleep_time)
                now = time.time()
        
        # Record request time
        self.request_times.append(now)
    
    def _setup_endpoints(self, custom_url: Optional[str]) -> List[str]:
        """Setup list of RPC endpoints dengan fallback."""
        if custom_url:
            return [custom_url] + self.SEPOLIA_ENDPOINTS
        return self.SEPOLIA_ENDPOINTS.copy()
    
    def _connect(self) -> Web3:
        """Establish connection ke Ethereum node dengan fallback."""
        for idx, endpoint in enumerate(self.endpoints):
            try:
                logger.info(f"Trying to connect to: {endpoint}")
                w3 = Web3(Web3.HTTPProvider(
                    endpoint,
                    request_kwargs={'timeout': self.timeout}
                ))
                
                if w3.is_connected():
                    self.current_endpoint_idx = idx
                    logger.info(f"Successfully connected to: {endpoint}")
                    return w3
                    
            except Exception as e:
                logger.warning(f"Failed to connect to {endpoint}: {e}")
                continue
        
        raise ConnectionError("Failed to connect to any RPC endpoint")
    
    def _retry_with_fallback(self, func, *args, **kwargs):
        """
        Execute function dengan retry dan endpoint fallback.
        
        Args:
            func: Function to execute
            *args, **kwargs: Function arguments
            
        Returns:
            Function result
            
        Raises:
            Exception: Jika semua retry attempts gagal
        """
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                # Enforce rate limiting sebelum request
                self._wait_for_rate_limit()
                
                return func(*args, **kwargs)
                
            except Exception as e:
                last_exception = e
                logger.warning(
                    f"Attempt {attempt + 1}/{self.max_retries} failed: {e}"
                )
                
                # Try fallback endpoint jika available
                if attempt < self.max_retries - 1:
                    self.current_endpoint_idx = (self.current_endpoint_idx + 1) % len(self.endpoints)
                    try:
                        self.w3 = self._connect()
                        # Re-detect provider type untuk new endpoint
                        self.provider_type = self._detect_provider(
                            self.endpoints[self.current_endpoint_idx]
                        )
                        self.rate_limit = self.RATE_LIMITS.get(self.provider_type, 10)
                        self.min_interval = 1.0 / self.rate_limit
                        logger.info(f"Switched to {self.provider_type} (rate: {self.rate_limit} req/s)")
                    except:
                        pass
                    
                    time.sleep(self.retry_delay * (attempt + 1))
        
        raise last_exception
    
    def get_latest_block_number(self) -> int:
        """
        Get latest block number dari network.
        
        Returns:
            Latest block number
        """
        def _get_block_number():
            return self.w3.eth.block_number
        
        return self._retry_with_fallback(_get_block_number)
    
    def get_block(
        self,
        block_identifier: int | str,
        full_transactions: bool = False
    ) -> BlockData:
        """
        Get block data by number atau hash.
        
        Args:
            block_identifier: Block number (int) atau hash (str)
            full_transactions: Include full transaction objects
            
        Returns:
            Block data dictionary
            
        Raises:
            BlockNotFound: Jika block tidak ditemukan
        """
        def _get_block():
            return self.w3.eth.get_block(block_identifier, full_transactions)
        
        return self._retry_with_fallback(_get_block)
    
    def get_block_range(
        self,
        start_block: int,
        end_block: int,
        step: int = 1
    ) -> List[BlockData]:
        """
        Get range of blocks.
        
        Args:
            start_block: Starting block number
            end_block: Ending block number (inclusive)
            step: Step size untuk fetching blocks
            
        Returns:
            List of block data
        """
        blocks = []
        total_blocks = (end_block - start_block + 1) // step
        
        logger.info(
            f"Fetching {total_blocks} blocks from {start_block} to {end_block} "
            f"(Rate limit: {self.rate_limit} req/s, Provider: {self.provider_type})"
        )
        
        start_time = time.time()
        
        for block_num in range(start_block, end_block + 1, step):
            try:
                block = self.get_block(block_num)
                blocks.append(block)
                
                # Log progress setiap 100 blocks
                if len(blocks) % 100 == 0:
                    elapsed = time.time() - start_time
                    rate = len(blocks) / elapsed if elapsed > 0 else 0
                    eta = (total_blocks - len(blocks)) / rate if rate > 0 else 0
                    logger.info(
                        f"Progress: {len(blocks)}/{total_blocks} blocks "
                        f"({len(blocks)/total_blocks*100:.1f}%) - "
                        f"Rate: {rate:.2f} blocks/s - "
                        f"ETA: {eta/60:.1f} min"
                    )
                    
            except BlockNotFound:
                logger.warning(f"Block {block_num} not found, skipping...")
                continue
            except Exception as e:
                logger.error(f"Error fetching block {block_num}: {e}")
                raise
        
        elapsed = time.time() - start_time
        logger.info(
            f"Completed fetching {len(blocks)} blocks in {elapsed/60:.2f} minutes "
            f"(avg: {len(blocks)/elapsed:.2f} blocks/s)"
        )
        
        return blocks
    
    def get_latest_blocks(self, n: int) -> List[BlockData]:
        """
        Get n latest blocks dari network.
        
        Args:
            n: Number of blocks to fetch
            
        Returns:
            List of block data (newest first)
        """
        latest = self.get_latest_block_number()
        start_block = max(0, latest - n + 1)
        
        logger.info(f"Fetching blocks {start_block} to {latest}")
        blocks = self.get_block_range(start_block, latest)
        
        # Return newest first
        return list(reversed(blocks))
    
    def get_gas_price(self) -> int:
        """
        Get current gas price (in Wei).
        
        Returns:
            Current gas price
        """
        def _get_gas_price():
            return self.w3.eth.gas_price
        
        return self._retry_with_fallback(_get_gas_price)
    
    def wei_to_gwei(self, wei: int) -> float:
        """
        Convert Wei to Gwei.
        
        Args:
            wei: Amount in Wei
            
        Returns:
            Amount in Gwei (float)
        """
        return self.w3.from_wei(wei, 'gwei')
    
    def gwei_to_wei(self, gwei: float) -> int:
        """
        Convert Gwei to Wei.
        
        Args:
            gwei: Amount in Gwei
            
        Returns:
            Amount in Wei (int)
        """
        return self.w3.to_wei(gwei, 'gwei')
    
    def is_syncing(self) -> bool | Dict[str, Any]:
        """
        Check if node is currently syncing.
        
        Returns:
            False if fully synced, or dict with sync info
        """
        return self.w3.eth.syncing
    
    def get_chain_id(self) -> int:
        """Get chain ID dari connected network."""
        return self.w3.eth.chain_id
    
    def extract_block_features(self, block: BlockData) -> Dict[str, Any]:
        """
        Extract features dari block data untuk ML model.
        
        Args:
            block: Block data from Web3
            
        Returns:
            Dictionary dengan features yang dibutuhkan
        """
        # Handle None baseFeePerGas (pre-EIP-1559 blocks)
        base_fee = block.get('baseFeePerGas', 0)
        
        return {
            'number': block['number'],
            'timestamp': block['timestamp'],
            'baseFeePerGas': base_fee,
            'gasUsed': block['gasUsed'],
            'gasLimit': block['gasLimit'],
            'txCount': len(block.get('transactions', [])),
            'difficulty': block.get('difficulty', 0),
            'totalDifficulty': block.get('totalDifficulty', 0),
        }
    
    def __repr__(self) -> str:
        """String representation of client."""
        return f"EthereumRPCClient(network={self.network}, chain_id={self.get_chain_id()})"


def test_connection():
    """Test RPC connection dan print info."""
    try:
        client = EthereumRPCClient(network='sepolia')
        
        print(f"\n{'='*60}")
        print(f"Connection Test Results")
        print(f"{'='*60}")
        print(f"Network: {client.network}")
        print(f"Chain ID: {client.get_chain_id()}")
        print(f"RPC Endpoint: {client.endpoints[client.current_endpoint_idx]}")
        print(f"Latest Block: {client.get_latest_block_number()}")
        print(f"Gas Price: {client.wei_to_gwei(client.get_gas_price()):.2f} Gwei")
        
        # Test fetch single block
        latest_block = client.get_block('latest')
        print(f"\nLatest Block Info:")
        print(f"  Number: {latest_block['number']}")
        print(f"  Timestamp: {latest_block['timestamp']}")
        print(f"  BaseFee: {client.wei_to_gwei(latest_block.get('baseFeePerGas', 0)):.4f} Gwei")
        print(f"  Gas Used: {latest_block['gasUsed']:,}")
        print(f"  Gas Limit: {latest_block['gasLimit']:,}")
        print(f"  Transactions: {len(latest_block['transactions'])}")
        
        print(f"\n{'='*60}")
        print("✓ Connection test successful!")
        print(f"{'='*60}\n")
        
        return client
        
    except Exception as e:
        print(f"\n✗ Connection test failed: {e}\n")
        raise


if __name__ == "__main__":
    # Run connection test
    client = test_connection()
