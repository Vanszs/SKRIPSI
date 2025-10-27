# Rate Limits & Provider Configuration

## 📊 **Supported Providers & Rate Limits**

| Provider | Free Tier Rate Limit | Auto-Detection | Notes |
|----------|---------------------|----------------|-------|
| **Alchemy** | 25 req/s | ✅ Yes | Best untuk production, reliable |
| **Infura** | 10 req/s | ✅ Yes | Populer, stable |
| **Etherscan** | 5 req/s | ✅ Yes | Untuk API calls ke Etherscan |
| **Public RPC** | ~5 req/s | ✅ Yes | Conservative limit, unreliable |

## 🔧 **How Rate Limiting Works**

RPC client menggunakan **sliding window algorithm** untuk enforce rate limits:

```python
# Auto-deteksi provider dari URL
if 'alchemy' in url → 25 req/s
if 'infura' in url → 10 req/s  
if 'etherscan' in url → 5 req/s
else → 5 req/s (safe default)
```

### **Request Tracking**
- Sistem track timestamps dari last N requests (N = rate limit)
- Jika N requests sudah dibuat dalam 1 detik terakhir, tunggu sebelum request berikutnya
- Otomatis sleep dengan buffer 10ms untuk safety

### **Progress Logging**
```
Fetching 1000 blocks (Rate limit: 25 req/s, Provider: alchemy)
Progress: 100/1000 blocks (10.0%) - Rate: 24.3 blocks/s - ETA: 0.6 min
Progress: 200/1000 blocks (20.0%) - Rate: 24.5 blocks/s - ETA: 0.5 min
...
Completed fetching 1000 blocks in 0.68 minutes (avg: 24.5 blocks/s)
```

## ⚙️ **Configuration**

### **Automatic (Recommended)**
Rate limit auto-detected dari RPC URL:

```bash
# .env
SEPOLIA_RPC_URL=https://eth-sepolia.g.alchemy.com/v2/YOUR_KEY
# → Auto-detects: 25 req/s
```

### **Manual Override**
Untuk custom rate limit:

```python
from src.rpc import EthereumRPCClient

# Override dengan custom rate limit
client = EthereumRPCClient(
    network='sepolia',
    rate_limit=30  # Custom: 30 req/s
)
```

## 🚀 **Performance Expectations**

### **Fetching 8000 Blocks**

| Provider | Rate Limit | Time Required |
|----------|-----------|---------------|
| Alchemy | 25 req/s | ~5.3 minutes |
| Infura | 10 req/s | ~13.3 minutes |
| Public RPC | 5 req/s | ~26.7 minutes |

**Formula:**
```
Time = (Total Blocks / Rate Limit) / 60 minutes
```

### **Real Example dengan Alchemy:**
```bash
$ python -m src.fetch --network sepolia --n-blocks 8000

# Output:
Connected to sepolia (Provider: alchemy, Rate Limit: 25 req/s)
Fetching 8000 blocks from 6434567 to 6442567
Progress: 100/8000 blocks (1.2%) - Rate: 24.8 blocks/s - ETA: 5.3 min
Progress: 200/8000 blocks (2.5%) - Rate: 24.9 blocks/s - ETA: 5.2 min
...
Completed fetching 8000 blocks in 5.35 minutes (avg: 24.9 blocks/s)
```

## ⚠️ **Troubleshooting**

### **Rate Limit Exceeded Errors**
```
Error: 429 Too Many Requests
```

**Solusi:**
1. Check apakah rate limit detection benar:
   ```python
   client = EthereumRPCClient()
   print(f"Provider: {client.provider_type}")
   print(f"Rate Limit: {client.rate_limit} req/s")
   ```

2. Manually set lower rate limit:
   ```python
   client = EthereumRPCClient(rate_limit=20)  # 20% safety margin
   ```

### **Slow Fetching Speed**
```
Rate: 5.2 blocks/s (expected: 25 blocks/s)
```

**Kemungkinan penyebab:**
- Network latency tinggi
- RPC endpoint slow response time
- Rate limit detection salah (detected sebagai 'public' instead of 'alchemy')

**Solusi:**
1. Verify RPC URL di `.env`:
   ```bash
   # Pastikan URL mengandung 'alchemy'
   SEPOLIA_RPC_URL=https://eth-sepolia.g.alchemy.com/v2/YOUR_KEY
   ```

2. Test connection speed:
   ```python
   python -c "from src.rpc import EthereumRPCClient; EthereumRPCClient().test_connection()"
   ```

## 📈 **Best Practices**

1. **Use Alchemy untuk production**: Paling reliable dengan 25 req/s
2. **Enable logging**: Set `LOG_LEVEL=DEBUG` untuk detailed rate limit info
3. **Batch operations**: Gunakan `get_block_range()` daripada multiple `get_block()` calls
4. **Monitor progress**: Watch log untuk ensure rate tidak drop significantly
5. **Fallback endpoints**: RPC client auto-switch ke backup endpoint jika primary fail

## 🔍 **Advanced: Custom Rate Limiter**

Untuk use case khusus, extend `EthereumRPCClient`:

```python
from src.rpc import EthereumRPCClient

class CustomRateLimitClient(EthereumRPCClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Custom rate limit logic
        self.burst_size = 50  # Allow burst of 50 requests
        self.burst_window = 2.0  # Within 2 seconds
    
    def _wait_for_rate_limit(self):
        # Custom implementation
        # ... your logic here ...
        super()._wait_for_rate_limit()
```

---

**Last Updated:** October 27, 2025  
**Compatible with:** Alchemy, Infura, Etherscan, Public RPCs
