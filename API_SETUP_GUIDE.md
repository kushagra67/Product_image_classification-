# API Keys Setup Guide

## Getting Free API Keys for Image Downloads

### 1. Unsplash API Key (Recommended)

**Steps:**
1. Go to: https://unsplash.com/developers
2. Click "Register as a developer"
3. Sign up or log in with your account
4. Click "Your apps" in the dashboard
5. Click "Create a new application"
6. Accept the terms and create the app
7. Copy your **Access Key** (starts with a long string of characters)

**Example API Key format:**
```
Eb_QQ8_VaM7k-7XxYJ8sQ_Qa...
```

### 2. Pexels API Key

**Steps:**
1. Go to: https://www.pexels.com/api/
2. Scroll down to "Request API"
3. Fill out the form with your details
4. Click "Request API key"
5. Check your email for the API key
6. Copy the **API key** provided

**Example API Key format:**
```
563492ad6f917000010000011234567890...
```

---

## How to Configure the Downloader

### Option A: Using Environment Variables (Recommended)

**Windows (PowerShell):**
```powershell
$env:UNSPLASH_API_KEY = "your_unsplash_key_here"
$env:PEXELS_API_KEY = "your_pexels_key_here"
python download_images.py
```

**Windows (Command Prompt):**
```cmd
set UNSPLASH_API_KEY=your_unsplash_key_here
set PEXELS_API_KEY=your_pexels_key_here
python download_images.py
```

### Option B: Edit the Script Directly

**Edit `download_images.py`:**

```python
# Line 19-20, replace:
UNSPLASH_API_KEY = "YOUR_UNSPLASH_API_KEY_HERE"
PEXELS_API_KEY = "YOUR_PEXELS_API_KEY_HERE"

# With your actual keys:
UNSPLASH_API_KEY = "Eb_QQ8_VaM7k-7XxYJ8sQ_Qa..."
PEXELS_API_KEY = "563492ad6f917000010000011234567890..."
```

Then run:
```powershell
python download_images.py
```

---

## Free Tier Limits

| Service | Free Tier Limit | Per Hour |
|---------|-----------------|----------|
| **Unsplash** | 50 requests/hour | Generous |
| **Pexels** | 200 requests/hour | Very generous |
| **Direct URLs** | Unlimited | Always available |

---

## Expected Download Results

With both API keys configured, you should get:
- **8-15 images per category** from Unsplash
- **8-15 images per category** from Pexels
- **8 images per category** from Direct URLs
- **Total: ~80-120+ new images** across all categories

---

## Troubleshooting

**Issue:** "No results found"
- Solution: API key may be incorrect or inactive. Check the API dashboard.

**Issue:** "Request limit exceeded"
- Solution: Wait an hour or use multiple API keys.

**Issue:** "Network error"
- Solution: Check internet connection. Script will retry with direct URLs.

---

## Next Steps After Downloading

1. **Review downloaded images:**
   ```powershell
   Get-ChildItem data/raw/books/
   Get-ChildItem data/raw/clothing/
   Get-ChildItem data/raw/electronics/
   Get-ChildItem data/raw/furniture/
   ```

2. **Delete low-quality images** (if any)

3. **Retrain the model:**
   ```powershell
   python train.py
   ```

4. **Test inference:**
   ```powershell
   python inference.py --image-dir data/raw
   ```

---

## Questions?

For API issues:
- Unsplash: https://unsplash.com/documentation
- Pexels: https://www.pexels.com/api/documentation/
