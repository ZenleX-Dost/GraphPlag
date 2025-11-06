# 🚂 Deploying GraphPlag to Railway

Railway is a modern platform that makes it incredibly easy to deploy Python applications. This guide will walk you through deploying GraphPlag API to Railway.

## 📋 Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Deploy (Recommended)](#quick-deploy-recommended)
- [Manual Setup](#manual-setup)
- [Environment Variables](#environment-variables)
- [Monitoring & Logs](#monitoring--logs)
- [Custom Domain](#custom-domain)
- [Troubleshooting](#troubleshooting)

---

## ✅ Prerequisites

1. **GitHub Account** - Your code should be pushed to GitHub
2. **Railway Account** - Sign up at [railway.app](https://railway.app) (free tier available)
3. **Repository Ready** - Ensure all files are committed and pushed

---

## 🚀 Quick Deploy (Recommended)

### Option 1: Deploy Button (Coming Soon)
Click this button to deploy with one click:

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/new/template?template=https://github.com/ZenleX-Dost/GraphPlag)

### Option 2: Deploy from GitHub (Manual)

#### Step 1: Create Railway Project

1. Go to [railway.app](https://railway.app)
2. Click **"New Project"**
3. Select **"Deploy from GitHub repo"**
4. Authorize Railway to access your GitHub
5. Select your **GraphPlag** repository

#### Step 2: Configure Build

Railway will automatically detect the configuration files:
- `railway.json` / `railway.toml` - Deployment config
- `nixpacks.toml` - Build configuration
- `Procfile` - Process definition
- `runtime.txt` - Python version

**The build will automatically:**
- Install Python 3.10
- Install all dependencies from `requirements.txt` and `requirements-api.txt`
- Apply GraKeL patches
- Start the API server

#### Step 3: Wait for Deployment

Railway will:
1. ✅ Build your application (~5-10 minutes first time)
2. ✅ Deploy to a public URL
3. ✅ Run health checks
4. ✅ Show deployment status

#### Step 4: Access Your API

Once deployed, Railway provides:
- **Public URL**: `https://your-app-name.up.railway.app`
- **API Documentation**: `https://your-app-name.up.railway.app/docs`
- **Health Check**: `https://your-app-name.up.railway.app/health`

---

## 🔧 Manual Setup

If automatic detection doesn't work, configure manually:

### 1. Set Build Command

```bash
pip install --upgrade pip && pip install -r requirements.txt && pip install -r requirements-api.txt
```

### 2. Set Start Command

```bash
python -m uvicorn api:app --host 0.0.0.0 --port $PORT
```

Or use the startup script:
```bash
bash start.sh
```

### 3. Configure Healthcheck

- **Path**: `/health`
- **Timeout**: 100 seconds
- **Method**: GET

---

## 🌍 Environment Variables

Railway automatically provides some variables. Add these in the Railway dashboard:

### Required Variables

| Variable | Value | Description |
|----------|-------|-------------|
| `PORT` | Auto-set by Railway | Server port |
| `API_KEY_1` | `your-secure-key` | API authentication key |

### Optional Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_CACHE` | `true` | Enable embedding cache |
| `CACHE_MAX_SIZE_MB` | `500` | Max cache size |
| `CACHE_MAX_AGE_DAYS` | `30` | Cache expiration |
| `MAX_FILE_SIZE_MB` | `50` | Max upload size |
| `ENABLE_CHUNKING` | `true` | Enable large file chunking |
| `MAX_CHUNK_SIZE` | `1000` | Chunk size for large files |
| `DEFAULT_THRESHOLD` | `0.7` | Similarity threshold |
| `LOG_LEVEL` | `INFO` | Logging level |

### Setting Environment Variables

1. Go to your Railway project
2. Click **"Variables"** tab
3. Click **"+ New Variable"**
4. Add key-value pairs
5. Click **"Deploy"** to apply changes

---

## 📊 Monitoring & Logs

### View Logs

1. Open your Railway project
2. Click **"Deployments"** tab
3. Click on the latest deployment
4. View real-time logs

### Common Log Messages

```
🚀 Starting GraphPlag API on Railway...
PORT: 8000
📦 Applying GraKeL patches...
✓ RandomWalk kernel patched
✓ ShortestPath kernel patched
🌐 Starting Uvicorn server...
INFO: Application startup complete.
```

### Metrics Dashboard

Railway provides:
- **CPU Usage**
- **Memory Usage**
- **Network Traffic**
- **Request Count**

Access via: Project → **Metrics** tab

---

## 🔗 Custom Domain

### Add Custom Domain

1. Go to your Railway project
2. Click **"Settings"** tab
3. Scroll to **"Domains"**
4. Click **"+ Add Domain"**
5. Enter your domain (e.g., `api.yourdomain.com`)
6. Add DNS records to your domain provider:
   ```
   Type: CNAME
   Name: api (or @)
   Value: your-app.up.railway.app
   ```
7. Wait for DNS propagation (~5-60 minutes)

### SSL Certificate

Railway automatically provides SSL certificates for custom domains via Let's Encrypt.

---

## 🧪 Testing Your Deployment

### 1. Health Check

```bash
curl https://your-app.up.railway.app/health
```

**Expected Response:**
```json
{
  "status": "healthy",
  "cache_enabled": true,
  "cache_stats": {...}
}
```

### 2. API Documentation

Visit: `https://your-app.up.railway.app/docs`

### 3. Test Comparison

```bash
curl -X POST "https://your-app.up.railway.app/compare/text" \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "text1": "This is a test document.",
    "text2": "This is a test document.",
    "method": "kernel",
    "threshold": 0.7
  }'
```

---

## 🚨 Troubleshooting

### Build Failures

**Problem**: Dependencies fail to install

**Solution**:
```bash
# Check requirements.txt for conflicts
# Ensure torch is compatible with CPU-only Railway environment
```

**Problem**: Out of memory during build

**Solution**:
- Railway free tier has limited build resources
- Consider upgrading to paid tier for more memory
- Reduce model size or dependencies

### Runtime Issues

**Problem**: Application crashes on startup

**Check Logs**:
1. Go to Railway project
2. Check deployment logs
3. Look for error messages

**Common Issues**:

1. **Missing Environment Variables**
   ```
   Error: API_KEY_1 not set
   ```
   **Fix**: Add required environment variables in Railway dashboard

2. **Port Binding Error**
   ```
   Error: Address already in use
   ```
   **Fix**: Ensure start command uses `$PORT` variable

3. **Model Download Issues**
   ```
   Error: Failed to download sentence-transformers model
   ```
   **Fix**: Railway will download models on first run (takes time)
   - First deployment may take 10-15 minutes
   - Subsequent deploys are faster (models cached)

### Performance Issues

**Problem**: Slow response times

**Solutions**:
1. **Enable Caching**:
   ```bash
   ENABLE_CACHE=true
   ```

2. **Upgrade Railway Plan**:
   - Free tier: Shared CPU, 512MB RAM
   - Hobby tier: 1GB RAM, better performance
   - Pro tier: 8GB RAM, dedicated resources

3. **Optimize Models**:
   - Use smaller sentence-transformer models
   - Reduce max chunk sizes

### Connection Timeout

**Problem**: Requests timeout

**Check**:
1. Healthcheck timeout in `railway.json` (set to 100s)
2. First request after idle may be slow (cold start)
3. Large file processing takes time

**Fix**:
```json
{
  "deploy": {
    "healthcheckTimeout": 120
  }
}
```

---

## 💰 Pricing & Limits

### Free Tier (Starter)
- **$5 free credit/month**
- Shared CPU
- 512MB RAM
- 100GB network egress
- Perfect for testing

### Hobby Tier
- **$5/month**
- 1GB RAM
- Better performance
- Good for small projects

### Pro Tier
- **$20/month**
- 8GB RAM
- Dedicated resources
- Production-ready

**Note**: GraphPlag with ML models needs at least **1GB RAM** for reliable operation.

---

## 🔄 Continuous Deployment

Railway automatically redeploys when you push to GitHub:

1. Make changes to your code
2. Commit and push to GitHub:
   ```bash
   git add .
   git commit -m "Update API"
   git push origin main
   ```
3. Railway detects changes and redeploys automatically
4. View deployment progress in Railway dashboard

---

## 📁 Project Structure

Files used by Railway:

```
GraphPlag/
├── railway.json          # Railway deployment config
├── railway.toml          # Alternative config format
├── nixpacks.toml         # Build configuration
├── Procfile              # Process definition
├── runtime.txt           # Python version
├── start.sh              # Startup script with patches
├── requirements.txt      # Core dependencies
├── requirements-api.txt  # API dependencies
└── api.py                # FastAPI application
```

---

## 🎯 Quick Start Commands

### View Status
```bash
# Install Railway CLI (optional)
npm i -g @railway/cli

# Login
railway login

# View status
railway status

# View logs
railway logs

# Open in browser
railway open
```

### Local Testing
```bash
# Test with same environment as Railway
PORT=8000 python -m uvicorn api:app --host 0.0.0.0 --port 8000
```

---

## 🔐 Security Best Practices

1. **Change Default API Keys**:
   ```python
   # In Railway dashboard, set environment variable:
   API_KEY_1=your-very-secure-random-key-here
   ```

2. **Enable HTTPS Only**:
   Railway provides HTTPS by default - ensure clients use it

3. **Rate Limiting**:
   Consider adding rate limiting for production:
   ```python
   # Add to api.py
   from slowapi import Limiter
   limiter = Limiter(key_func=get_remote_address)
   ```

4. **CORS Configuration**:
   Update allowed origins in `api.py`:
   ```python
   app.add_middleware(
       CORSMiddleware,
       allow_origins=["https://yourdomain.com"],  # Restrict in production
       allow_credentials=True,
       allow_methods=["*"],
       allow_headers=["*"],
   )
   ```

---

## 🎉 Success Checklist

- ✅ Repository pushed to GitHub
- ✅ Railway project created
- ✅ Build completed successfully
- ✅ Deployment shows "Active"
- ✅ Health check returns 200 OK
- ✅ API docs accessible at `/docs`
- ✅ Environment variables configured
- ✅ API key changed from default
- ✅ Test comparison works
- ✅ Logs show no errors

---

## 📚 Additional Resources

- **Railway Documentation**: https://docs.railway.app
- **Railway Discord**: https://discord.gg/railway
- **FastAPI Deployment**: https://fastapi.tiangolo.com/deployment/
- **GraphPlag Docs**: See `DEPLOYMENT.md` for full documentation

---

## 🆘 Getting Help

1. **Check Logs**: Railway dashboard → Deployments → Logs
2. **Railway Discord**: Very active and helpful community
3. **GitHub Issues**: Report bugs in GraphPlag repository
4. **Documentation**: Read `DEPLOYMENT.md` for detailed info

---

## 🚀 Next Steps

After successful deployment:

1. **Test All Endpoints**: Use `/docs` for interactive testing
2. **Monitor Performance**: Check Railway metrics
3. **Set Up Alerts**: Configure Railway notifications
4. **Create Frontend**: Build a web interface (see `FEATURES_SUMMARY.md`)
5. **Add Analytics**: Integrate usage tracking
6. **Scale Up**: Upgrade Railway plan if needed

---

**Congratulations! Your GraphPlag API is now live on Railway! 🎉**

Access your API at: `https://your-app.up.railway.app/docs`
