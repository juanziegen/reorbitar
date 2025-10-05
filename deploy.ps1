# Quick Deployment Script for Windows
# Usage: .\deploy.ps1 [vercel|netlify|surge]

param(
    [Parameter(Position=0)]
    [ValidateSet('vercel','netlify','surge','build-only')]
    [string]$Platform = 'build-only'
)

Write-Host "🚀 Satellite Debris Removal - Deployment Script" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

# Navigate to frontend
Set-Location frontend

# Check if node_modules exists
if (-not (Test-Path "node_modules")) {
    Write-Host "📦 Installing dependencies..." -ForegroundColor Yellow
    npm install
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Installation failed!" -ForegroundColor Red
        exit 1
    }
}

# Build the project
Write-Host "🔨 Building production bundle..." -ForegroundColor Yellow
npm run build

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Build failed!" -ForegroundColor Red
    exit 1
}

Write-Host "✅ Build successful!" -ForegroundColor Green
Write-Host ""

# Deploy based on platform
switch ($Platform) {
    'vercel' {
        Write-Host "🚀 Deploying to Vercel..." -ForegroundColor Cyan
        Set-Location ..
        vercel --prod
    }
    'netlify' {
        Write-Host "🚀 Deploying to Netlify..." -ForegroundColor Cyan
        netlify deploy --prod --dir=build
    }
    'surge' {
        Write-Host "🚀 Deploying to Surge..." -ForegroundColor Cyan
        Set-Location build
        surge
    }
    'build-only' {
        Write-Host "✅ Build complete! Files are in frontend/build/" -ForegroundColor Green
        Write-Host ""
        Write-Host "To deploy, run:" -ForegroundColor Yellow
        Write-Host "  .\deploy.ps1 vercel   # Deploy to Vercel" -ForegroundColor White
        Write-Host "  .\deploy.ps1 netlify  # Deploy to Netlify" -ForegroundColor White
        Write-Host "  .\deploy.ps1 surge    # Deploy to Surge" -ForegroundColor White
        Write-Host ""
        Write-Host "Or test locally:" -ForegroundColor Yellow
        Write-Host "  cd frontend" -ForegroundColor White
        Write-Host "  npx serve -s build" -ForegroundColor White
    }
}

Write-Host ""
Write-Host "✨ Done!" -ForegroundColor Green
