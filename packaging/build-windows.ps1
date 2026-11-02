param(
    [string]$Python = ".\.venv\Scripts\python.exe",
    [switch]$Installer
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Push-Location $repoRoot
try {
    if (-not (Test-Path $Python)) {
        throw "Python environment not found: $Python. Install requirements-build.txt first."
    }

    & $Python -m PyInstaller --clean --noconfirm --distpath dist --workpath build packaging/MWdriver.spec
    if ($LASTEXITCODE -ne 0) {
        throw "PyInstaller failed with exit code $LASTEXITCODE"
    }

    if ($Installer) {
        $compiler = Get-Command ISCC.exe -ErrorAction SilentlyContinue
        $compilerPath = if ($null -ne $compiler) {
            $compiler.Source
        } else {
            Join-Path $env:LOCALAPPDATA "Programs\Inno Setup 6\ISCC.exe"
        }
        if (-not (Test-Path $compilerPath)) {
            throw "Inno Setup is required for the installer. Install it, then rerun with -Installer."
        }
        & $compilerPath packaging/MWdriver.iss
        if ($LASTEXITCODE -ne 0) {
            throw "Inno Setup failed with exit code $LASTEXITCODE"
        }
    }
}
finally {
    Pop-Location
}