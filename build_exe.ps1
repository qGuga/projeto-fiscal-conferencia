$ErrorActionPreference = "Stop"
$python = (Get-Command python -ErrorAction Stop).Source

& $python -m PyInstaller `
    --noconfirm `
    --clean `
    --onedir `
    --name ConferenciaFiscal `
    --collect-all streamlit `
    --collect-all pandas `
    --collect-all numpy `
    --collect-all pyarrow `
    --collect-all altair `
    --collect-all reportlab `
    --add-data "app.py;." `
    --add-data ".streamlit;.streamlit" `
    launcher.py

Write-Host "Executável criado em: dist\ConferenciaFiscal\ConferenciaFiscal.exe"