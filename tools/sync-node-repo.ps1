<#
.SYNOPSIS
  Copia comfyui-reanimator/ al repositorio PUBLICO del nodo y deja el commit hecho.

.DESCRIPTION
  El nodo vive dentro del monorepo privado (junto a la web, Supabase y las
  claves), pero ComfyUI Manager necesita un repo donde el nodo este en la RAIZ:
  el README manda clonarlo dentro de custom_nodes/. Este script es el puente
  entre las dos cosas, para que publicar no signifique copiar carpetas a mano y
  que la version instalada acabe siendo mas vieja que la que se prueba.

  Copia en espejo: lo que ya no esta en el monorepo se borra del repo publico.
  Por eso solo acepta un destino vacio, o uno que ya sea este mismo repo.

  NO hace push. El push lo decide una persona, mirando antes que no se va nada
  que no deba salir.

.EXAMPLE
  .\sync-node-repo.ps1
  .\sync-node-repo.ps1 -Target D:\repos\comfyui-reanimator -Message "LTX 2.3 keyframes"
#>
[CmdletBinding()]
param(
  [string]$Target = "S:\Antigravity\CLAUDE\comfyui-reanimator",
  [string]$Message
)

$ErrorActionPreference = "Stop"
$Source = Split-Path -Parent $PSScriptRoot   # tools\ -> raiz del paquete

if (-not (Test-Path (Join-Path $Source "pyproject.toml"))) {
  throw "No encuentro pyproject.toml en $Source. Este script vive en comfyui-reanimator\tools\."
}

# Guarda del espejo. /MIR borra en el destino, asi que un destino equivocado
# —el escritorio, el monorepo, ComfyUI entero— se lleva ficheros por delante.
# Se aceptan tres casos y ninguno mas: no existe, esta vacio, o ya es este repo.
if (Test-Path $Target) {
  $entries = @(Get-ChildItem -Force $Target)
  $isNodeRepo = (Test-Path (Join-Path $Target ".git")) -and
                (Test-Path (Join-Path $Target "pyproject.toml"))
  if ($entries.Count -gt 0 -and -not $isNodeRepo) {
    throw "$Target no esta vacio y no parece el repo del nodo (falta .git o pyproject.toml). " +
          "Sincronizar ahi borraria lo que haya. Elige otro -Target."
  }
} else {
  New-Item -ItemType Directory -Path $Target | Out-Null
}

if (-not (Test-Path (Join-Path $Target ".git"))) {
  Write-Host "Inicializando repo en $Target"
  git -C $Target init -b main | Out-Null
}

Write-Host "Espejo: $Source  ->  $Target"
# /XD .git: el historial del destino es suyo y no se toca.
robocopy $Source $Target /MIR /XD "__pycache__" ".git" ".venv" /XF "*.pyc" /NFL /NDL /NJH /NJS | Out-Null
if ($LASTEXITCODE -ge 8) { throw "robocopy fallo con codigo $LASTEXITCODE" }

if (-not $Message) {
  $sha = (git -C $Source rev-parse --short HEAD).Trim()
  $Message = "Sync from monorepo $sha"
}

git -C $Target add -A
$staged = git -C $Target diff --cached --name-only
if (-not $staged) {
  Write-Host "Nada que commitear: el repo publico ya esta al dia."
} else {
  git -C $Target commit -m $Message | Out-Null
  Write-Host "Commit hecho: $Message"
  Write-Host ($staged -join "`n")
}

if (-not (git -C $Target remote)) {
  Write-Host ""
  Write-Host "Todavia sin remoto. Cuando el repo exista en GitHub:"
  Write-Host "  git -C `"$Target`" remote add origin https://github.com/unluckyandlucky/comfyui-reanimator"
  Write-Host "  git -C `"$Target`" push -u origin main"
}
