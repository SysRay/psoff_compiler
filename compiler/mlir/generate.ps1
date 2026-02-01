$directory = "generated"
$mlir_src = "F:\Source\llvm\llvm-project\mlir\include"
$mlir_bin = "F:\Source\llvm\llvm-project\buildM\bin"

$tempDir = "templates"
$compFiles = Get-ChildItem -Path $tempDir -Filter *.td

$baseFileName = "psOffGpu.td"
$inputFile = "$tempDir/$baseFileName"

$outputHeader = Join-Path $directory "$baseFileName.h.inc"
$outputSource = Join-Path $directory "$baseFileName.cpp.inc"

$attrHeader = Join-Path $directory "$baseFileName.attr.h.inc"
$attrSource = Join-Path $directory "$baseFileName.attr.cpp.inc"

$opHeader = Join-Path $directory "$baseFileName.op.h.inc"
$opSource = Join-Path $directory "$baseFileName.op.cpp.inc"

$enumHeader = Join-Path $directory "$baseFileName.enum.h.inc"
$enumSource = Join-Path $directory "$baseFileName.enum.cpp.inc"

$rewriteHeader = Join-Path $directory "$baseFileName.rewrite.h.inc"
$passHeader = Join-Path $directory "$baseFileName.pass.h.inc"


try {
    Write-Host "Generating $($baseFileName)..."
    & $mlir_bin\mlir-tblgen -gen-dialect-decls $inputFile -o $outputHeader -I $mlir_src -I $tempDir

    # Check if compilation was successful
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Successfully compiled $($baseFileName) to $outputHeader" -ForegroundColor Green
    } else {
        Write-Host "Compilation failed for $($baseFileName)" -ForegroundColor Red
    }

    & $mlir_bin\mlir-tblgen -gen-attrdef-decls $inputFile -o $attrHeader -I $mlir_src -I $tempDir
    & $mlir_bin\mlir-tblgen -gen-attrdef-defs $inputFile -o $attrSource -I $mlir_src -I $tempDir

    & $mlir_bin\mlir-tblgen -gen-enum-decls $inputFile -o $enumHeader -I $mlir_src -I $tempDir
    & $mlir_bin\mlir-tblgen -gen-enum-defs $inputFile -o $enumSource -I $mlir_src -I $tempDir

    & $mlir_bin\mlir-tblgen -gen-op-decls $inputFile -o $opHeader -I $mlir_src -I $tempDir
    & $mlir_bin\mlir-tblgen -gen-op-defs $inputFile -o $opSource -I $mlir_src -I $tempDir

    & $mlir_bin\mlir-tblgen -gen-dialect-defs $inputFile -o $outputSource -I $mlir_src -I $tempDir

    & $mlir_bin\mlir-tblgen -gen-rewriters $inputFile -o $rewriteHeader -I $mlir_src -I $tempDir
    & $mlir_bin\mlir-tblgen -gen-pass-decls $inputFile -o $passHeader -I $mlir_src -I $tempDir
}
catch {
    Write-Host "Error compiling $($baseFileName): $_" -ForegroundColor Red
}

Write-Host "Compilation process complete." -ForegroundColor Cyan