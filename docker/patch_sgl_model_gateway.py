#!/usr/bin/env python3
from pathlib import Path
import sys


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    if old in text:
        path.write_text(text.replace(old, new, 1), encoding="utf-8")
        return
    if new in text:
        return
    raise SystemExit(f"Did not find expected snippet in {path}")


def main() -> None:
    root = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path("/sgl-workspace/sglang/sgl-model-gateway")
    )

    replace_once(
        root / "src/app_context.rs",
        """                WasmModuleManager::new(WasmRuntimeConfig::default())
                    .map_err(|e| format!("Failed to initialize WASM module manager: {}", e))?,
""",
        """                WasmModuleManager::new(WasmRuntimeConfig::default()),
""",
    )

    replace_once(
        root / "src/core/steps/wasm_module_registration.rs",
        "                wasm_bytes,\n",
        "                wasm_bytes: wasm_bytes.into(),\n",
    )


if __name__ == "__main__":
    main()
