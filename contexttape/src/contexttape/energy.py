from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

try:
    import psutil  # type: ignore
except Exception:
    psutil = None

# Optional energy measurement backend: pyRAPL (Intel RAPL)
try:
    import pyRAPL  # type: ignore
    _HAS_PYRAPL = True
except Exception:
    _HAS_PYRAPL = False


@dataclass
class EnergySnapshot:
    joules_pkg: Optional[float] = None
    joules_dram: Optional[float] = None
    backend: str = "unavailable"


class EnergyMeter:
    """
    Best-effort energy meter.
    Prefers pyRAPL. Falls back to Linux RAPL files. Else 'unavailable'.
    """
    def __init__(self) -> None:
        self.backend = "unavailable"
        self._session = None
        self._rapl_path = None
        if _HAS_PYRAPL:
            try:
                pyRAPL.setup()
                self.backend = "pyRAPL"
            except Exception:
                self.backend = "unavailable"
        if self.backend == "unavailable":
            p = "/sys/class/powercap/intel-rapl:0/energy_uj"
            if os.path.exists(p):
                self._rapl_path = p
                self.backend = "linux_rapl_file"

    def start(self):
        if self.backend == "pyRAPL":
            self._session = pyRAPL.Measurement("ctx_bench")
            self._session.begin()
        elif self.backend == "linux_rapl_file":
            self._start_uj = self._read_linux_rapl_uj()

    def stop(self) -> EnergySnapshot:
        if self.backend == "pyRAPL" and self._session:
            self._session.end()
            m = self._session.result.pkg  # list per socket
            dram = self._session.result.dram
            pkg_j = float(sum(m)) if m else None
            dram_j = float(sum(dram)) if dram else None
            return EnergySnapshot(joules_pkg=pkg_j, joules_dram=dram_j, backend=self.backend)
        elif self.backend == "linux_rapl_file":
            end = self._read_linux_rapl_uj()
            if hasattr(self, "_start_uj") and self._start_uj is not None and end is not None:
                delta_j = max(0.0, (end - self._start_uj) / 1e6)  # microjoules -> joules
                return EnergySnapshot(joules_pkg=delta_j, joules_dram=None, backend=self.backend)
            return EnergySnapshot(backend=self.backend)
        else:
            return EnergySnapshot(backend=self.backend)

    def _read_linux_rapl_uj(self) -> Optional[int]:
        try:
            with open(self._rapl_path, "r") as f:
                return int(f.read().strip())
        except Exception:
            return None


class EnergyManager:
    """
    Policy for adapting retrieval params based on battery/power/thermal.
    """
    def __init__(self, low_battery_thresh: float = 0.20) -> None:
        self.low_battery_thresh = low_battery_thresh

    def get_battery_level(self) -> Optional[float]:
        if psutil and hasattr(psutil, "sensors_battery"):
            try:
                b = psutil.sensors_battery()
                if b and b.percent is not None:
                    return float(b.percent) / 100.0
            except Exception:
                pass
        return None

    def get_power_usage(self) -> Optional[float]:
        # Placeholder hook for platform-specific power
        return None

    def get_thermal_state(self) -> Optional[str]:
        return None

    def decide_params(self, k: int, max_power_budget: Optional[float] = None) -> dict:
        battery = self.get_battery_level()
        power = self.get_power_usage()
        stride = 1
        quant = "fp32"
        new_k = k

        if battery is not None and battery < self.low_battery_thresh:
            new_k = max(1, k // 2)
            quant = "int8"
        elif (power is not None) and (max_power_budget is not None) and (power > max_power_budget):
            stride = 2

        return {"k": new_k, "stride": stride, "quant_level": quant}
