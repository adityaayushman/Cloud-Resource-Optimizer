"""Instance catalogue, provider pricing and the multi-cloud selection layer.

Pricing is *deterministic*: the spot component is derived from a hash of the
provider name and a 5-minute time bucket rather than a fresh random draw, plus a
smooth diurnal term. The same timestamp therefore always yields the same price,
and prices drift continuously rather than jumping - which is what makes the
arbitrage recommendation stable across a page refresh and reproducible in the
evaluation harness. (Prices within one bucket are near-identical but not bitwise
equal, because the diurnal term is continuous in time.)
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

from .models import CloudProvider, InstanceType, Region

# --------------------------------------------------------------------------
# Instance sizing
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class InstanceSpec:
    cpu: float
    ram: float
    base_cost_per_hour: float
    energy_efficiency: float
    max_power_watts: float


# RAM:CPU ratios - small/medium/large are 2.0, memory is 4.0.
INSTANCE_SPECS: dict[InstanceType, InstanceSpec] = {
    # Smaller instances are more power-efficient per core but cost more per core.
    InstanceType.SMALL: InstanceSpec(2, 4, 0.050, 1.20, 50),
    InstanceType.MEDIUM: InstanceSpec(4, 8, 0.100, 1.00, 100),
    InstanceType.LARGE: InstanceSpec(8, 16, 0.200, 0.80, 200),
    # Memory-optimised: same cores as MEDIUM, double the RAM, ~30% dearer.
    InstanceType.MEMORY: InstanceSpec(4, 16, 0.130, 1.00, 110),
    # Compute-dense: better price and power per core, as real catalogues offer.
    InstanceType.XLARGE: InstanceSpec(16, 32, 0.380, 0.70, 360),
}


# --------------------------------------------------------------------------
# Providers
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ProviderSpec:
    base_multiplier: float   # baseline on-demand price relative to AWS
    base_latency_ms: float   # median round-trip for the primary region
    volatility: float        # amplitude of the spot-price band
    reliability: float       # modelled availability (fraction)
    pue: float               # power usage effectiveness of the provider's estate


# The three criteria are deliberately independent, each favouring a different
# provider: cost favours GCP, latency favours AWS, carbon favours GCP via PUE.
# An earlier version derived the carbon term from `reliability`, which is not a
# carbon quantity at all - it made the carbon weight a proxy for availability
# and left the carbon slider with no distinguishable effect.
PROVIDER_SPECS: dict[CloudProvider, ProviderSpec] = {
    CloudProvider.AWS:   ProviderSpec(1.00, 38.0, 0.18, 0.9995, 1.14),
    CloudProvider.AZURE: ProviderSpec(0.95, 45.0, 0.10, 0.9990, 1.12),
    CloudProvider.GCP:   ProviderSpec(0.90, 52.0, 0.06, 0.9992, 1.10),
}


@dataclass(frozen=True)
class RegionSpec:
    cost_multiplier: float
    latency_offset_ms: float
    carbon_intensity: float  # kg CO2 per kWh


REGION_SPECS: dict[Region, RegionSpec] = {
    Region.US_EAST:    RegionSpec(1.00, 0.0, 0.40),
    Region.EU_WEST:    RegionSpec(1.05, 12.0, 0.30),
    Region.ASIA_SOUTH: RegionSpec(1.20, 28.0, 0.70),
}

PRICE_BUCKET_SECONDS = 300  # spot prices re-quote every 5 minutes


def _deterministic_unit(seed: str) -> float:
    """Stable pseudo-random value in [0, 1) derived from a string seed."""
    digest = hashlib.sha256(seed.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def price_index(provider: CloudProvider, at: float) -> float:
    """Current price multiplier for a provider at simulation/wall time `at`.

    The result is the provider's baseline multiplier perturbed by a smooth,
    deterministic spot component inside +/- `volatility`.
    """
    spec = PROVIDER_SPECS[provider]
    bucket = int(at // PRICE_BUCKET_SECONDS)
    u = _deterministic_unit(f"{provider.value}:{bucket}")
    # Blend a smooth diurnal term with the bucket noise so the series looks
    # like a real spot market rather than white noise.
    diurnal = math.sin(2 * math.pi * (at % 86400) / 86400.0)
    swing = spec.volatility * (0.6 * (2 * u - 1) + 0.4 * diurnal)
    return round(spec.base_multiplier * (1.0 + swing), 6)


def hourly_cost(
    instance: InstanceType,
    provider: CloudProvider,
    region: Region,
    at: float,
) -> float:
    """On-demand hourly price for one instance."""
    spec = INSTANCE_SPECS[instance]
    return (
        spec.base_cost_per_hour
        * price_index(provider, at)
        * REGION_SPECS[region].cost_multiplier
    )


def latency_ms(provider: CloudProvider, region: Region) -> float:
    return PROVIDER_SPECS[provider].base_latency_ms + REGION_SPECS[region].latency_offset_ms


def carbon_kg_per_kwh(region: Region) -> float:
    """Grid carbon intensity for a region, before provider efficiency."""
    return REGION_SPECS[region].carbon_intensity


def effective_carbon(provider: CloudProvider, region: Region) -> float:
    """kg CO2 per kWh of *useful* compute: grid intensity scaled by PUE.

    PUE is the ratio of total facility power to IT power, so a provider with a
    lower PUE emits less for the same delivered compute in the same region.
    """
    return REGION_SPECS[region].carbon_intensity * PROVIDER_SPECS[provider].pue


# --------------------------------------------------------------------------
# Multi-cloud selection (US-08, US-09, US-18)
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class SelectionWeights:
    cost: float = 0.55
    latency: float = 0.25
    carbon: float = 0.20

    def normalised(self) -> "SelectionWeights":
        total = self.cost + self.latency + self.carbon
        if total <= 0:
            return SelectionWeights()
        return SelectionWeights(self.cost / total, self.latency / total, self.carbon / total)


def _min_max(values: list[float]) -> tuple[float, float]:
    lo, hi = min(values), max(values)
    return lo, (hi if hi > lo else lo + 1e-9)


def score_providers(
    at: float,
    instance: InstanceType = InstanceType.MEDIUM,
    region: Region = Region.US_EAST,
    weights: SelectionWeights | None = None,
) -> list[dict]:
    """Score every provider on cost, latency and carbon.

    Each criterion is min-max normalised across the candidate set so the three
    incommensurable units can be combined; lower score is better.
    """
    w = (weights or SelectionWeights()).normalised()
    candidates = list(PROVIDER_SPECS)

    costs = [hourly_cost(instance, p, region, at) for p in candidates]
    lats = [latency_ms(p, region) for p in candidates]
    carbons = [effective_carbon(p, region) for p in candidates]

    c_lo, c_hi = _min_max(costs)
    l_lo, l_hi = _min_max(lats)
    g_lo, g_hi = _min_max(carbons)

    rows = []
    for provider, cost, lat, carbon in zip(candidates, costs, lats, carbons):
        n_cost = (cost - c_lo) / (c_hi - c_lo)
        n_lat = (lat - l_lo) / (l_hi - l_lo)
        n_carbon = (carbon - g_lo) / (g_hi - g_lo)
        score = w.cost * n_cost + w.latency * n_lat + w.carbon * n_carbon
        rows.append(
            {
                "provider": provider.value,
                "price_index": price_index(provider, at),
                "hourly_cost": round(cost, 5),
                "latency_ms": round(lat, 1),
                "carbon_kg_per_kwh": round(carbon, 4),
                "grid_intensity": round(carbon_kg_per_kwh(region), 3),
                "pue": PROVIDER_SPECS[provider].pue,
                "reliability": PROVIDER_SPECS[provider].reliability,
                "score": round(score, 5),
            }
        )

    rows.sort(key=lambda r: r["score"])
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


def select_provider(
    at: float,
    instance: InstanceType = InstanceType.MEDIUM,
    region: Region = Region.US_EAST,
    weights: SelectionWeights | None = None,
) -> tuple[CloudProvider, list[dict]]:
    """Return the best provider for a new instance plus the full scoreboard."""
    rows = score_providers(at, instance, region, weights)
    return CloudProvider(rows[0]["provider"]), rows
