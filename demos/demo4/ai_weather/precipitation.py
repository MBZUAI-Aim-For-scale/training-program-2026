"""Convert rainfall amounts using source metadata rather than value magnitudes."""

import warnings


def precipitation_to_mm(data, unit_cvt=None):
    """Convert declared rainfall amounts to mm; require a fallback if unlabeled.

    Source units take precedence over catalog multipliers. Rate units require
    time integration and are rejected. The returned mm metadata makes repeated
    conversion safe, and the input array is never modified in place.
    """
    import logging
    import math

    units = str(data.attrs.get("units") or "").strip().casefold()
    normalized = units.replace(" ", "").replace("**", "^").replace("²", "2")
    factors = {
        "m": 1000.0, "meter": 1000.0, "meters": 1000.0,
        "metre": 1000.0, "metres": 1000.0,
        "mm": 1.0, "millimeter": 1.0, "millimeters": 1.0,
        "millimetre": 1.0, "millimetres": 1.0,
        "cm": 10.0, "centimeter": 10.0, "centimeters": 10.0,
        "centimetre": 10.0, "centimetres": 10.0,
        "kgm-2": 1.0, "kgm^-2": 1.0, "kg/m2": 1.0, "kg/m^2": 1.0,
    }
    if normalized in factors:
        factor = factors[normalized]
    elif units:
        raise ValueError(f"Unsupported precipitation units {units!r}. Supply rainfall amounts in m, cm, mm or kg m-2; rates require time integration.")
    else:
        if unit_cvt is None:
            raise ValueError("Precipitation units are missing. Set source units or an explicit dataset multiplier to mm.")
        factor = float(unit_cvt)
        if not math.isfinite(factor) or factor <= 0:
            raise ValueError("The fallback precipitation multiplier must be finite and positive.")
        if factor == 1.0:
            logging.getLogger("ai_weather.precipitation").info(
                "Precipitation units are missing; assuming mm using explicit configured multiplier 1.0 (values unchanged).")
        else:
            warnings.warn(f"Precipitation units are missing; using configured multiplier {factor}. Confirm source units before interpreting scores.", RuntimeWarning)

    converted = data * factor
    converted.attrs = dict(data.attrs)
    converted.attrs["units"] = "mm"
    return converted
