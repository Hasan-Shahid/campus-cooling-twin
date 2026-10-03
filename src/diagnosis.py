"""Rule-based cause labels for flagged anomalies -- hints from the shape of
the deviation, not confirmed diagnoses. Mirrors the reference demo's TYPES
dict, adapted to what our three detection rules can actually tell us."""

TYPES = {
    "hvac_no_occupancy": {
        "name": "HVAC running, room looks empty",
        "why": "The HVAC unit is on, no motion has been detected in the room for 2 hours, and CO2 has not risen more than 20 ppm above the room's own baseline in that time.",
        "check": "Check whether the room is scheduled/booked for this time, and whether the HVAC has a manual override or missed an auto-off schedule. Caveat: motion sensors undercount real occupancy (per the dataset's own documentation) and the CO2 threshold is not calibrated against ground truth, so treat this as a lead, not a confirmed fault.",
    },
    "setpoint_gap": {
        "name": "Indoor temp far from setpoint while HVAC is on",
        "why": "The room temperature differs from the HVAC setpoint by more than 3 C while the unit is actively running.",
        "check": "Check for a stuck/undersized HVAC unit, a door or window left open, or a setpoint that was changed but not reflected in behavior.",
    },
    "temp_residual": {
        "name": "Unexpected temperature (model residual)",
        "why": "The next-step indoor temperature deviated from what Model A predicted given the room's recent temperature, weather, and HVAC state -- a pattern the model hasn't seen before.",
        "check": "Check for a sensor fault, an unrecorded HVAC manual override, or an external heat/cold source not captured by the model's inputs.",
    },
    "energy_spike": {
        "name": "Energy spike (model residual)",
        "why": "Block-level energy use deviated sharply from what Model B predicted given weather, average indoor temp, and HVAC activity.",
        "check": "Check for equipment left running outside its usual schedule, a faulty meter reading, or an HVAC unit cycling inefficiently.",
    },
}

# Priority order when multiple room-level rules fire on the same row -- most
# actionable/specific first.
PRIORITY = ["setpoint_gap", "hvac_no_occupancy", "temp_residual"]


def classify_room_row(row) -> str | None:
    for key in PRIORITY:
        col = f"anomaly_{key}"
        if col in row and bool(row[col]):
            return key
    return None
