import json
import math
import os
import sys
from pathlib import Path

species_index = None


def init_evo_data():
    path = f"{os.getcwd()}/evolution_chains.json"
    global species_index
    with open(path, "r", encoding="utf-8") as f:
        chains = json.load(f)

    species_index = {
        name.lower(): chain for chain in chains for name in chain["species"]
    }


def get_evolution_chain(species_name: str):
    init_evo_data()  # Ensure data is initialized
    if species_index is None:
        raise RuntimeError("Call init_evo_data() first.")

    return species_index.get(species_name.lower())


# --- CP multipliers, official values (GamePress/GOHub) ---
CPM = {
    1: 0.094,
    1.5: 0.1351374318,
    2: 0.16639787,
    2.5: 0.192650919,
    3: 0.21573247,
    3.5: 0.2365726613,
    4: 0.25572005,
    4.5: 0.2735303812,
    5: 0.29024988,
    5.5: 0.3060573775,
    6: 0.3210876,
    6.5: 0.3354450362,
    7: 0.34921268,
    7.5: 0.3624577511,
    8: 0.3752356,
    8.5: 0.387592416,
    9: 0.39956728,
    9.5: 0.4111935514,
    10: 0.4225,
    10.5: 0.4329264091,
    11: 0.44310755,
    11.5: 0.4530599591,
    12: 0.4627984,
    12.5: 0.472336093,
    13: 0.48168495,
    13.5: 0.4908558003,
    14: 0.49985844,
    14.5: 0.508701765,
    15: 0.51739395,
    15.5: 0.5259425113,
    16: 0.5343543,
    16.5: 0.5426357375,
    17: 0.5507927,
    17.5: 0.5588305862,
    18: 0.5667545,
    18.5: 0.5745691333,
    19: 0.5822789,
    19.5: 0.5898879072,
    20: 0.5974,
    20.5: 0.6048236651,
    21: 0.6121573,
    21.5: 0.6194041216,
    22: 0.6265671,
    22.5: 0.6336491432,
    23: 0.64065295,
    23.5: 0.6475809666,
    24: 0.65443563,
    24.5: 0.6612192524,
    25: 0.667934,
    25.5: 0.6745818959,
    26: 0.6811649,
    26.5: 0.6876849038,
    27: 0.69414365,
    27.5: 0.70054287,
    28: 0.7068842,
    28.5: 0.7131691091,
    29: 0.7193991,
    29.5: 0.7255756136,
    30: 0.7317,
    30.5: 0.7347410093,
    31: 0.7377695,
    31.5: 0.7407855938,
    32: 0.74378943,
    32.5: 0.7467812109,
    33: 0.74976104,
    33.5: 0.7527290867,
    34: 0.7556855,
    34.5: 0.7586303683,
    35: 0.76156384,
    35.5: 0.7644860647,
    36: 0.76739717,
    36.5: 0.7702972656,
    37: 0.7731865,
    37.5: 0.7760649616,
    38: 0.77893275,
    38.5: 0.7817900548,
    39: 0.784637,
    39.5: 0.7874736075,
    40: 0.7903,
    # XL levels – good enough approximation for PvP purposes
    40.5: 0.792803968,
    41: 0.79530001,
    41.5: 0.797800015,
    42: 0.8003,
    42.5: 0.802799995,
    43: 0.8053,
    43.5: 0.8078,
    44: 0.81029999,
    44.5: 0.812799985,
    45: 0.81529999,
    45.5: 0.81779999,
    46: 0.82029999,
    46.5: 0.82279999,
    47: 0.82529999,
    47.5: 0.82779999,
    48: 0.83029999,
    48.5: 0.83279999,
    49: 0.83529999,
    49.5: 0.83779999,
    50: 0.84029999,
    50.5: 0.84279999,
    51: 0.84529999,
}


def load_pokedex():
    path = f"{os.getcwd()}/pokedex.json"
    # path = "pokedex.json"
    p = Path(path)
    if not p.exists():
        raise SystemExit("pokedex.json not found – run build_pokedex.py first")
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def best_level_for_ivs(base, ivs, cp_cap, min_level=1, max_level=51):
    atk_b, def_b, sta_b = base
    iv_a, iv_d, iv_s = ivs
    best = None

    steps = int((max_level - min_level) * 2) + 1
    for step in range(steps + 1):
        level = min_level + step * 0.5
        if level not in CPM:
            continue
        cpm = CPM[level]

        atk = (atk_b + iv_a) * cpm
        deff = (def_b + iv_d) * cpm
        hpp = (sta_b + iv_s) * cpm
        hp = math.floor((sta_b + iv_s) * cpm)

        if hp < 10:
            hp = 10

        cp = max(10, math.floor(atk * math.sqrt(deff) * math.sqrt(hpp) / 10))

        if cp <= cp_cap:
            best = (level, cp, atk, deff, hp)

    return best  # None if it can never fit under cp_cap


def rank_all_ivs(base, cp_cap, min_level=1, max_level=51):
    rows = []
    for iv_a in range(16):
        for iv_d in range(16):
            for iv_s in range(16):
                best = best_level_for_ivs(
                    base, (iv_a, iv_d, iv_s), cp_cap, min_level, max_level
                )
                if not best:
                    continue
                level, cp, atk, deff, hp = best
                stat_prod = atk * deff * hp
                rows.append(
                    {
                        "atk_iv": iv_a,
                        "def_iv": iv_d,
                        "sta_iv": iv_s,
                        "level": level,
                        "cp": cp,
                        "atk": atk,
                        "def": deff,
                        "hp": hp,
                        "stat_prod": stat_prod,
                    }
                )

    rows.sort(key=lambda r: r["stat_prod"], reverse=True)
    if not rows:
        return []

    best_sp = rows[0]["stat_prod"]
    for idx, row in enumerate(rows, start=1):
        row["rank"] = idx
        row["perfect_pct"] = 100.0 * row["stat_prod"] / best_sp

    return rows


def get_iv(name, ivs, league="great"):
    atk_iv, def_iv, sta_iv = ivs

    if league.startswith("g"):
        cp_cap = 1500
    elif league.startswith("u"):
        cp_cap = 2500
    elif league.startswith("m"):
        cp_cap = 10000  # effectively no cap
    else:
        raise SystemExit("league must be great / ultra / master")

    pokedex = load_pokedex()
    if name not in pokedex:
        print(f"Pokemon '{name}' not found in pokedex.json")
        print("Example key formats:")
        for k in list(pokedex.keys())[:10]:
            print(" ", k)
        sys.exit(1)

    base = pokedex[name]["atk"], pokedex[name]["def"], pokedex[name]["sta"]
    rows = rank_all_ivs(base, cp_cap)

    target = None
    for row in rows:
        if (row["atk_iv"], row["def_iv"], row["sta_iv"]) == (atk_iv, def_iv, sta_iv):
            target = row
            break

    if not target:
        print("These IVs never fit under the CP cap for this league.")
        sys.exit(0)

    return target["rank"]


def get_all_ranks(name, ivs):
    ev_chain = get_evolution_chain(name.lower())

    final_string = ""
    for pokemon_name in ev_chain["species"]:
        rank = get_iv(pokemon_name.capitalize(), ivs)
        final_string += f"{pokemon_name[:5]}: {rank} - "
    final_string = final_string.rstrip(" - ")
    return final_string


if __name__ == "__main__":
    print(get_all_ranks("Marowak", [1, 15, 15]))
