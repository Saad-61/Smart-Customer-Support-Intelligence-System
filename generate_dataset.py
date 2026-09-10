"""
generate_dataset.py
-------------------
Synthetic Riot Games Customer Support Ticket Dataset Generator.

Produces a realistic, ML-ready CSV with intentional data quality issues
(missing values, duplicates, malformed timestamps, outliers, class imbalance)
so that Module 1 (Data Cleaning) has real problems to solve and document.

Usage:
    python src/generate_dataset.py
    python src/generate_dataset.py --rows 5000 --seed 99 --out data/raw/tickets.csv

Output columns:
    ticket_id, customer_id, ticket_text, product, created_at,
    previous_tickets, resolution_time, priority,
    category, resolved
"""

import argparse
import random
import uuid
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. CONFIGURATION — tweak class distributions here
# ---------------------------------------------------------------------------

PRODUCTS = [
    "League of Legends",
    "Valorant",
    "Teamfight Tactics",
    "Wild Rift",
    "Legends of Runeterra",
]


# (category, weight) — intentionally imbalanced so EDA finds it
CATEGORIES = [
    ("Account Ban / Suspension",   0.22),
    ("Missing RP / Purchase Issue", 0.18),
    ("Cheat / Hacker Report",       0.14),
    ("Client Bug / Crash",          0.12),
    ("Ranked & Matchmaking",        0.11),
    ("Login / 2FA Issue",           0.09),
    ("Refund Request",              0.06),
    ("Chat Restriction Appeal",     0.04),
    ("Champion / Skin Bug",         0.03),
    ("Server Latency / Lag",        0.01),  # rare — intentional
]

# (priority, weight)
PRIORITIES = [
    ("HIGH",   0.20),
    ("MEDIUM", 0.50),
    ("LOW",    0.30),
]

# ---------------------------------------------------------------------------
# 2. TICKET TEXT TEMPLATES  (per category)
# ---------------------------------------------------------------------------

TEMPLATES = {
    "Account Ban / Suspension": [
        "My account {account} was permanently banned but I never cheated. Please review the ban.",
        "I received a 14-day suspension for allegedly using scripts. I have never used any third-party software. Please investigate.",
        "My League account got banned out of nowhere. I was just playing ranked and then I got logged out. Can you explain why?",
        "Hello, my account was flagged and restricted. I believe this is a false positive from your anti-cheat. I need an appeal.",
        "I got permabanned for 'toxic behaviour' but all I did was mute my teammates. This seems unfair.",
        "Account banned with no explanation. I spent over $500 on this account. Please look into this immediately.",
        "My smurf account was banned, now my main account is also restricted. I do not understand what happened.",
        "I was banned for account sharing. I let my brother play once. Is there a way to appeal this decision?",
        "Permanent ban issued on my account. I have 3000 hours on this game and have never violated ToS. Please help.",
        "Got a chat ban that escalated to a full account suspension. The context was taken out of context. I want to appeal.",
    ],
    "Missing RP / Purchase Issue": [
        "I bought 3250 RP but it never showed up in my account. I was charged on my card though.",
        "I purchased the Battle Pass but didn't receive the rewards. Transaction ID: {txn}.",
        "My Riot Points are missing after I topped up. I have the bank receipt. Please refund or credit my account.",
        "I was charged twice for the same RP bundle. I see two transactions on my statement for {amount}.",
        "Bought a bundle in the store, got charged but the content never appeared in my collection.",
        "RP purchase failed but the money was deducted from my account. This is the second time this has happened.",
        "I accidentally purchased the wrong champion. Can I get a refund? I only have one refund token left.",
        "The Lunar New Year bundle charged me but only delivered half the content. Missing the ward skin and icon.",
        "Transaction stuck as 'pending' for 48 hours. My bank confirmed the charge went through. Please help.",
        "Purchased RP via prepaid card, entered the code but nothing was credited. Code: {code}.",
    ],
    "Cheat / Hacker Report": [
        "There is a blatant scripter in my ranked game. They had 0ms reaction time and perfect skill shots all game.",
        "Reporting a player for using wallhacks in Valorant. Their username is {username}. Please review game ID {game_id}.",
        "This player in my last game was clearly using an aimbot. Every bullet was a headshot through smoke.",
        "I keep getting matched against the same scripter. Already reported in-game but they are still playing.",
        "Player using Elo-boosting service. They were clearly being played by someone else based on behaviour change.",
        "Blatant cheater in Diamond ranked. Please investigate account {account}. They were toggling aimbot openly.",
        "Possible griefing bot detected on the enemy team. All five of them stood still in base.",
        "Someone is ddosing our team. Two of my teammates disconnected within seconds of each other.",
        "This player admitted to using scripts in chat. I have a screenshot. How do I submit it?",
        "Cheater ruined 3 games in a row. Your reporting system seems broken. They are still not banned.",
    ],
    "Client Bug / Crash": [
        "The game client crashes every time I enter champion select. I reinstalled but the problem persists.",
        "After the latest patch, my client freezes on the loading screen. I cannot get into any game.",
        "Getting a black screen on launch. The process shows in Task Manager but nothing displays.",
        "The client update bricked my installation. Now I get error code {error_code} on startup.",
        "FPS dropping from 200 to single digits every few minutes since patch {patch}. My hardware is fine.",
        "Audio cuts out completely mid-game after around 20 minutes of play. Rolling back drivers did not fix it.",
        "Client keeps logging me out randomly. I have to re-enter my password multiple times per session.",
        "The shop tab in the client is blank. I cannot browse or purchase anything. Tried clearing cache.",
        "Constant 'Connection Error' messages in client even though my internet is working perfectly.",
        "After updating, all my settings were reset. Also my custom game configurations are gone.",
    ],
    "Ranked & Matchmaking": [
        "I lost LP for a loss that was not my fault. My teammates went AFK in the first 5 minutes.",
        "Matchmaking is putting me against players 3 tiers above me. This has been happening all week.",
        "I was in promo series but the server crashed mid-game. The loss was counted against me.",
        "Lost 40 LP for a game I disconnected from due to power outage. Can this be reversed?",
        "My MMR seems stuck. I win 8 games in a row and gain 18 LP, then lose once and drop 25 LP.",
        "Ranked queue dodging penalty applied even though I did not dodge. I was in loading screen.",
        "I completed my placement matches but my rank is much lower than last season. Is this expected?",
        "Duo partner and I are being put into the wrong queue tier. We are Diamond 2 and 3 but matched in Plat.",
        "Remake did not trigger even though a player was AFK from champion select. Lost LP unfairly.",
        "My clash tournament registration was cancelled without notification. Please refund the ticket fee.",
    ],
    "Login / 2FA Issue": [
        "I cannot log into my account. I reset my password but the 2FA code is going to an old email I lost access to.",
        "Account recovery is not working. I no longer have access to the email or phone number on record.",
        "2FA is sending codes but they expire before I can enter them. Clock sync on my phone is correct.",
        "I bought a new phone and now I cannot receive my 2FA SMS. Please help me recover access.",
        "Login fails with 'incorrect password' even after a password reset. I have cleared cookies and cache.",
        "Locked out of my account after too many failed login attempts. I was trying to recover access legitimately.",
        "Getting error UX-1 when attempting to sign in. This started after your maintenance window yesterday.",
        "I used 'Login with Google' but Google account is now deleted. How do I recover my Riot account?",
        "The authenticator app code is being rejected. I re-scanned the QR code and still getting errors.",
        "Account logged in on an unrecognised device. I changed password but 2FA was not set up. How do I add it now?",
    ],
    "Refund Request": [
        "I would like a refund for the champion I purchased yesterday. I have not played them yet.",
        "Bought the wrong skin. I meant to buy the Prestige version but bought the regular one. Can you swap it?",
        "I purchased a champion that is currently on free rotation. I would like a refund as this was misleading.",
        "My son made an unauthorized purchase on my account. He is 9 years old. Please refund {amount}.",
        "I used my last refund token on an item I now see is in the next sale. Can I have an exception?",
        "Requesting a refund on a bundle because it was not clearly labelled that it included champions I already own.",
        "I was gifted a skin I already owned. The duplicate was auto-converted to orange essence. I want a refund.",
        "I purchased an emote for a character I no longer play. Refund would be appreciated.",
    ],
    "Chat Restriction Appeal": [
        "I received a 25-game chat restriction for a single game where I was defending myself. This is disproportionate.",
        "My chat restriction ended but I am still unable to type in-game. There seems to be a bug.",
        "I was punished for saying 'gg ez' which is a common phrase. I disagree this violates community standards.",
        "Chat restriction applied to my account but I have honour level 5. Seems like an error in your system.",
        "I called out a griefer and got chat banned while they did not. The system feels inconsistent.",
        "Got a communication ban for using my native language which the filter incorrectly flagged.",
        "I want to appeal my chat restriction. The logs being cited were from a conversation I did not start.",
        "3rd chat restriction this year for increasingly minor things. I feel I am being targeted unfairly.",
    ],
    "Champion / Skin Bug": [
        "The new Ahri skin has a visual glitch where her tails disappear during her ultimate animation.",
        "Jinx's passive is not triggering correctly after the latest patch. The movement speed buff is inconsistent.",
        "My Pulsefire Ezreal skin's recall animation is broken. It just shows the default animation instead.",
        "Several ability sound effects are missing for my Prestige Lux skin since the last update.",
        "The Arcane Vi skin causes my game to crash during the level-up animation. Reproducible every time.",
        "Kindred's Q ability is not applying on-hit effects correctly with the Spirit Blossom skin equipped.",
        "My custom Draven skin shows the wrong splash art in the loading screen.",
        "The Dragon Trainer Tristana skin has a missing particle effect on her W ability.",
    ],
    "Server Latency / Lag": [
        "Consistent 300ms ping on EUW since the server migration. My ISP says nothing changed on their end.",
        "Packet loss of around 15 percent during peak hours. Unplayable. Other games are fine.",
        "Getting lag spikes every few minutes that last about 2 seconds. This started after last Tuesday's patch.",
        "The EUW server has been unstable all weekend. Multiple disconnects per gaming session.",
    ],
}

# ---------------------------------------------------------------------------
# 3. HELPER GENERATORS
# ---------------------------------------------------------------------------

def _pick_weighted(items):
    names = [x[0] for x in items]
    weights = [x[1] for x in items]
    return random.choices(names, weights=weights, k=1)[0]


def _random_date(start: datetime, end: datetime) -> datetime:
    delta = end - start
    return start + timedelta(seconds=random.randint(0, int(delta.total_seconds())))


def _resolution_hours(priority: str, resolved: bool) -> float | None:
    if not resolved:
        return None
    base = {"HIGH": 4, "MEDIUM": 24, "LOW": 72}[priority]
    noise = random.expovariate(1 / base)
    return round(noise, 2)


def _fill_template(text: str) -> str:
    text = text.replace("{account}", f"RiotUser#{random.randint(10000, 99999)}")
    text = text.replace("{txn}", f"TXN-{uuid.uuid4().hex[:8].upper()}")
    text = text.replace("{amount}", f"${random.choice([10, 20, 25, 35, 50])}.00")
    text = text.replace("{code}", f"RITO-{uuid.uuid4().hex[:6].upper()}")
    text = text.replace("{username}", f"Player{random.randint(1000, 9999)}")
    text = text.replace("{game_id}", f"NA1_{random.randint(10**9, 10**10)}")
    text = text.replace("{error_code}", f"0x{random.randint(0x1000, 0xFFFF):X}")
    text = text.replace("{patch}", f"{random.randint(13,14)}.{random.randint(1,24)}")
    return text

# ---------------------------------------------------------------------------
# 4. CORE GENERATOR
# ---------------------------------------------------------------------------

def generate_tickets(n: int, seed: int) -> pd.DataFrame:
    random.seed(seed)
    np.random.seed(seed)

    # Simulate ~800 unique customers submitting n tickets
    n_customers = max(50, n // 6)
    customer_pool = [f"CUST-{uuid.uuid4().hex[:8].upper()}" for _ in range(n_customers)]

    # A small subset of customers submit many tickets (power users / chronic complainers)
    heavy_users = random.sample(customer_pool, max(5, n_customers // 20))

    records = []
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2024, 6, 30)

    ticket_counter = 1
    # Track previous ticket count per customer
    customer_ticket_counts: dict[str, int] = {}

    for _ in range(n):
        # Bias heavy users to appear more often
        if random.random() < 0.25:
            cid = random.choice(heavy_users)
        else:
            cid = random.choice(customer_pool)

        customer_ticket_counts[cid] = customer_ticket_counts.get(cid, 0) + 1

        category = _pick_weighted(CATEGORIES)
        priority = _pick_weighted(PRIORITIES)
        product = random.choice(PRODUCTS)
        resolved = random.random() < 0.82  # ~18% unresolved

        template = random.choice(TEMPLATES[category])
        ticket_text = _fill_template(template)
        created_at = _random_date(start_date, end_date)
        resolution_time = _resolution_hours(priority, resolved)

        records.append({
            "ticket_id": f"RGT-{ticket_counter:06d}",
            "customer_id": cid,
            "ticket_text": ticket_text,
            "product": product,
            "created_at": created_at.strftime("%Y-%m-%d %H:%M:%S"),
            "previous_tickets": customer_ticket_counts[cid] - 1,
            "resolution_time": resolution_time,
            "priority": priority,
            "category": category,
            "resolved": resolved,
        })
        ticket_counter += 1

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 5. INJECT REALISTIC DATA QUALITY ISSUES
#    (intentional — Module 1 cleans these and documents each decision)
# ---------------------------------------------------------------------------

def inject_data_quality_issues(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = df.copy()
    n = len(df)

    # --- 5a. Missing values (simulate real-world gaps) ---
    # ~4% of ticket_text missing (customer submitted empty form)
    missing_text_idx = rng.choice(n, size=int(n * 0.04), replace=False)
    df.loc[missing_text_idx, "ticket_text"] = np.nan


    # ~3% of previous_tickets missing
    missing_prev_idx = rng.choice(n, size=int(n * 0.03), replace=False)
    df.loc[missing_prev_idx, "previous_tickets"] = np.nan

    # ~2% of resolution_time missing for resolved tickets (data entry gap)
    resolved_mask = df["resolved"] == True
    resolved_idx = df[resolved_mask].index.tolist()
    missing_res_idx = rng.choice(resolved_idx, size=int(len(resolved_idx) * 0.02), replace=False)
    df.loc[missing_res_idx, "resolution_time"] = np.nan

    # --- 5b. Malformed timestamps (~1.5%) ---
    bad_ts_idx = rng.choice(n, size=int(n * 0.015), replace=False)
    bad_formats = [
        "13/32/2023 10:00:00",   # impossible date
        "2023/99/01 00:00",       # invalid month
        "not-a-date",
        "Jan 2023",               # incomplete
        "",                       # empty string
    ]
    for i in bad_ts_idx:
        df.at[i, "created_at"] = random.choice(bad_formats)

    # --- 5c. Exact duplicate rows (~2%) ---
    n_dupes = int(n * 0.02)
    dupe_source_idx = rng.choice(n, size=n_dupes, replace=False)
    dupe_rows = df.iloc[dupe_source_idx].copy()
    # Give them new ticket IDs but keep everything else identical
    dupe_rows["ticket_id"] = [f"RGT-DUPE-{i:04d}" for i in range(n_dupes)]
    df = pd.concat([df, dupe_rows], ignore_index=True)

    # --- 5d. Outlier resolution times (~0.5%) ---
    n_outliers = max(3, int(n * 0.005))
    outlier_idx = rng.choice(n, size=n_outliers, replace=False)
    for i in outlier_idx:
        if pd.notna(df.at[i, "resolution_time"]):
            df.at[i, "resolution_time"] = random.choice([0.001, 9999.0, -5.0, 50000.0])

    # --- 5e. Negative previous_tickets (~0.3%) ---
    neg_idx = rng.choice(n, size=max(2, int(n * 0.003)), replace=False)
    for i in neg_idx:
        df.at[i, "previous_tickets"] = random.choice([-1, -5, -999])

    # --- 5f. Near-duplicate tickets (semantically same, slightly different wording) ---
    # Inject 15 pairs of near-duplicates — same customer, rephrased complaint
    near_dupe_templates = [
        ("I was charged twice for the same order.",
         "My credit card was billed twice for a single purchase."),
        ("Cannot log into my account after password reset.",
         "Login is failing even though I just reset my password."),
        ("Game crashes on the loading screen every time.",
         "Client freezes during loading screen, crash every game."),
    ]
    pair_counter = 9000
    for text_a, text_b in near_dupe_templates:
        for _ in range(5):
            cid = random.choice(df["customer_id"].tolist())
            for txt in [text_a, text_b]:
                df = pd.concat([df, pd.DataFrame([{
                    "ticket_id": f"RGT-ND-{pair_counter:04d}",
                    "customer_id": cid,
                    "ticket_text": txt,
                    "product": "League of Legends",
                    "created_at": "2023-06-15 14:30:00",
                    "previous_tickets": 2,
                    "resolution_time": 10.0,
                    "priority": "MEDIUM",
                    "category": "Missing RP / Purchase Issue",
                    "resolved": True,
                }])], ignore_index=True)
                pair_counter += 1

    # --- 5g. Shuffle so issues are not at the end ---
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    return df


# ---------------------------------------------------------------------------
# 6. ENTRYPOINT
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate Riot Games support ticket dataset")
    parser.add_argument("--rows",  type=int,   default=3000,                        help="Base ticket count before quality-issue injection (default: 3000)")
    parser.add_argument("--seed",  type=int,   default=42,                          help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--out",   type=str,   default="data/raw/tickets.csv",      help="Output CSV path")
    parser.add_argument("--clean", action="store_true",                             help="Skip data quality injection (produce a clean dataset for comparison)")
    args = parser.parse_args()

    print(f"[generate_dataset] Generating {args.rows} base tickets (seed={args.seed})...")
    df = generate_tickets(args.rows, args.seed)

    if not args.clean:
        print("[generate_dataset] Injecting data quality issues...")
        df = inject_data_quality_issues(df, args.seed)

    df.to_csv(args.out, index=False)

    print(f"[generate_dataset] Saved {len(df)} rows to '{args.out}'")
    print(f"\n--- Dataset Summary ---")
    print(f"  Shape          : {df.shape}")
    print(f"  Columns        : {list(df.columns)}")
    print(f"\n  Category distribution:")
    print(df["category"].value_counts().to_string())
    print(f"\n  Priority distribution:")
    print(df["priority"].value_counts().to_string())
    print(f"\n  Missing values per column:")
    print(df.isnull().sum().to_string())
    print(f"\n  Resolved tickets: {df['resolved'].sum()} / {len(df)}")
    print(f"\n[generate_dataset] Done.")


if __name__ == "__main__":
    main()
