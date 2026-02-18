"""
Stock Picker Pro – Konstanty, tooltips, konfigurace
====================================================
Centrální místo pro všechny neměnné hodnoty, slovníky a konfiguraci.
"""

from typing import Dict, List, Optional

# ============================================================================
# VERZE A ZÁKLADNÍ KONFIGURACE
# ============================================================================
APP_NAME = "Stock Picker Pro"
APP_VERSION = "v9.0"
GEMINI_MODEL = "gemini-2.5-flash-lite"
MAX_AI_RETRIES = 3
RETRY_DELAY = 2

# ============================================================================
# MAKRO KALENDÁŘ (statický – záložní hodnoty; dynamicky rozšiřováno v main)
# ============================================================================
MACRO_CALENDAR: List[Dict] = [
    {"date": "2026-02-20", "event": "FOMC Minutes Release", "importance": "High"},
    {"date": "2026-03-06", "event": "US Employment Report (NFP)", "importance": "High"},
    {"date": "2026-03-11", "event": "US CPI (Inflation Data)", "importance": "High"},
    {"date": "2026-03-18", "event": "FOMC Meeting (Interest Rate Decision)", "importance": "Critical"},
    {"date": "2026-03-25", "event": "US GDP (Q4 2025 Final)", "importance": "Medium"},
    {"date": "2026-04-03", "event": "US Employment Report (NFP)", "importance": "High"},
    {"date": "2026-04-08", "event": "US CPI (Inflation Data)", "importance": "High"},
    {"date": "2026-04-29", "event": "FOMC Meeting (Interest Rate Decision)", "importance": "Critical"},
    {"date": "2026-05-15", "event": "US CPI (Inflation Data)", "importance": "High"},
    {"date": "2026-06-10", "event": "FOMC Meeting (Interest Rate Decision)", "importance": "Critical"},
]

# ============================================================================
# GURU / SOCIÁLNÍ SÍTĚ
# ============================================================================
GURUS: Dict[str, Dict[str, str]] = {
    "CZ/SK Scéna": {
        "Jaroslav Brychta": "JaroslavBrychta",
        "Dominik Stroukal": "stroukal",
        "Jaroslav Šura": "jarsura",
        "Tomáš Plecháč": "TPlechac",
        "Akciový Guru": "akciovyguru",
        "Nicnevim": "Nicnevim11",
        "Bulios": "Bulios_cz",
        "Michal Semotan": "MichalSemotan",
    },
    "Global & News": {
        "Walter Bloomberg (News)": "DeItaone",
        "Brian Feroldi (Education)": "BrianFeroldi",
        "App Economy Insights": "AppEconomyInsights",
    },
}

# ============================================================================
# SEKTOR → PEERS MAPA
# ============================================================================
SECTOR_PEERS: Dict[str, Dict[str, List[str]]] = {
    "Technology": {
        "AAPL": ["MSFT", "GOOGL", "META", "NVDA"],
        "MSFT": ["AAPL", "GOOGL", "META", "AMZN"],
        "GOOGL": ["AAPL", "MSFT", "META", "AMZN"],
        "META": ["AAPL", "GOOGL", "SNAP", "PINS"],
        "NVDA": ["AMD", "INTC", "QCOM", "AVGO"],
        "TSLA": ["RIVN", "LCID", "F", "GM"],
        "NFLX": ["DIS", "PARA", "WBD"],
        "AMD": ["NVDA", "INTC", "QCOM", "AVGO"],
        "INTC": ["NVDA", "AMD", "QCOM", "AVGO"],
        "CRM": ["NOW", "ORCL", "SAP", "WDAY"],
        "ADBE": ["CRM", "NOW", "WDAY", "ORCL"],
        "NOW": ["CRM", "ADBE", "WDAY", "ORCL"],
    },
    "Consumer Cyclical": {
        "AMZN": ["WMT", "TGT", "EBAY", "BABA"],
        "TSLA": ["F", "GM", "RIVN", "LCID"],
        "NKE": ["UA", "ADDYY", "LULU", "SKX"],
    },
    "Healthcare": {
        "JNJ": ["PFE", "UNH", "ABT", "MRK"],
        "PFE": ["JNJ", "MRK", "ABBV", "LLY"],
        "UNH": ["CVS", "HUM", "CI", "CNC"],
        "LLY": ["NVO", "ABBV", "MRK", "PFE"],
    },
    "Financial Services": {
        "JPM": ["BAC", "WFC", "C", "GS"],
        "V": ["MA", "PYPL", "SQ"],
        "MA": ["V", "PYPL", "SQ", "AXP"],
        "KOMB.PR": ["MONETA.PR", "JPM", "BAC"],
        "MONETA.PR": ["KOMB.PR", "JPM", "BAC"],
    },
    "Communication Services": {
        "T": ["VZ", "TMUS"],
        "VZ": ["T", "TMUS"],
        "TMUS": ["T", "VZ"],
    },
    "Utilities": {
        "CEZ.PR": ["NEE", "DUK", "SO", "D"],
        "NEE": ["DUK", "SO", "D", "AEP"],
    },
    "Energy": {
        "XOM": ["CVX", "COP", "EOG", "SLB"],
        "CVX": ["XOM", "COP", "EOG", "BP"],
    },
    "Consumer Defensive": {
        "WMT": ["COST", "TGT", "KR", "AMZN"],
        "KO": ["PEP", "MDLZ", "GIS", "HSY"],
        "PEP": ["KO", "MDLZ", "GIS", "CPB"],
    },
}

# ============================================================================
# TOOLTIP VYSVĚTLIVKY PRO METRIKY
# ============================================================================
METRIC_TOOLTIPS: Dict[str, str] = {
    # Valuace
    "P/E": (
        "Price-to-Earnings: cena akcie děleno zisk na akcii (EPS). "
        "Říká, kolik korun platíš za 1 Kč zisku. "
        "P/E < 15 = levné, > 30 = drahé. Závisí hodně na sektoru."
    ),
    "P/B": (
        "Price-to-Book: cena / účetní hodnota na akcii. "
        "P/B < 1 = firma se obchoduje pod hodnotou svého majetku. "
        "Skvělé pro banky a výrobní firmy."
    ),
    "P/S": (
        "Price-to-Sales: cena / tržby na akcii. "
        "Užitečné pro firmy bez zisku (startupy, SaaS). "
        "P/S < 2 = levné, > 10 = drahé (závisí na sektoru)."
    ),
    "PEG": (
        "PEG Ratio = P/E ÷ roční růst EPS (v %). "
        "Zohledňuje růst. PEG < 1 = potenciálně podhodnoceno, > 2 = drahé vzhledem k růstu. "
        "(Lynch: PEG 1 = férová cena)"
    ),
    "EV/EBITDA": (
        "Enterprise Value / EBITDA: celková hodnota firmy (tržní cap + dluh - cash) "
        "děleno provozní zisk před odpisy. Lepší než P/E pro porovnání firem "
        "s různou dluhovou strukturou. < 10 = levné."
    ),
    "DCF": (
        "Discounted Cash Flow: model, který diskontuje budoucí free cash flow "
        "na současnou hodnotu. Výsledkem je 'férová cena' akcie. "
        "Velmi citlivé na předpoklady (WACC, growth rate)."
    ),
    "MOS": (
        "Margin of Safety: jak velký je 'polštář' mezi férovou cenou (DCF) "
        "a aktuální tržní cenou. MOS > 0 = cena je pod férovkou (příležitost), "
        "MOS < 0 = cena je nad férovkou."
    ),
    "Graham Number": (
        "Konzervativní fair value podle Benjamina Grahama = √(22,5 × EPS × Účetní hodnota/akcii). "
        "Dobré jako dolní mez valuace. Pokud cena < Graham Number = potenciálně levné."
    ),
    # Rentabilita
    "ROE": (
        "Return on Equity: čistý zisk / vlastní kapitál. "
        "Jak efektivně firma zhodnocuje kapitál akcionářů. "
        "ROE > 15 % = skvělé, > 30 % = výjimečné (Buffett benchmark)."
    ),
    "ROA": (
        "Return on Assets: čistý zisk / celková aktiva. "
        "Jak efektivně firma využívá veškerý majetek. "
        "ROA > 5 % = dobré, závisí na kapitálové náročnosti sektoru."
    ),
    "ROIC": (
        "Return on Invested Capital: NOPAT (zisk po daních) / (vlastní kapitál + dluh). "
        "Nejlepší ukazatel ekonomické eficiency. "
        "ROIC > WACC = firma vytváří hodnotu pro akcionáře."
    ),
    "Op. Margin": (
        "Provozní marže: provozní zisk / tržby. "
        "Kolik % z každé koruny tržeb zbyde po zaplacení nákladů (bez daní a úroků). "
        "> 15 % = zdravé, > 30 % = silný byznys model."
    ),
    "Profit Margin": (
        "Čistá marže: čistý zisk / tržby. "
        "Kolik % z tržeb je skutečný zisk po všech nákladech, daních a úrocích. "
        "> 10 % = dobré."
    ),
    "Gross Margin": (
        "Hrubá marže: (tržby - COGS) / tržby. "
        "Kolik zbyde před provozními náklady. "
        "Vysoká hrubá marže (> 50 %) naznačuje silný brand nebo moat (technologie, SW)."
    ),
    # Růst
    "Rev. Growth": (
        "Meziroční růst tržeb. > 10 % = solidní, > 20 % = rychlý růst. "
        "Záporný = varování. "
        "Pozor: high growth + nízká marže = riziková kombinace."
    ),
    "EPS Growth": (
        "Meziroční růst zisku na akcii (EPS). "
        "Důležitější než růst tržeb – říká, jestli firma roste ziskově. "
        "> 10 % = dobré, > 20 % = výborné."
    ),
    # Finanční zdraví
    "Current Ratio": (
        "Current Ratio = oběžná aktiva / krátkodobé závazky. "
        "Schopnost splácet krátkodobé dluhy. "
        "> 1,5 = zdravé, < 1 = možné problémy s likviditou."
    ),
    "Quick Ratio": (
        "Quick Ratio = (oběžná aktiva - zásoby) / krátkodobé závazky. "
        "Konzervativnější verze Current Ratio (bez zásob, které se hůř prodávají). "
        "> 1 = zdravé."
    ),
    "D/E": (
        "Debt-to-Equity: celkový dluh / vlastní kapitál. "
        "Finanční páka. D/E > 2 = vysoká zadluženost (riziko). D/E < 0,5 = konzervativní. "
        "Liší se hodně podle sektoru (utilities mají typicky vysoké D/E)."
    ),
    "Debt/Equity": (
        "Debt-to-Equity: celkový dluh / vlastní kapitál. "
        "Finanční páka. D/E > 2 = vysoká zadluženost (riziko). D/E < 0,5 = konzervativní. "
        "Liší se hodně podle sektoru."
    ),
    "FCF Yield": (
        "Free Cash Flow Yield = FCF / tržní kapitalizace. "
        "Kolik % z tržní hodnoty firmy generuje v hotovosti. "
        "> 5 % = atraktivní. Přesnější než dividend yield pro ocenění firmy."
    ),
    "Net Debt/EBITDA": (
        "Čistý dluh (Debt - Cash) / EBITDA. "
        "Říká, za kolik let provozu firma splatí čistý dluh. "
        "< 1× = konzervativní, 1–3× = zdravé, > 4× = vysoké riziko."
    ),
    "Rule of 40": (
        "Klíčová metrika pro SaaS/tech firmy: Růst tržeb (%) + Provozní marže (%). "
        "Součet > 40 = firma vyváží růst a ziskovost. "
        "< 40 = firma buď roste pomalu NEBO pálí příliš mnoho cash."
    ),
    "FCF Margin": (
        "Free Cash Flow Margin = FCF / tržby. "
        "Kolik korun z každé koruny tržeb se přemění na volnou hotovost. "
        "> 15 % = velmi dobrá cash generace, > 25 % = výjimečná (Apple, Microsoft)."
    ),
    "Dividend Safety": (
        "Dividend Safety Score (0–5): hodnotí udržitelnost dividendy. "
        "Zohledňuje payout ratio (FCF), D/E a historii výplat. "
        "5 = velmi bezpečná, 0 = vysoké riziko snížení dividendy."
    ),
    "Insider Ownership": (
        "Podíl akcií ve vlastnictví insiderů (CEO, CFO, ředitelé, velcí akcionáři). "
        "Vysoký % = silné 'skin in the game' – management má zájem na růstu kurzu. "
        "> 10 % = výrazný zájem insiderů."
    ),
    # Technická
    "RSI": (
        "Relative Strength Index (0–100): měří rychlost a změnu cenových pohybů. "
        "RSI > 70 = překoupeno (možný obrat dolů), RSI < 30 = přeprodáno (možný obrat nahoru). "
        "Neutrální: 40–60."
    ),
    "MACD": (
        "Moving Average Convergence Divergence: rozdíl EMA12 a EMA26. "
        "Když MACD překříží signální linii zdola = bullish signál. Shora = bearish. "
        "Lagging indikátor (reaguje se zpožděním)."
    ),
    "MA50/MA200": (
        "Klouzavé průměry za 50 a 200 dní. "
        "Golden Cross (MA50 > MA200) = bullish trend. Death Cross (MA50 < MA200) = bearish trend. "
        "Cena nad MA200 = long-term uptrend."
    ),
    "BB": (
        "Bollinger Bands: střední pásmo (MA20) ± 2× směrodatná odchylka. "
        "Cena u horního pásma = překoupeno, u dolního = přeprodáno. "
        "'Squeeze' (pásma blízko) = čeká se velký pohyb."
    ),
    # Riziko
    "Piotroski": (
        "Piotroski F-Score (0–9): 9-bodový test fundamentální kvality "
        "(ziskovost, likvidita, efektivita). "
        "8–9 = silná firma, 0–2 = slabá. Dobrý filtr pro value investing."
    ),
    "Altman Z": (
        "Altman Z-Score: model predikce bankrotu. "
        "Z > 2,99 = bezpečná zóna, 1,81–2,99 = šedá zóna, < 1,81 = riziko bankrotu. "
        "Pro průmyslové firmy (ne banky/pojišťovny)."
    ),
    "Short Int.": (
        "Short Interest: % akcií v oběhu, které jsou vypůjčeny a prodány na krátko. "
        "> 10 % = vysoký short zájem (spekulanti sázejí na pokles). "
        "Může být bullish trigger (short squeeze)."
    ),
    "Earnings Q.": (
        "Earnings Quality (CFO / Net Income): poměr provozního cash flow k čistému zisku. "
        "< 0,8 = zisk může být 'papírový' (accruals, účetní triky). "
        "> 1,1 = vynikající – firma vydělává více v cash než reportuje."
    ),
    # Insider
    "Insider Sig.": (
        "Insider Trading Signal: vážený součet nákupů a prodejů insiderů (CEO, CFO, ředitelé) "
        "za posledních 6 měsíců. Zohledňuje roli (CEO = 3×) a hodnotu transakce. "
        "+100 = silný bullish signál."
    ),
    # DCF pokročilé
    "WACC": (
        "Weighted Average Cost of Capital: vážené průměrné náklady kapitálu. "
        "Diskontní sazba v DCF modelu. Čím vyšší WACC, tím nižší fair value. "
        "Zahrnuje cenu dluhu i vlastního kapitálu (CAPM)."
    ),
    "Terminal Growth": (
        "Terminální růst: předpokládaný věčný růst FCF po skončení projekčního období. "
        "Typicky 2–3 % (≈ inflace/GDP). "
        "Velmi citlivý parametr – malá změna = velký dopad na fair value."
    ),
    "Implied Growth": (
        "Reverse DCF: jaký růst FCF trh aktuálně 'očekává' při aktuální ceně akcie. "
        "Pokud je implied growth vyšší než realistický, akcie je pravděpodobně předražená."
    ),
    # Monte Carlo
    "P10/P90": (
        "Percentily Monte Carlo simulace: P10 = pesimistický scénář "
        "(jen 10 % simulací dopadlo hůře), P90 = optimistický (jen 10 % dopadlo lépe). "
        "Medián je robustnější střed než průměr."
    ),
}


def metric_help(key: str) -> Optional[str]:
    """Vrátí tooltip text pro danou metriku nebo None."""
    return METRIC_TOOLTIPS.get(key)
