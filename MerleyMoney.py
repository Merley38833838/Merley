import time
import sqlite3
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from datetime import datetime, timedelta
import requests

# === CONFIGURATION ===
# Twelvedata
API_KEY_TWELVEDATA = "81cd251769144de699bedcbb446c2a67"
BASE_URL_TWELVEDATA = "https://api.twelvedata.com/time_series"

# Tiingo
API_KEY_TIINGO = "9b007d540a0df37b811a30cbb5e64ce1cd5c4e90"
BASE_URL_TIINGO = "https://api.tiingo.com/tiingo/fx"

# Telegram
BOT_TOKEN = "8155863530:AAEI94bQL4Z-7O55uP6D8Bsuu-RYga-6T-w"
CHANNEL_ID = "-1002261216001"

# Paires
PAIRS_TWELVEDATA = ["EUR/USD", "USD/JPY"]
PAIR_TIINGO = "XAU/USD"
PAIRS_DB = ["EURUSD", "USDJPY", "XAUUSD"]

# Autres
DB_NAME = "forex_trading.db"
UPDATE_INTERVAL = 180  # 3 minutes en secondes

# === FONCTIONS UTILITAIRES ===
def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    data = {"chat_id": CHANNEL_ID, "text": message}
    try:
        response = requests.post(url, data=data)
        response.raise_for_status()
        print("Message Telegram envoyé avec succès.")
    except requests.exceptions.RequestException as e:
        print(f"Erreur lors de l'envoi du message Telegram : {e}")

def format_signal_message(pair, signal, price, sl, tp, timestamp):
    signal_type = "BUY" if "buy" in signal.lower() else "SELL"
    timestamp_str = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")

    message = f"""
FOREX🚨

📈  {pair}
➡️  {signal_type} at {price:.5f}

💰 TP : {tp:.5f}
🗿SL : {sl:.5f}

🕒 : {timestamp_str}
"""
    return message

def fetch_tiingo_data(symbol):
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(minutes=120 * 5)  # 120 candles of 5min

    url = f"{BASE_URL_TIINGO}/{symbol.replace('/', '')}/prices"
    headers = {"Content-Type": "application/json", "Authorization": f"Token {API_KEY_TIINGO}"}
    params = {
        "resampleFreq": "5min",
        "startDate": start_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "endDate": end_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "columns": "open,high,low,close"
    }

    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()

        if not data:
            print(f"Aucune donnée Tiingo reçue pour {symbol}.")
            return []

        ohlcv = []
        for candle in data:
            timestamp = datetime.strptime(candle["date"], "%Y-%m-%dT%H:%M:%S.%fZ").timestamp()
            ohlcv.append((
                timestamp,
                candle["open"],
                candle["high"],
                candle["low"],
                candle["close"],
                0  # Volume non fourni par Tiingo, on met 0
            ))
        print(f"Données Tiingo récupérées pour {symbol}: {len(ohlcv)} bougies")
        return ohlcv
    except requests.exceptions.RequestException as e:
        print(f"Erreur lors de la récupération des données Tiingo pour {symbol}: {e}")
        return []

def fetch_twelvedata(pair, last_request_time):
    retries = 3
    for attempt in range(retries):
        try:
            current_time = time.time()
            if last_request_time is not None:
                elapsed = current_time - last_request_time
                wait_time = max(12 - elapsed, 0)
                if wait_time > 0:
                    time.sleep(wait_time)

            params = {
                "symbol": pair,
                "interval": "5min",
                "outputsize": 60,
                "apikey": API_KEY_TWELVEDATA
            }
            response = requests.get(BASE_URL_TWELVEDATA, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if "values" not in data or not data["values"]:
                print(f"Erreur: Pas de données pour {pair}. Réponse: {data}")
                return [], current_time

            ohlcv = []
            for candle in data["values"]:
                timestamp = datetime.strptime(candle["datetime"], "%Y-%m-%d %H:%M:%S").timestamp()
                ohlcv.append((
                    float(timestamp),  # Conversion en float
                    float(candle["open"]),
                    float(candle["high"]),
                    float(candle["low"]),
                    float(candle["close"]),
                    float(candle.get("volume", 0))
                ))
            print(f"Données Twelvedata récupérées pour {pair}: {len(ohlcv)} bougies")
            return ohlcv, time.time()
        except requests.exceptions.RequestException as e:
            print(f"Erreur lors de la récupération des données pour {pair} (tentative {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(5)
            else:
                return [], time.time()
        except Exception as e:
            print(f"Erreur inattendue pour {pair}: {e}")
            return [], time.time()

# === GESTION DE LA BASE DE DONNÉES ===
def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    for pair in PAIRS_DB:
        c.execute(f'''CREATE TABLE IF NOT EXISTS {pair}_data 
                     (timestamp REAL, open REAL, high REAL, low REAL, close REAL, volume REAL)''')
        c.execute(f'DELETE FROM {pair}_data')  # Vider les tables
    c.execute('''CREATE TABLE IF NOT EXISTS Signals 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, pair TEXT, signal TEXT, price REAL, sl REAL, tp REAL, timestamp REAL, status TEXT, exit_price REAL, exit_timestamp REAL)''')
    conn.commit()
    conn.close()

def save_to_db(data_dict):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    for pair, data in data_dict.items():
        pair_db = pair.replace("/", "")
        if data:
            c.executemany(f'INSERT INTO {pair_db}_data VALUES (?,?,?,?,?,?)', data)
            c.execute(f'DELETE FROM {pair_db}_data WHERE timestamp NOT IN (SELECT timestamp FROM {pair_db}_data ORDER BY timestamp DESC LIMIT 120)')
    conn.commit()
    conn.close()

def load_data(pair):
    pair_db = pair.replace("/", "")
    conn = sqlite3.connect(DB_NAME)
    data = pd.read_sql_query(f"SELECT * FROM {pair_db}_data ORDER BY timestamp DESC LIMIT 120", conn)
    conn.close()
    return data

# === INDICATEURS ===
def atr(data, period=14):
    if len(data) < period:
        return pd.Series([0] * len(data), index=data.index)
    data['tr'] = np.maximum(data['high'] - data['low'], 
                             np.maximum(abs(data['high'] - data['close'].shift(1)), 
                                        abs(data['low'] - data['close'].shift(1))))
    data['atr'] = data['tr'].rolling(window=period).mean()
    return data['atr'].fillna(0)

def rsi(data, period=14):
    if len(data) < period:
        return pd.Series([50] * len(data), index=data.index)
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    data['rsi'] = 100 - (100 / (1 + rs))
    return data['rsi'].fillna(50)

def session_surge(data):
    if len(data) == 0:
        return pd.Series([0] * len(data), index=data.index)
    data['hour'] = data['timestamp'].apply(lambda x: datetime.fromtimestamp(x).hour)
    data['session_weight'] = np.where((data['hour'] >= 8) & (data['hour'] <= 17), 2,
                                      np.where((data['hour'] >= 13) & (data['hour'] <= 17), 1.5,
                                               0.8))
    data['price_change'] = data['close'].pct_change()
    data['session_surge'] = data['price_change'] * data['session_weight']
    return data['session_surge'].fillna(0)

# === CALENDRIER ÉCONOMIQUE ===
def check_economic_calendar():
    now = datetime.now()
    current_time = now.hour * 3600 + now.minute * 60 + now.second
    current_weekday = now.weekday()
    
    events = [
        {"name": "NFP", "time": 12 * 3600 + 30 * 60, "weekday": 4, "pairs": ["EUR/USD", "USD/JPY", "XAU/USD"]},
        {"name": "Taux Fed", "time": 18 * 3600, "weekday": 2, "pairs": ["EUR/USD", "USD/JPY", "XAU/USD"]},
        {"name": "Taux BCE", "time": 12 * 3600 + 45 * 60, "weekday": 3, "pairs": ["EUR/USD"]},
        {"name": "Taux BoJ", "time": 3 * 3600, "weekday": 2, "pairs": ["USD/JPY"]},
        {"name": "IPC US", "time": 12 * 3600 + 30 * 60, "weekday": 2, "pairs": ["EUR/USD", "USD/JPY", "XAU/USD"]}
    ]
    
    for event in events:
        event_time = event["time"]
        if current_weekday == event["weekday"]:
            if abs(current_time - event_time) <= 30 * 60:
                return event["pairs"], event["name"]
    return [], None

# === GESTION DES SIGNALS ===
def save_signal(signal, pair, price, sl, tp, timestamp):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("INSERT INTO Signals (pair, signal, price, sl, tp, timestamp, status) VALUES (?, ?, ?, ?, ?, ?, ?)", 
              (pair, signal, price, sl, tp, timestamp, "open"))
    conn.commit()
    conn.close()

def get_open_signals():
    conn = sqlite3.connect(DB_NAME)
    signals = pd.read_sql_query("SELECT * FROM Signals WHERE status='open' ORDER BY timestamp DESC", conn)
    conn.close()
    return signals

def close_signal(signal_id, exit_price, exit_timestamp):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("UPDATE Signals SET status='closed', exit_price=?, exit_timestamp=? WHERE id=?", 
              (exit_price, exit_timestamp, signal_id))
    conn.commit()
    conn.close()

def count_signals_today():
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM Signals WHERE timestamp >= ? AND timestamp < ?", 
              (today, today + 86400))
    count = c.fetchone()[0]
    conn.close()
    return count

# === PRÉPARATION DES DONNÉES POUR LE ML ===
def prepare_ml_data(data, pair):
    if len(data) < 24:
        print(f"Pas assez de données pour {pair} ({len(data)} bougies).")
        return None, None, None, None
    
    data['atr'] = atr(data)
    data['rsi'] = rsi(data)
    data['session_surge'] = session_surge(data)
    
    prices = data['close'].values.reshape(-1, 1)
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=3, n_init=10, random_state=42).fit(prices)
    levels = kmeans.cluster_centers_.flatten()
    
    data['future_return'] = data['close'].shift(-24) / data['close'] - 1
    data['target'] = (data['future_return'] > 0).astype(int)
    
    features = ['atr', 'rsi', 'session_surge']
    X = data[features].fillna(0)
    y = data['target'].fillna(0)
    
    return X, y, data, levels

# === ENTRAÎNEMENT ET PRÉDICTION ===
def train_and_predict(X, y, data, pair, levels):
    total_samples = len(X)
    if total_samples < 2:
        print(f"Pas assez de données pour entraîner le modèle pour {pair} ({total_samples} échantillons).")
        return None, None, None, None
    
    train_size = int(0.8 * total_samples)
    if train_size == total_samples:
        train_size -= 1
    
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    if len(X_test) == 0:
        print(f"X_test est vide pour {pair} (train_size={train_size}, total_samples={total_samples}).")
        return None, None, None, None
    
    if len(np.unique(y_train)) < 2:
        print(f"Pas assez de variation dans les données d'entraînement pour {pair}.")
        return None, None, None, None
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy du modèle pour {pair}: {accuracy:.2f}")
    
    latest_price = data['close'].iloc[-1]
    signal = None
    sl = None
    tp = None
    
    threshold = 0.85 if pair == "XAU/USD" else 0.90
    
    # Calcul SL et TP en points (pourcentage pour XAU/USD, pips pour les autres)
    if pair in ["EUR/USD", "USD/JPY"]:
        sl_pips = 40
        tp_pips = 80
        sl = latest_price - sl_pips * 0.0001
        tp = latest_price + tp_pips * 0.0001
        # Arrondir à trois chiffres après la virgule
        sl = round(sl, 5)
        tp = round(tp, 5)
    else:  # XAU/USD
        # Fourchette de TP et SL en points (100 points = 1$)
        tp_min_points = 300
        tp_max_points = 850
        sl_min_points = -700
        sl_max_points = -300
        
        # Calcul du TP et SL aléatoires dans les fourchettes spécifiées
        tp_points = np.random.randint(tp_min_points, tp_max_points + 1)
        sl_points = np.random.randint(sl_min_points, sl_max_points + 1)
        
        # Conversion des points en prix
        tp = latest_price + (tp_points / 100)
        sl = latest_price + (sl_points / 100)
    
    # Formatage du prix
    price_format = ".5f" if pair in ["EUR/USD", "USD/JPY"] else ".2f"
    
    if probabilities[-1] > threshold:
        signal = f"buy {latest_price:{price_format}} (Probabilité: {probabilities[-1]:.2f}, SL: {sl:{price_format}}, TP: {tp:{price_format}})"
    elif probabilities[-1] < (1 - threshold):
        signal = f"sell {latest_price:{price_format}} (Probabilité: {1-probabilities[-1]:.2f}, SL: {sl:{price_format}}, TP: {tp:{price_format}})"
    
    return signal, latest_price, sl, tp

# === SUIVI DES TRADES ===
def check_closed_trades(current_timestamp, data_dict):
    signals = get_open_signals()
    for _, signal in signals.iterrows():
        signal_id = signal['id']
        pair = signal['pair']
        signal_type = signal['signal'].split()[0]
        entry_price = signal['price']
        sl = signal['sl']
        tp = signal['tp']
        entry_timestamp = signal['timestamp']
        
        if current_timestamp - entry_timestamp >= 7200:
            if pair not in data_dict or len(data_dict[pair]) == 0:
                print(f"Données manquantes pour clôturer le trade #{signal_id} ({pair}).")
                continue
            
            exit_price = data_dict[pair]['close'].iloc[-1]
            
            if signal_type == "buy":
                price_data = data_dict[pair]
                min_price = price_data[price_data['timestamp'] >= entry_timestamp]['low'].min()
                max_price = price_data[price_data['timestamp'] >= entry_timestamp]['high'].max()
                if min_price <= sl:
                    exit_price = sl
                    exit_timestamp = entry_timestamp + 7200
                elif max_price >= tp:
                    exit_price = tp
                    exit_timestamp = entry_timestamp + 7200
                else:
                    exit_timestamp = entry_timestamp + 7200
            else:
                price_data = data_dict[pair]
                max_price = price_data[price_data['timestamp'] >= entry_timestamp]['high'].max()
                min_price = price_data[price_data['timestamp'] >= entry_timestamp]['low'].min()
                if max_price >= sl:
                    exit_price = sl
                    exit_timestamp = entry_timestamp + 7200
                elif min_price <= tp:
                    exit_price = tp
                    exit_timestamp = entry_timestamp + 7200
                else:
                    exit_timestamp = entry_timestamp + 7200
            
            entry_date = datetime.fromtimestamp(entry_timestamp).strftime('%Y-%m-%d %H:%M:%S')
            exit_date = datetime.fromtimestamp(exit_timestamp).strftime('%Y-%m-%d %H:%M:%S')
            duration = (exit_timestamp - entry_timestamp) / 60
            print(f"Trade #{signal_id} ({pair}) clôturé :")
            print(f"Entrée: {entry_date}, Sortie: {exit_date}, Durée: {duration:.2f} min")
            print(f"Entrée: {entry_price:.2f}, Sortie: {exit_price:.2f}")
            close_signal(signal_id, exit_price, exit_timestamp)

# === ANALYSE DES DONNÉES ===
def analyze_data(data_dict):
    current_timestamp = time.time()
    
    check_closed_trades(current_timestamp, data_dict)
    
    signals_today = count_signals_today()
    if signals_today >= 3:
        print("Limite de 3 signaux par jour atteinte.")
        return
    
    open_signals = get_open_signals()
    if not open_signals.empty:
        latest_signal = open_signals.iloc[0]
        time_since_signal = current_timestamp - latest_signal['timestamp']
        signal_id = latest_signal['id']
        pair = latest_signal['pair']
        if time_since_signal < 7200:
            print(f"Trade #{signal_id} ({pair}) en cours, prochain signal dans {(7200 - time_since_signal)/60:.2f} min.")
            return
    
    closed_signals = pd.read_sql_query("SELECT * FROM Signals WHERE status='closed' ORDER BY exit_timestamp DESC LIMIT 1", sqlite3.connect('forex_trading.db'))
    if not closed_signals.empty:
        last_closed = closed_signals.iloc[0]
        time_since_last_closed = current_timestamp - last_closed['exit_timestamp']
        if time_since_last_closed < 3600:
            print(f"Délai minimum après clôture, prochain signal dans {(3600 - time_since_last_closed)/60:.2f} min.")
            return
    
    affected_pairs, event_name = check_economic_calendar()
    if affected_pairs:
        print(f"Annonce majeure prévue ({event_name}), trading bloqué pour {affected_pairs}.")
        return
    
    best_signal = None
    best_prob = 0
    best_pair = None
    best_price = None
    best_sl = None
    best_tp = None
    
    for pair in PAIRS_TWELVEDATA + [PAIR_TIINGO]:
        if pair in affected_pairs:
            continue
        data = data_dict[pair]
        if len(data) < 24:
            print(f"Données insuffisantes pour {pair} ({len(data)} bougies).")
            continue
        
        X, y, data, levels = prepare_ml_data(data, pair)
        if X is None:
            continue
        
        signal, price, sl, tp = train_and_predict(X, y, data, pair, levels)
        if signal:
            prob = float(signal.split("Probabilité: ")[1].split(",")[0])
            threshold = 0.85 if pair == "XAU/USD" else 0.90
            adjusted_prob = prob + 0.05 if pair == "XAU/USD" else prob
            if adjusted_prob > best_prob and prob >= threshold:
                best_signal = signal
                best_prob = adjusted_prob
                best_pair = pair
                best_price = price
                best_sl = sl
                best_tp = tp
    
    if best_signal:
        print(f"Signal {best_pair}: {best_signal}")
        
        # Formater le message Telegram
        telegram_message = format_signal_message(best_pair, best_signal, best_price, best_sl, best_tp, current_timestamp)
        
        # Envoyer le message via Telegram
        send_telegram_message(telegram_message)
        
        # Sauvegarder le signal
        save_signal(best_signal, best_pair, best_price, best_sl, best_tp, current_timestamp)
    else:
        print("Aucun signal clair pour l'instant.")

# === FONCTION PRINCIPALE ===
def run_bot():
    init_db()

    last_request_time_twelvedata = None
    data_dict = {}

    # Initialisation des données Twelvedata
    for pair in PAIRS_TWELVEDATA:
        data_dict[pair] = load_data(pair)

    # Initialisation des données Tiingo
    data_dict[PAIR_TIINGO] = load_data(PAIR_TIINGO)

    while True:
        print("Vérification des données...")

        # Vérification des données Twelvedata
        need_fetch_twelvedata = False
        for pair in PAIRS_TWELVEDATA:
            data = data_dict[pair]
            if len(data) < 24:
                need_fetch_twelvedata = True
                break
            latest_timestamp = data['timestamp'].max()
            current_timestamp = time.time()
            if (current_timestamp - latest_timestamp) > 300:  # 5min
                need_fetch_twelvedata = True
                break

        # Vérification des données Tiingo
        need_fetch_tiingo = False
        data = data_dict[PAIR_TIINGO]
        if len(data) < 24:
            need_fetch_tiingo = True
        else:
            latest_timestamp = data['timestamp'].max()
            current_timestamp = time.time()
            if (current_timestamp - latest_timestamp) > 300:  # 5min
                need_fetch_tiingo = True

        # Récupération des données si nécessaire
        if need_fetch_twelvedata or need_fetch_tiingo:
            print("Récupération des données...")
            new_data_dict = {}

            # Récupération des données Twelvedata
            if need_fetch_twelvedata:
                for pair in PAIRS_TWELVEDATA:
                    data, last_request_time_twelvedata = fetch_twelvedata(pair, last_request_time_twelvedata)
                    new_data_dict[pair] = data

            # Récupération des données Tiingo
            if need_fetch_tiingo:
                tiingo_data = fetch_tiingo_data(PAIR_TIINGO)
                new_data_dict[PAIR_TIINGO] = tiingo_data

            save_to_db(new_data_dict)
            print("Données mises à jour!")

            # Recharge des données après la mise à jour
            for pair in PAIRS_TWELVEDATA:
                data_dict[pair] = load_data(pair)
            data_dict[PAIR_TIINGO] = load_data(PAIR_TIINGO)
        else:
            print("Données récentes disponibles, pas de nouvelle récupération.")

        print("Analyse en cours...")
        analyze_data(data_dict)
        print("Analyse OK!")

        time.sleep(UPDATE_INTERVAL)

# === POINT D'ENTRÉE DU SCRIPT ===
if __name__ == "__main__":
    run_bot()