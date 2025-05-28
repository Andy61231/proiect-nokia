from flask import Flask, render_template_string, request
import pandas as pd
import folium
from math import radians, cos, sin, asin, sqrt
import numpy as np
import os

app = Flask(__name__)

# --- Funcțiile auxiliare (adauga_punct_dupa_doua_cifre, calculeaza_distant, get_color)
def adauga_punct_dupa_doua_cifre(val):
    val_str = str(val).strip().replace(",", "").replace(".", "")
    if not val_str.isdigit():
        return np.nan
    if len(val_str) > 2:
        try:
            return float(val_str[:2] + '.' + val_str[2:])
        except ValueError:
            return np.nan
    return np.nan

def calculeaza_distant(lat1, lon1, lat2, lon2):
    try:
        coords = [float(lon1), float(lat1), float(lon2), float(lat2)]
    except (ValueError, TypeError):
        return float('inf')
    if any(pd.isna(c) for c in coords):
        return float('inf')
    try:
        lon1_rad, lat1_rad, lon2_rad, lat2_rad = map(radians, coords)
    except ValueError:
        return float('inf')
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    a = sin(dlat/2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon/2)**2
    if not (0 <= a <= 1):
        return 0.0 if a < 0 else float('inf')
    c = 2 * asin(sqrt(a)); r = 6371
    return c * r

def get_color(rssi):
    colors = {
        'foarte_bun': 'darkgreen', 'bun': 'limegreen', 'slab': 'orange',
        'limita': 'red', 'fara_semnal': 'darkred', 'invalid': 'grey'
    }
    try:
        rssi_float = float(rssi)
        if pd.isna(rssi_float):
            return colors['invalid']
    except (ValueError, TypeError):
        return colors['invalid']
    if rssi_float >= -85: return colors['foarte_bun']
    elif -100 <= rssi_float < -85: return colors['bun']
    elif -115 <= rssi_float < -100: return colors['slab']
    elif -125 <= rssi_float < -115: return colors['limita']
    else: return colors['fara_semnal']

# ===========================
# RUTA PRINCIPALĂ FLASK
# ===========================
@app.route("/")
def index():
    selected_tech = request.args.get('tech', 'ALL').upper()
    selected_psc_pci_str = request.args.get('psc_pci', '').strip()
    selected_operator = request.args.get('operator', 'ALL')

    input_filename = "date_procesate_optim.csv"
    df = pd.DataFrame(columns=['lat', 'long', 'rssi', 'tech', 'psc_pci', 'net_op_name'])
    map_error_message = None
    psc_pci_filter_applied = False
    selected_psc_pci_num = None
    operator_list = []

    if selected_psc_pci_str:
        try:
            selected_psc_pci_num = float(selected_psc_pci_str)
        except ValueError:
            map_error_message = f"Valoarea introdusă pentru PSC/PCI ('{selected_psc_pci_str}') nu este un număr valid. Filtrul PSC/PCI nu va fi aplicat."
            selected_psc_pci_str = ""

    if not os.path.exists(input_filename):
        map_error_message = f"Eroare: Fișierul de date '{input_filename}' nu a fost găsit."

    else:
        try:
            dtype_spec = {'lat': str, 'long': str, 'rssi': str, 'tech': str, 'psc_pci': str, 'net_op_name': str}
            df_temp = pd.read_csv(input_filename, sep=',', dtype=dtype_spec, low_memory=False)
            df_temp.columns = df_temp.columns.str.strip().str.lower()
            required_cols = ['lat', 'long']
            missing_required = [col for col in required_cols if col not in df_temp.columns]
            if missing_required:
                 map_error_message = f"Eroare: Coloanele {missing_required} lipsesc din {input_filename}."
            else:
                df = df_temp
                if selected_tech != 'ALL' and 'tech' not in df.columns:
                    map_error_message = (map_error_message or "") + " Avertisment: Coloana 'tech' lipsește. Filtrarea după tehnologie nu este posibilă."
                if selected_psc_pci_num is not None and 'psc_pci' not in df.columns:
                    map_error_message = (map_error_message or "") + " Avertisment: Coloana 'psc_pci' lipsește. Filtrarea după PSC/PCI nu este posibilă."
                    selected_psc_pci_num = None
                if selected_operator != 'ALL' and 'net_op_name' not in df.columns:
                    map_error_message = (map_error_message or "") + " Avertisment: Coloana 'net_op_name' lipsește. Filtrarea după operator nu este posibilă."
                    selected_operator = 'ALL'
                if 'net_op_name' in df.columns:
                    operator_list = sorted(list(df['net_op_name'].dropna().astype(str).str.strip().unique()))
        except pd.errors.EmptyDataError:
            map_error_message = f"Avertisment: Fișierul {input_filename} este gol."
        except Exception as e:
            map_error_message = f"Eroare necunoscută la citirea fișierului {input_filename}: {e}"
            df = pd.DataFrame(columns=['lat', 'long', 'rssi', 'tech', 'psc_pci', 'net_op_name'])

    if not df.empty:
        try:
            df['lat'] = df['lat'].apply(adauga_punct_dupa_doua_cifre)
            df['long'] = df['long'].apply(adauga_punct_dupa_doua_cifre)
            if 'rssi' in df.columns: df['rssi'] = pd.to_numeric(df['rssi'], errors='coerce')
            if 'psc_pci' in df.columns: df['psc_pci'] = pd.to_numeric(df['psc_pci'], errors='coerce')
            if 'net_op_name' in df.columns: df['net_op_name'] = df['net_op_name'].astype(str).str.strip()
            if 'tech' in df.columns:
                 df['tech'] = df['tech'].astype(str).str.strip().str.upper()
                 tech_mapping = {'NR': '5G','LTE': '4G', 'WCDMA': '3G', 'UMTS': '3G', 'GSM': '2G'}
                 df['tech'] = df['tech'].replace(tech_mapping)

            df = df.dropna(subset=['lat', 'long']).copy()

            if 'tech' in df.columns and selected_tech != 'ALL':
                df = df[df['tech'] == selected_tech]
                # print(f"Rânduri după filtrare tehnologie ('{selected_tech}'): {len(df)}")

            if 'net_op_name' in df.columns and selected_operator != 'ALL':
                df = df[df['net_op_name'] == selected_operator]
                # print(f"Rânduri după filtrare operator ('{selected_operator}'): {len(df)}")

            if selected_psc_pci_num is not None and 'psc_pci' in df.columns:
                df = df[df['psc_pci'] == selected_psc_pci_num]
                psc_pci_filter_applied = True
                # print(f"Rânduri după filtrare PSC/PCI ('{selected_psc_pci_num}'): {len(df)}")

            if not df.empty:
                # print("Filtrare puncte apropiate (<5m)...")
                df = df.reset_index(drop=True)
                indices_to_keep = []
                if len(df) > 0:
                    indices_to_keep.append(0)
                    last_valid_idx = 0
                    for current_idx in range(1, len(df)):
                        lat1, lon1 = df.loc[last_valid_idx, 'lat'], df.loc[last_valid_idx, 'long']
                        lat2, lon2 = df.loc[current_idx, 'lat'], df.loc[current_idx, 'long']
                        if pd.notna(lat1) and pd.notna(lon1) and pd.notna(lat2) and pd.notna(lon2):
                            dist = calculeaza_distant(lat1, lon1, lat2, lon2)
                            if dist >= 0.005: # < 5 metri
                                indices_to_keep.append(current_idx)
                                last_valid_idx = current_idx
                df = df.loc[indices_to_keep].reset_index(drop=True)
                # print(f"Rânduri după filtrarea punctelor apropiate: {len(df)}")
            # else:
                 # print("DataFrame gol după filtrele anterioare, se sare peste filtrarea punctelor apropiate.")
        except KeyError as e:
             map_error_message = f"Eroare: Coloana '{e}' necesară pentru procesare nu a fost găsită în fișier."
             df = pd.DataFrame(columns=['lat', 'long', 'rssi', 'tech', 'psc_pci', 'net_op_name'])
        except Exception as e:
             map_error_message = f"Eroare în timpul procesării datelor: {e}"
             df = pd.DataFrame(columns=['lat', 'long', 'rssi', 'tech', 'psc_pci', 'net_op_name'])

    map_is_empty = df.empty
    filters_applied = (selected_tech != 'ALL' or selected_psc_pci_str != '' or selected_operator != 'ALL')

    if map_is_empty:
        center_lat, center_long = 45.7555, 21.2255 # Timișoara
        if filters_applied and not map_error_message:
            map_error_message = "Nu s-au găsit date pentru filtrele selectate."
    elif not filters_applied:
        center_lat, center_long = 45.7555, 21.2255 # Timișoara
    else:
        # print("Filtre specifice aplicate. Se calculează centrul din datele filtrate.")
        center_lat = df['lat'].mean()
        center_long = df['long'].mean()
        if pd.isna(center_lat) or pd.isna(center_long):
            # print("Avertisment: Media coordonatelor filtrate este NaN, se folosește centrul default.")
            center_lat, center_long = 45.7555, 21.2255 # Timișoara

    m = folium.Map(location=[center_lat, center_long], zoom_start=13, tiles='CartoDB Positron')

    legend_html = '''
     <div style="position: fixed; top: 10px; right: 10px; width: 210px; background-color: rgba(255, 255, 255, 0.85);
         border:1px solid grey; z-index:9999; font-size:12px; padding:8px; border-radius: 5px; box-shadow: 2px 2px 5px rgba(0,0,0,0.2);">
         <b style="font-size: 14px; display: block; margin-bottom: 5px;">Legenda Nivel Semnal (RSSI)</b>
         <div style="margin-bottom: 3px;"><i style="background:darkgreen; border-radius:50%; width:12px; height:12px; display:inline-block; margin-right: 5px; vertical-align: middle;"></i> Foarte bun (≥ -85 dBm)</div>
         <div style="margin-bottom: 3px;"><i style="background:limegreen; border-radius:50%; width:12px; height:12px; display:inline-block; margin-right: 5px; vertical-align: middle;"></i> Bun (-85 la -100 dBm)</div>
         <div style="margin-bottom: 3px;"><i style="background:orange; border-radius:50%; width:12px; height:12px; display:inline-block; margin-right: 5px; vertical-align: middle;"></i> Slab (-100 la -115 dBm)</div>
         <div style="margin-bottom: 3px;"><i style="background:red; border-radius:50%; width:12px; height:12px; display:inline-block; margin-right: 5px; vertical-align: middle;"></i> Limita (-115 la -125 dBm)</div>
         <div style="margin-bottom: 3px;"><i style="background:darkred; border-radius:50%; width:12px; height:12px; display:inline-block; margin-right: 5px; vertical-align: middle;"></i> Fără semnal (< -125 dBm)</div>
         <div><i style="background:grey; border-radius:50%; width:12px; height:12px; display:inline-block; margin-right: 5px; vertical-align: middle;"></i> Necunoscut / Invalid</div>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))

    if not map_is_empty:
        for current_idx, row in enumerate(df.itertuples(index=False)):
            lat = getattr(row, 'lat', None)
            lon = getattr(row, 'long', None)

            if pd.isna(lat) or pd.isna(lon):
                continue

            rssi_val = getattr(row, 'rssi', None)
            tech_val = getattr(row, 'tech', 'N/A')
            psc_pci_val = getattr(row, 'psc_pci', None)
            operator_val = getattr(row, 'net_op_name', 'N/A')
            color = get_color(rssi_val)

            psc_pci_display = 'N/A'
            if pd.notna(psc_pci_val):
                try:
                    psc_pci_display = f"{int(psc_pci_val)}"
                except (ValueError, TypeError):
                    try:
                         psc_pci_display = f"{psc_pci_val:.0f}"
                    except (ValueError, TypeError):
                         psc_pci_display = str(psc_pci_val)

            tooltip_text = (f"Operator: {operator_val}<br>"
                            f"Tehnologie: {tech_val}<br>"
                            f"RSSI: {rssi_val if pd.notna(rssi_val) else 'N/A'}<br>"
                            f"PSC/PCI: {psc_pci_display}")

            folium.Circle(
                location=[lat, lon], radius=3, color=color, fill=True,
                fill_color=color, fill_opacity=0.7, tooltip=tooltip_text
            ).add_to(m)

            next_idx = current_idx + 1
            if next_idx < len(df):
                 next_row_data = df.iloc[next_idx]
                 next_lat = next_row_data.get('lat')
                 next_lon = next_row_data.get('long')

                 if pd.notna(next_lat) and pd.notna(next_lon):
                    dist = calculeaza_distant(lat, lon, next_lat, next_lon)
                    if dist < 0.01: # < 10 metri
                        line_color = color
                        folium.PolyLine(
                            locations=[[lat, lon], [next_lat, next_lon]],
                            color=line_color, weight=1.5, opacity=0.6
                        ).add_to(m)
    else:
        if map_error_message:
             error_html = f'''
             <div style="position: fixed; top: 10px; left: 50%; transform: translateX(-50%);
                  background-color: rgba(255, 100, 100, 0.8); color: white; padding: 10px;
                  border-radius: 5px; z-index: 10000; font-size: 14px; text-align: center; max-width: 80%;">
                  {map_error_message}
             </div>
             '''
             m.get_root().html.add_child(folium.Element(error_html))

    map_html = m._repr_html_()

    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Harta Semnal ({operator_display} / {tech_display}{psc_pci_display_title})</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            html, body { margin: 0; padding: 0; height: 100%; font-family: sans-serif; }
            #map { height: 100%; width: 100%; }
            .filter-container { position: fixed; top: 10px; left: 10px; background-color: rgba(255, 255, 255, 0.9); padding: 10px 15px; border-radius: 5px; z-index: 1000; border: 1px solid #ccc; box-shadow: 2px 2px 5px rgba(0,0,0,0.2); display: flex; flex-wrap: wrap; align-items: center; gap: 15px; }
            .filter-group { display: flex; align-items: center; margin-bottom: 5px; }
            .filter-group label { margin-right: 5px; font-size: 13px; font-weight: bold; white-space: nowrap; }
            .filter-group select, .filter-group input[type='text'] { font-size: 13px; padding: 4px; border: 1px solid #ccc; border-radius: 3px; }
            .filter-group input[type='text'] { width: 60px; }
            .filter-container button { font-size: 13px; padding: 4px 10px; cursor: pointer; background-color: #eee; border: 1px solid #ccc; border-radius: 3px; margin-left: 10px; }
            .filter-container button:hover { background-color: #ddd; }
            @media (max-width: 600px) {
                .filter-container { flex-direction: column; align-items: flex-start; }
                .filter-group { width: 100%; justify-content: space-between; }
                 .filter-group label { min-width: 80px; }
                .filter-container button { width: 100%; margin-left: 0; margin-top: 10px; }
            }
        </style>
    </head>
    <body>
        <div class="filter-container">
            <form method="GET" action="/" id="filter-form" style="display: contents;">
                 <div class="filter-group">
                    <label for="operator-select">Operator:</label>
                    <select name="operator" id="operator-select">
                        <option value="ALL" {% if current_operator == 'ALL' %}selected{% endif %}>Toți</option>
                        {% for op in operator_options %}
                            <option value="{{ op }}" {% if current_operator == op %}selected{% endif %}>{{ op }}</option>
                        {% endfor %}
                    </select>
                </div>
                <div class="filter-group">
                    <label for="tech-select">Tehnologie:</label>
                    <select name="tech" id="tech-select">
                        <option value="ALL" {% if current_tech == 'ALL' %}selected{% endif %}>Toate</option>
                        <option value="2G" {% if current_tech == '2G' %}selected{% endif %}>2G</option>
                        <option value="3G" {% if current_tech == '3G' %}selected{% endif %}>3G</option>
                        <option value="4G" {% if current_tech == '4G' %}selected{% endif %}>4G</option>
                        <option value="5G" {% if current_tech == '5G' %}selected{% endif %}>5G</option>
                    </select>
                </div>
                <div class="filter-group">
                    <label for="psc-pci-input">PSC/PCI:</label>
                    <input type="text" name="psc_pci" id="psc-pci-input" value="{{ current_psc_pci }}" placeholder="ex: 123">
                </div>
                <button type="submit">Aplică Filtre</button>
            </form>
        </div>
        <div id="map">
             {{ map_content|safe }}
        </div>
    </body>
    </html>
    """

    tech_display_text = "Toate" if selected_tech == 'ALL' else selected_tech
    operator_display_text = "Toți" if selected_operator == 'ALL' else selected_operator
    psc_pci_title_text = f", PSC/PCI: {selected_psc_pci_str}" if selected_psc_pci_str else ""

    return render_template_string(html_template,
                                  map_content=map_html,
                                  current_tech=selected_tech,
                                  current_psc_pci=selected_psc_pci_str,
                                  current_operator=selected_operator,
                                  operator_options=operator_list,
                                  tech_display=tech_display_text,
                                  operator_display=operator_display_text,
                                  psc_pci_display_title=psc_pci_title_text)

# Această parte este pentru dezvoltare locală și va fi ignorată de Gunicorn/Waitress
# Dacă o rulezi direct cu "python main.py", serverul de dezvoltare Flask va porni.
# Pentru producție sau acces prin DuckDNS, vei folosi un server WSGI (Gunicorn/Waitress).
if __name__ == "__main__":
    print("Pentru a rula serverul de dezvoltare Flask, decomentează linia de mai jos.")
    print("Pentru acces prin DuckDNS, folosește Gunicorn sau Waitress.")
    # app.run(host='0.0.0.0', port=5001, debug=True) # debug=True NU este pentru producție!