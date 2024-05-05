import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from flask import Flask, render_template, request, jsonify
import networkx as nx
import csv
import matplotlib.pyplot as plt
import matplotlib.backends.backend_agg as agg
import base64
from io import BytesIO
from datetime import datetime, timedelta
import random

app = Flask(__name__)

def citeste_graf(fisier_csv):
    G = nx.Graph()
    with open(fisier_csv, 'r') as file:
        reader = csv.reader(file)
        header = next(reader)
        for row in reader:
            strada1, strada2, distanta, timp_standard, *orar = row
            G.add_edge(strada1, strada2, distanta=float(distanta), timp_standard=float(timp_standard),
                       trafic_orar=dict(zip(header[4:], map(int, orar))))
    return G

def citeste_trafic(fisier_csv):
    trafic = {}
    with open(fisier_csv, 'r') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            zi, ora, strada_a, strada_b, numar_participanti = row
            cheie = (zi, ora, strada_a, strada_b)
            trafic[cheie] = int(numar_participanti)
    return trafic

def deseneaza_graf_si_ruta(graf, ruta_optima):
    plt.clf()
    pos = nx.spring_layout(graf)
    nx.draw(graf, pos, with_labels=True, font_weight='bold', node_size=700, node_color='skyblue', font_size=8)

    # Adaugă etichete pentru muchii
    edge_labels = {(strada1, strada2): f"{graf[strada1][strada2]['distanta']} km" for strada1, strada2 in graf.edges}
    nx.draw_networkx_edge_labels(graf, pos, edge_labels=edge_labels)

    edges = [(ruta_optima[i], ruta_optima[i + 1]) for i in range(len(ruta_optima) - 1)]
    nx.draw_networkx_edges(graf, pos, edgelist=edges, edge_color='red', width=2)

    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    img_base64 = base64.b64encode(img.getvalue()).decode()

    return img_base64


def invata_din_date(fisier_strazi, fisier_trafic):

    graf_invatat = citeste_graf(fisier_strazi)
    trafic_invatat = citeste_trafic(fisier_trafic)


    noduri_graf_invatat = list(graf_invatat.nodes)
    muchii_graf_invatat = list(graf_invatat.edges)
    media_trafic = sum(trafic_invatat.values()) / len(trafic_invatat)


    torch.manual_seed(42)
    model = torch.nn.Linear(5, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()


    scaler = StandardScaler()
    X_train, X_test, y_train, y_test = train_test_split(noduri_graf_invatat, muchii_graf_invatat, test_size=0.2,
                                                        random_state=42)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    rezultat_invatare = f"Învățare din datele furnizate: {len(noduri_graf_invatat)} noduri în graf, {len(muchii_graf_invatat)} muchii, trafic mediu: {media_trafic}"

    return rezultat_invatare
def calculeaza_rute_optime(graf, start, destinatie):
    return nx.shortest_path(graf, start, destinatie, weight='distanta')

def timp_estimat_catre_destinatie(graf, ruta_optima, ora_pornire, trafic):
    timp_estimat_total = 0
    factor_corectie = 1.0
    trafic_aglomerat = False

    for strada1, strada2 in zip(ruta_optima[:-1], ruta_optima[1:]):
        cheie_ora = f'ora_{ora_pornire}'
        trafic_cheie = (datetime.now().strftime("%A"), cheie_ora, strada1, strada2)

        if trafic_cheie in trafic:
            numar_participanti = trafic[trafic_cheie]

            if numar_participanti > 50:
                trafic_aglomerat = True
                factor_corectie += 0.2

            trafic_orar = graf[strada1][strada2]['trafic_orar']
            if cheie_ora in trafic_orar:
                timp_estimat_total += trafic_orar[cheie_ora] * factor_corectie
            else:
                cheie_ora_ieri = f'ora_{23}'
                if cheie_ora_ieri in trafic_orar:
                    timp_estimat_total += trafic_orar[cheie_ora_ieri] * factor_corectie
                else:
                    timp_estimat_total += 0
        else:
            trafic_orar = graf[strada1][strada2]['trafic_orar']
            if cheie_ora in trafic_orar:
                timp_estimat_total += trafic_orar[cheie_ora]
            else:
                cheie_ora_ieri = f'ora_{23}'
                if cheie_ora_ieri in trafic_orar:
                    timp_estimat_total += trafic_orar[cheie_ora_ieri]
                else:
                    timp_estimat_total += 0

    return timp_estimat_total, trafic_aglomerat

def calculeaza_timpul_estimat(ora_pornire, timp_estimat):
    ora_sosire = ora_pornire + timedelta(minutes=timp_estimat)
    return ora_sosire.strftime('%H:%M')

def obtine_fenomen_meteorologic():
    fenomene = ["insorit", "ploua", "ninge"]
    fenomen = random.choice(fenomene)
    timp_suplimentar = 0

    if fenomen in ["ploua", "ninge"]:
        timp_suplimentar = random.randint(3, 10)

    return fenomen, timp_suplimentar

def predictie_trafic_actual(trafic, ziua_curenta, strada_start, strada_destinatie):
    ore_reale = []
    participanti_reali = []
    participanti_predictie = []

    for cheie, numar_participanti in trafic.items():
        zi, ora, strada_a, strada_b = cheie
        if zi == ziua_curenta and ((strada_a == strada_start and strada_b == strada_destinatie) or
                                    (strada_a == strada_destinatie and strada_b == strada_start)):
            ore_reale.append(ora)
            participanti_reali.append(numar_participanti)

    trafic_predictie = [(participanti_reali[i-1] + participanti_reali[i] + participanti_reali[i+1]) / 3
                        for i in range(1, len(participanti_reali)-1)]

    ore_predictie = ore_reale[1:-1]

    return ore_reale, participanti_reali, ore_predictie, trafic_predictie

def grafic_evolutie_trafic(trafic, strada_start, strada_destinatie):
    ziua_curenta = datetime.now().strftime("%A")
    ore_reale, participanti_reali, ore_predictie, trafic_predictie = predictie_trafic_actual(trafic, ziua_curenta, strada_start, strada_destinatie)

    plt.rcParams.update({'font.size': 12})


    fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')  # Setează culoarea de fundal la alb


    ax.plot(ore_reale, participanti_reali, color='b', label='Real', marker='o', linestyle='-')
    ax.plot(ore_predictie, trafic_predictie, color='r', label='Predictie', marker='o', linestyle='--')

    ax.set_title(f'Predictie flux trafic ruta {strada_start}-{strada_destinatie} ')
    ax.set_xlabel('Ora')
    ax.set_ylabel('Numar Participanti')


    ax.set_xticks(range(24))
    ax.set_xticklabels([str(i) for i in range(24)])


    ax.legend()


    ax.grid(True, linestyle='--', alpha=0.7)


    img = BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    img_base64 = base64.b64encode(img.getvalue()).decode()
    plt.close()

    return img_base64



@app.route('/')
def index():
    return render_template('index.html')

def grafic_timp_actual_si_predictie(trafic, ziua_curenta, strada_start, strada_destinatie):
    ore_actuale, participanti_reali, ore_predictie, trafic_predictie = predictie_trafic_actual(trafic, ziua_curenta, strada_start, strada_destinatie)


    plt.rcParams.update({'font.size': 12})


    fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')  # Setează culoarea de fundal la alb

    ax.plot(ore_actuale, participanti_reali, color='g', label='Actual', marker='o', linestyle='-')
    ax.plot(ore_predictie, trafic_predictie, color='r', label='Predictie', marker='o', linestyle='--')


    ax.set_title(f'Trafic actual și predictie pentru {strada_start}-{strada_destinatie}')
    ax.set_xlabel('Ora')
    ax.set_ylabel('Numar Participanti')


    ax.set_xticks(range(24))
    ax.set_xticklabels([str(i) for i in range(24)])


    ax.legend()

    ax.grid(True, linestyle='--', alpha=0.7)

    img = BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    img_base64 = base64.b64encode(img.getvalue()).decode()
    plt.close()

    return img_base64

@app.route('/calculeaza_ruta', methods=['POST'])
def calculeaza_ruta():
    strada_start = request.form['strada_start']
    strada_destinatie = request.form['strada_destinatie']

    graf_strazi = citeste_graf('strazi_cu_trafic.csv')
    ruta_optima = calculeaza_rute_optime(graf_strazi, strada_start, strada_destinatie)

    trafic = citeste_trafic('trafic.csv')
    ora_pornire = datetime.now().strftime("%H:%M")
    timp_estimat, trafic_aglomerat = timp_estimat_catre_destinatie(graf_strazi, ruta_optima, ora_pornire, trafic)

    fenomen, timp_suplimentar = obtine_fenomen_meteorologic()
    timp_estimat += timp_suplimentar

    ora_sosire_estimata = calculeaza_timpul_estimat(datetime.strptime(ora_pornire, "%H:%M"), timp_estimat)

    img_base64 = deseneaza_graf_si_ruta(graf_strazi, ruta_optima)
    grafic_base64 = grafic_evolutie_trafic(trafic, strada_start, strada_destinatie)
    grafic_timp_actual_predictie_base64 = grafic_timp_actual_si_predictie(trafic, datetime.now().strftime("%A"), strada_start, strada_destinatie)

    return jsonify({'ruta_optima': ruta_optima,
                    'ora_sosire_estimata': ora_sosire_estimata,
                    'trafic_aglomerat': trafic_aglomerat,
                    'graf_base64': img_base64,
                    'fenomen_meteorologic': fenomen,
                    'timp_suplimentar': timp_suplimentar,
                    'grafic_trafic': grafic_base64,
                    'grafic_timp_actual_predictie': grafic_timp_actual_predictie_base64})



if __name__ == '__main__':
    app.run(debug=True)
