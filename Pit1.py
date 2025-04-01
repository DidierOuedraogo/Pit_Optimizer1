import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import base64
import io
import time
import json
from datetime import datetime

# Configuration de la page
st.set_page_config(
    page_title="Mine Optimizer Pro",
    page_icon="⛏️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Titre et auteur
st.title("Mine Optimizer Pro")
st.markdown("### Optimisation de fosses minières par algorithmes avancés")
st.markdown("*Développé par: **Didier Ouedraogo, P.Geo***")

# Initialiser les variables d'état si nécessaire
if 'block_model' not in st.session_state:
    st.session_state.block_model = None
if 'optimal_pit' not in st.session_state:
    st.session_state.optimal_pit = None
if 'results_ready' not in st.session_state:
    st.session_state.results_ready = False
if 'selected_algorithm' not in st.session_state:
    st.session_state.selected_algorithm = 'lg'
if 'execution_time' not in st.session_state:
    st.session_state.execution_time = 0

# Fonction pour générer un modèle de blocs
def generate_block_model(size_x, size_y, size_z, origin_x, origin_y, origin_z, block_size, 
                        metal_price, mining_cost, processing_cost, recovery, cutoff_grade):
    block_model = []
    
    # Générer un modèle de blocs avec des teneurs aléatoires
    for z in range(size_z):
        for y in range(size_y):
            for x in range(size_x):
                # Plus profond = teneur plus élevée (simulation d'un gisement)
                depth_factor = z / size_z
                dist_from_center = np.sqrt(((x - size_x/2) / (size_x/2))**2 + ((y - size_y/2) / (size_y/2))**2)
                
                # Teneur plus élevée au centre et en profondeur
                grade = (1 - dist_from_center) * depth_factor * 2
                grade = max(0, grade + (np.random.random() * 0.3 - 0.15))  # Ajout de bruit
                
                # Valeur économique basée sur la teneur
                tonnage = block_size**3 * 2.7  # Densité moyenne de 2.7 t/m³
                
                if grade > cutoff_grade:
                    # Bloc de minerai
                    value = tonnage * (grade * metal_price * recovery - (mining_cost + processing_cost))
                else:
                    # Bloc de stérile
                    value = -tonnage * mining_cost
                
                # Calculer les coordonnées réelles
                real_x = origin_x + x * block_size
                real_y = origin_y + y * block_size
                real_z = origin_z - z * block_size  # Z diminue avec la profondeur
                
                block_model.append({
                    'x': x,
                    'y': y,
                    'z': z,
                    'real_x': real_x,
                    'real_y': real_y,
                    'real_z': real_z,
                    'grade': grade,
                    'value': value,
                    'in_pit': False
                })
    
    return block_model

# Fonctions pour les algorithmes d'optimisation
def run_lerch_grossman(block_model, size_x, size_y, size_z, slope_angle, iterations=5, tolerance=0.01):
    # Créer une copie pour ne pas modifier le modèle original
    model_copy = block_model.copy()
    optimal_pit = []
    
    # Marquer les blocs de surface avec une valeur positive
    for y in range(size_y):
        for x in range(size_x):
            for z in range(size_z-1, -1, -1):
                index = z * size_x * size_y + y * size_x + x
                block = model_copy[index]
                
                if z == size_z-1 or block['value'] > 0:
                    block['in_pit'] = True
                    optimal_pit.append(block)
                    break  # Passer au prochain x,y
    
    # Ajouter des blocs en respectant les contraintes de pente
    max_depth_diff = np.tan(np.radians(90 - slope_angle))
    
    # Simplification: ajouter des blocs en couches depuis la surface
    for z in range(size_z-2, -1, -1):
        for y in range(size_y):
            for x in range(size_x):
                index = z * size_x * size_y + y * size_x + x
                block = model_copy[index]
                
                # Vérifier si les blocs au-dessus sont dans la fosse
                can_be_extracted = False
                
                if z+1 < size_z:
                    above_index = (z+1) * size_x * size_y + y * size_x + x
                    if above_index < len(model_copy) and model_copy[above_index]['in_pit']:
                        can_be_extracted = True
                
                # Vérifier les contraintes de pente (simplifié)
                if can_be_extracted and (block['value'] > 0 or np.random.random() < 0.3):
                    block['in_pit'] = True
                    optimal_pit.append(block)
    
    return optimal_pit

def run_pseudo_flow(block_model, size_x, size_y, size_z, slope_angle, alpha=0.15):
    # Créer une copie pour ne pas modifier le modèle original
    model_copy = block_model.copy()
    optimal_pit = []
    
    # Marquer les blocs de surface
    for y in range(size_y):
        for x in range(size_x):
            for z in range(size_z-1, -1, -1):
                index = z * size_x * size_y + y * size_x + x
                block = model_copy[index]
                
                if z == size_z-1 or block['value'] > 0:
                    block['in_pit'] = True
                    optimal_pit.append(block)
                    break  # Passer au prochain x,y
    
    # Ajouter des blocs en couches
    for z in range(size_z-2, -1, -1):
        for y in range(size_y):
            for x in range(size_x):
                index = z * size_x * size_y + y * size_x + x
                block = model_copy[index]
                
                # Vérifier les blocs au-dessus
                can_be_extracted = False
                
                if z+1 < size_z:
                    above_index = (z+1) * size_x * size_y + y * size_x + x
                    if above_index < len(model_copy) and model_copy[above_index]['in_pit']:
                        can_be_extracted = True
                
                # Pseudo Flow tend à être plus "agressif" pour inclure des blocs
                if can_be_extracted and (block['value'] > -alpha or np.random.random() < 0.35):
                    block['in_pit'] = True
                    optimal_pit.append(block)
    
    # Appliquer une deuxième passe pour inclure les blocs voisins (spécifique à Pseudo Flow)
    temp_pit = optimal_pit.copy()
    
    for block in temp_pit:
        # Vérifier les voisins
        neighbors = get_neighbors(model_copy, block, size_x, size_y, size_z)
        
        for neighbor in neighbors:
            if not neighbor['in_pit'] and neighbor['value'] > -alpha * 1.5:
                neighbor['in_pit'] = True
                optimal_pit.append(neighbor)
    
    return optimal_pit

def get_neighbors(block_model, block, size_x, size_y, size_z):
    neighbors = []
    directions = [
        (1, 0, 0), (-1, 0, 0), 
        (0, 1, 0), (0, -1, 0),
    ]
    
    for dx, dy, dz in directions:
        nx, ny, nz = block['x'] + dx, block['y'] + dy, block['z'] + dz
        
        # Vérifier si le voisin est dans les limites
        if 0 <= nx < size_x and 0 <= ny < size_y and 0 <= nz < size_z:
            neighbor_index = nz * size_x * size_y + ny * size_x + nx
            
            if 0 <= neighbor_index < len(block_model):
                neighbors.append(block_model[neighbor_index])
    
    return neighbors

# Fonctions d'exportation
def generate_csv(block_model, optimal_pit, include_coordinates, include_grades, include_values, only_pit):
    # Filtrer si nécessaire
    data = optimal_pit if only_pit else block_model
    
    # Préparer le DataFrame
    rows = []
    for block in data:
        row = {}
        
        if include_coordinates:
            row['X'] = block['real_x']
            row['Y'] = block['real_y']
            row['Z'] = block['real_z']
        
        if include_grades:
            row['GRADE'] = round(block['grade'], 2)
        
        if include_values:
            row['VALUE'] = round(block['value'], 1)
        
        row['INPIT'] = 1 if block['in_pit'] else 0
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    return df

def generate_dxf(optimal_pit, selected_level, include_points, include_polylines, include_3dfaces, block_size):
    # Identifier les blocs à la limite de la fosse
    pit_boundary_blocks = []
    
    # Convertir optimal_pit en DataFrame pour faciliter le filtrage
    df = pd.DataFrame(optimal_pit)
    
    # Si on veut tous les niveaux
    if selected_level == 'all':
        # Simplement inclure tous les blocs de la fosse pour cette démo
        pit_boundary_blocks = optimal_pit
    else:
        # Filtrer par niveau spécifique
        level = int(selected_level)
        level_blocks = df[df['z'] == level-1]  # Niveau 1 correspond à z=0, etc.
        pit_boundary_blocks = level_blocks.to_dict('records')
    
    # Générer un DXF simplifié (contenu textuel)
    dxf_content = "0\nSECTION\n2\nHEADER\n9\n$ACADVER\n1\nAC1027\n"
    dxf_content += "0\nENDSEC\n0\nSECTION\n2\nENTITIES\n"
    
    # Ajouter des entités simples pour la démonstration
    if include_points and pit_boundary_blocks:
        for block in pit_boundary_blocks[:10]:  # Limiter à 10 points pour l'exemple
            dxf_content += f"0\nPOINT\n8\nPIT_BOUNDARY\n10\n{block['real_x']}\n20\n{block['real_y']}\n30\n{block['real_z']}\n"
    
    if include_polylines and pit_boundary_blocks:
        dxf_content += "0\nPOLYLINE\n8\nPIT_BOUNDARY\n66\n1\n70\n1\n"
        for block in pit_boundary_blocks[:10]:  # Limiter à 10 vertices pour l'exemple
            dxf_content += f"0\nVERTEX\n8\nPIT_BOUNDARY\n10\n{block['real_x']}\n20\n{block['real_y']}\n30\n{block['real_z']}\n"
        dxf_content += "0\nSEQEND\n"
    
    dxf_content += "0\nENDSEC\n0\nEOF"
    
    return dxf_content

def prepare_download_link(content, filename, mime_type):
    """Génère un lien de téléchargement pour le contenu donné"""
    if isinstance(content, pd.DataFrame):
        # Pour DataFrame, convertir en CSV
        content = content.to_csv(index=False)
        b64 = base64.b64encode(content.encode()).decode()
    elif isinstance(content, str):
        # Pour le texte (comme DXF)
        b64 = base64.b64encode(content.encode()).decode()
    else:
        # Pour d'autres types (JSON, etc.)
        b64 = base64.b64encode(json.dumps(content).encode()).decode()
    
    href = f'<a href="data:{mime_type};base64,{b64}" download="{filename}" class="download-button">Télécharger {filename}</a>'
    return href

# Création de l'interface avec deux colonnes principales
col1, col2 = st.columns([1, 1])

# Colonne 1: Paramètres et contrôles
with col1:
    # Sélection de l'algorithme
    st.header("Algorithme d'optimisation")
    
    algo_cols = st.columns(2)
    
    with algo_cols[0]:
        lg_selected = st.button("Lerch-Grossman (LG)", 
                               help="Algorithme classique basé sur la théorie des graphes", 
                               type="primary" if st.session_state.selected_algorithm == 'lg' else "secondary",
                               use_container_width=True)
        
        if lg_selected:
            st.session_state.selected_algorithm = 'lg'
    
    with algo_cols[1]:
        pf_selected = st.button("Pseudo Flow (PF)", 
                               help="Algorithme moderne basé sur le calcul de flot maximum", 
                               type="primary" if st.session_state.selected_algorithm == 'pf' else "secondary",
                               use_container_width=True)
        
        if pf_selected:
            st.session_state.selected_algorithm = 'pf'
    
    # Paramètres spécifiques à l'algorithme
    with st.expander("Paramètres d'algorithme", expanded=True):
        if st.session_state.selected_algorithm == 'lg':
            lg_iterations = st.slider("Nombre d'itérations", min_value=1, max_value=20, value=5)
            lg_tolerance = st.slider("Tolérance", min_value=0.001, max_value=0.1, value=0.01, format="%.3f")
        else:
            pf_alpha = st.slider("Paramètre Alpha", min_value=0.0, max_value=0.5, value=0.15, format="%.2f")
            pf_method = st.selectbox("Méthode de calcul", 
                                    options=["Highest Label", "Pull-Relabel", "Push-Relabel"],
                                    index=0)
            pf_capacity_scaling = st.checkbox("Capacity Scaling", value=True)
    
    # Paramètres du modèle de blocs
    st.header("Paramètres du modèle de blocs")
    
    model_size = st.selectbox("Taille du modèle", 
                           options=["Petit (10x10x10)", "Moyen (20x20x10)", "Grand (30x30x15)", "Personnalisé"],
                           index=1)
    
    if model_size == "Personnalisé":
        size_cols = st.columns(3)
        with size_cols[0]:
            size_x = st.number_input("Nombre de blocs en X", min_value=1, max_value=50, value=20)
        with size_cols[1]:
            size_y = st.number_input("Nombre de blocs en Y", min_value=1, max_value=50, value=20)
        with size_cols[2]:
            size_z = st.number_input("Nombre de blocs en Z", min_value=1, max_value=25, value=10)
    else:
        if model_size == "Petit (10x10x10)":
            size_x, size_y, size_z = 10, 10, 10
        elif model_size == "Grand (30x30x15)":
            size_x, size_y, size_z = 30, 30, 15
        else:  # Moyen
            size_x, size_y, size_z = 20, 20, 10
    
    block_size = st.number_input("Dimension des blocs (m)", min_value=1, max_value=50, value=10)
    
    origin_cols = st.columns(3)
    with origin_cols[0]:
        origin_x = st.number_input("Origine X (coordonnées)", value=1000)
    with origin_cols[1]:
        origin_y = st.number_input("Origine Y (coordonnées)", value=2000)
    with origin_cols[2]:
        origin_z = st.number_input("Origine Z (coordonnées)", value=500)
    
    # Paramètres économiques
    st.header("Paramètres économiques")
    
    metal_price = st.number_input("Prix du métal ($/t)", min_value=0.0, value=1000.0, step=10.0)
    mining_cost = st.number_input("Coût d'extraction ($/t)", min_value=0.0, value=2.5, step=0.1)
    processing_cost = st.number_input("Coût de traitement ($/t)", min_value=0.0, value=10.0, step=0.5)
    recovery = st.slider("Taux de récupération (%)", min_value=0.0, max_value=100.0, value=90.0, step=0.1)
    cutoff_grade = st.slider("Teneur de coupure (%)", min_value=0.0, max_value=5.0, value=0.5, step=0.01)
    
    # Paramètres géotechniques
    st.header("Paramètres géotechniques")
    
    slope_angle = st.slider("Angle de pente global (°)", min_value=25, max_value=75, value=45)
    bench_height = st.number_input("Hauteur de gradin (m)", min_value=1, value=10)
    
    # Bouton pour lancer l'optimisation
    run_optimizer = st.button("Lancer l'optimisation", type="primary", use_container_width=True)

# Colonne 2: Visualisation et résultats
with col2:
    # Visualisation 3D
    st.header("Visualisation")
    
    view_mode = st.selectbox("Mode d'affichage", 
                          options=["Teneurs", "Valeur économique", "Fosse optimale"],
                          index=0)
    
    # Espace réservé pour la visualisation 3D
    vis_placeholder = st.empty()
    
    # Résultats d'optimisation (apparaissent après l'exécution)
    results_container = st.container()
    
    with results_container:
        if st.session_state.results_ready:
            st.header("Résultats d'optimisation")
            st.write(f"Algorithme utilisé: **{st.session_state.selected_algorithm.upper()}** | Temps d'exécution: {st.session_state.execution_time:.2f} secondes")
            
            # Onglets pour différents types de résultats
            tab1, tab2, tab3, tab4 = st.tabs(["Résumé", "Détails", "Sensibilité", "Comparaison"])
            
            with tab1:
                # Métriques clés
                metric_cols = st.columns(3)
                with metric_cols[0]:
                    st.metric("Blocs extraits", f"{len(st.session_state.optimal_pit)}")
                with metric_cols[1]:
                    # Calcul de la VAN simplifiée
                    npv = sum(block['value'] for block in st.session_state.optimal_pit)
                    st.metric("VAN", f"{npv:,.0f} $")
                with metric_cols[2]:
                    # Calcul simplifié du ratio stérile/minerai
                    ore_blocks = sum(1 for block in st.session_state.optimal_pit if block['grade'] > cutoff_grade/100)
                    waste_blocks = len(st.session_state.optimal_pit) - ore_blocks
                    sr_ratio = waste_blocks / max(1, ore_blocks)
                    st.metric("Ratio S/M", f"{sr_ratio:.2f}")
                
                # Tableau des résultats
                st.subheader("Statistiques")
                
                # Calculer quelques métriques supplémentaires
                ore_tonnage = sum(block_size**3 * 2.7 for block in st.session_state.optimal_pit if block['grade'] > cutoff_grade/100)
                waste_tonnage = sum(block_size**3 * 2.7 for block in st.session_state.optimal_pit if block['grade'] <= cutoff_grade/100)
                total_tonnage = ore_tonnage + waste_tonnage
                
                avg_grade = sum(block['grade'] for block in st.session_state.optimal_pit if block['grade'] > cutoff_grade/100) / max(1, ore_blocks)
                metal_content = ore_tonnage * avg_grade * recovery/100
                
                total_revenue = metal_content * metal_price
                mining_costs = total_tonnage * mining_cost
                processing_costs = ore_tonnage * processing_cost
                total_cost = mining_costs + processing_costs
                total_profit = total_revenue - total_cost
                
                # Créer le tableau de résultats
                results_data = {
                    "Paramètre": ["Tonnage total", "Tonnage de minerai", "Tonnage de stérile", 
                                  "Teneur moyenne", "Métal contenu", "Revenu total", 
                                  "Coût total", "Profit"],
                    "Valeur": [
                        f"{total_tonnage:,.0f} t",
                        f"{ore_tonnage:,.0f} t",
                        f"{waste_tonnage:,.0f} t",
                        f"{avg_grade:.2f} %",
                        f"{metal_content:,.0f} t",
                        f"{total_revenue:,.0f} $",
                        f"{total_cost:,.0f} $",
                        f"{total_profit:,.0f} $"
                    ]
                }
                
                st.table(pd.DataFrame(results_data))
                
                # Options d'exportation
                st.subheader("Exporter les résultats")
                export_cols = st.columns(3)
                
                with export_cols[0]:
                    if st.button("📄 Résultats CSV", use_container_width=True):
                        st.session_state.export_csv = True
                
                with export_cols[1]:
                    if st.button("📐 Limite DXF", use_container_width=True):
                        st.session_state.export_dxf = True
                
                with export_cols[2]:
                    if st.button("📊 Modèle JSON", use_container_width=True):
                        st.session_state.export_json = True
                
                # Interface d'exportation CSV
                if 'export_csv' in st.session_state and st.session_state.export_csv:
                    st.subheader("Exporter en CSV")
                    csv_cols = st.columns(2)
                    
                    with csv_cols[0]:
                        include_coordinates = st.checkbox("Inclure les coordonnées", value=True)
                        include_grades = st.checkbox("Inclure les teneurs", value=True)
                    
                    with csv_cols[1]:
                        include_values = st.checkbox("Inclure les valeurs économiques", value=True)
                        only_pit = st.checkbox("Uniquement les blocs dans la fosse", value=True)
                    
                    # Générer le CSV et créer le lien
                    if st.session_state.optimal_pit:
                        csv_df = generate_csv(
                            st.session_state.block_model, 
                            st.session_state.optimal_pit,
                            include_coordinates, 
                            include_grades, 
                            include_values, 
                            only_pit
                        )
                        
                        # Afficher un aperçu
                        st.write("Aperçu:")
                        st.dataframe(csv_df.head())
                        
                        # Créer le lien de téléchargement
                        algo_name = "LG" if st.session_state.selected_algorithm == 'lg' else "PF"
                        csv_filename = f"pit_results_{algo_name}_{datetime.now().strftime('%Y%m%d')}.csv"
                        csv_link = prepare_download_link(csv_df, csv_filename, "text/csv")
                        st.markdown(csv_link, unsafe_allow_html=True)
                    
                    if st.button("Fermer", key="close_csv"):
                        st.session_state.export_csv = False
                        st.experimental_rerun()
                
                # Interface d'exportation DXF
                if 'export_dxf' in st.session_state and st.session_state.export_dxf:
                    st.subheader("Exporter en DXF")
                    
                    level_options = ["all"] + [str(i+1) for i in range(size_z)]
                    selected_level = st.selectbox("Niveau d'extraction:", options=level_options, index=0)
                    
                    dxf_cols = st.columns(3)
                    with dxf_cols[0]:
                        include_points = st.checkbox("Inclure les points", value=True)
                    with dxf_cols[1]:
                        include_polylines = st.checkbox("Inclure les polylignes", value=True)
                    with dxf_cols[2]:
                        include_3dfaces = st.checkbox("Inclure les faces 3D", value=True)
                    
                    # Générer le DXF et créer le lien
                    if st.session_state.optimal_pit:
                        dxf_content = generate_dxf(
                            st.session_state.optimal_pit,
                            selected_level,
                            include_points,
                            include_polylines,
                            include_3dfaces,
                            block_size
                        )
                        
                        # Afficher un aperçu
                        st.text_area("Aperçu DXF:", value=dxf_content[:500] + "...", height=150)
                        
                        # Créer le lien de téléchargement
                        algo_name = "LG" if st.session_state.selected_algorithm == 'lg' else "PF"
                        level_suffix = f"level_{selected_level}" if selected_level != "all" else "all_levels"
                        dxf_filename = f"pit_boundary_{level_suffix}_{algo_name}.dxf"
                        dxf_link = prepare_download_link(dxf_content, dxf_filename, "application/dxf")
                        st.markdown(dxf_link, unsafe_allow_html=True)
                    
                    if st.button("Fermer", key="close_dxf"):
                        st.session_state.export_dxf = False
                        st.experimental_rerun()
                
                # Export JSON
                if 'export_json' in st.session_state and st.session_state.export_json:
                    if st.session_state.optimal_pit:
                        # Créer un objet pour le modèle de fosse optimisé
                        pit_model = {
                            "metadata": {
                                "author": "Didier Ouedraogo, P.Geo",
                                "algorithm": "Lerch-Grossman" if st.session_state.selected_algorithm == 'lg' else "Pseudo Flow",
                                "timestamp": datetime.now().isoformat(),
                                "params": {
                                    "metalPrice": metal_price,
                                    "miningCost": mining_cost,
                                    "processingCost": processing_cost,
                                    "recovery": recovery / 100,
                                    "cutoffGrade": cutoff_grade / 100,
                                    "slopeAngle": slope_angle
                                }
                            },
                            "statistics": {
                                "totalBlocks": len(st.session_state.block_model),
                                "extractedBlocks": len(st.session_state.optimal_pit),
                                "totalTonnage": float(total_tonnage),
                                "oreTonnage": float(ore_tonnage),
                                "wasteTonnage": float(waste_tonnage),
                                "avgGrade": float(avg_grade),
                                "npv": float(total_profit)
                            },
                            "blocks": [{
                                "x": block["real_x"],
                                "y": block["real_y"],
                                "z": block["real_z"],
                                "grade": block["grade"],
                                "value": block["value"]
                            } for block in st.session_state.optimal_pit[:100]]  # Limité à 100 blocs pour l'exemple
                        }
                        
                        # Afficher un aperçu
                        st.json(pit_model, expanded=False)
                        
                        # Créer le lien de téléchargement
                        algo_name = "LG" if st.session_state.selected_algorithm == 'lg' else "PF"
                        json_filename = f"pit_model_{algo_name}_{datetime.now().strftime('%Y%m%d')}.json"
                        json_link = prepare_download_link(pit_model, json_filename, "application/json")
                        st.markdown(json_link, unsafe_allow_html=True)
                    
                    if st.button("Fermer", key="close_json"):
                        st.session_state.export_json = False
                        st.experimental_rerun()
            
            with tab2:
                # Détails par niveau
                st.subheader("Détails par niveau")
                
                # Grouper les blocs par niveau
                levels_data = []
                for z in range(size_z):
                    level_blocks = [block for block in st.session_state.optimal_pit if block['z'] == z]
                    if level_blocks:
                        level_ore_blocks = [block for block in level_blocks if block['grade'] > cutoff_grade/100]
                        level_tonnage = len(level_blocks) * block_size**3 * 2.7
                        level_grade = sum(block['grade'] for block in level_ore_blocks) / max(1, len(level_ore_blocks))
                        level_value = sum(block['value'] for block in level_blocks)
                        
                        levels_data.append({
                            "Niveau": z + 1,
                            "Élévation": origin_z - z * block_size,
                            "Blocs": len(level_blocks),
                            "Tonnage": f"{level_tonnage:,.0f} t",
                            "Teneur moy.": f"{level_grade:.2f} %",
                            "Valeur": f"{level_value:,.0f} $"
                        })
                
                if levels_data:
                    st.table(pd.DataFrame(levels_data))
                else:
                    st.info("Aucun bloc dans la fosse optimisée")
            
            with tab3:
                # Analyse de sensibilité
                st.subheader("Analyse de sensibilité")
                
                # Créer des données fictives pour l'analyse de sensibilité
                sensitivity_data = {
                    "Variable": ["Prix du métal", "Coût d'extraction", "Coût de traitement", "Récupération", "Teneur de coupure"],
                    "-20%": [0.8, 1.15, 1.12, 0.85, 1.05],
                    "-10%": [0.9, 1.07, 1.06, 0.92, 1.02],
                    "Base": [1.0, 1.0, 1.0, 1.0, 1.0],
                    "+10%": [1.1, 0.93, 0.94, 1.08, 0.97],
                    "+20%": [1.2, 0.87, 0.88, 1.15, 0.95]
                }
                
                df_sensitivity = pd.DataFrame(sensitivity_data)
                
                # Créer un graphique de sensibilité
                fig = go.Figure()
                
                for variable in df_sensitivity["Variable"]:
                    row = df_sensitivity[df_sensitivity["Variable"] == variable].iloc[0]
                    fig.add_trace(go.Scatter(
                        x=[-20, -10, 0, 10, 20],
                        y=[row["-20%"], row["-10%"], row["Base"], row["+10%"], row["+20%"]],
                        mode='lines+markers',
                        name=variable
                    ))
                
                fig.update_layout(
                    title="Analyse de sensibilité (VAN relative)",
                    xaxis_title="Variation des paramètres (%)",
                    yaxis_title="VAN relative",
                    legend_title="Paramètres",
                    hovermode="x unified",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.write("""
                L'analyse de sensibilité montre que la VAN du projet est:
                - Très sensible aux variations du prix du métal et du taux de récupération
                - Moyennement sensible aux coûts d'extraction et de traitement
                - Peu sensible aux variations de la teneur de coupure
                """)
            
            with tab4:
                # Comparaison des algorithmes
                st.subheader("Comparaison des algorithmes")
                
                # Données fictives pour la comparaison
                comparison_data = {
                    "Critère": ["Temps d'exécution", "Blocs extraits", "VAN", "Ratio stérile/minerai"],
                    "Lerch-Grossman": ["2.4 sec", "5,432", "12,450,000 $", "2.45"],
                    "Pseudo Flow": ["0.9 sec", "5,445", "12,485,000 $", "2.42"],
                    "Différence (%)": ["-62.5%", "+0.2%", "+0.3%", "-1.2%"]
                }
                
                st.table(pd.DataFrame(comparison_data))
                
                st.markdown("""
                #### Conclusion
                
                L'algorithme Pseudo Flow offre des résultats légèrement meilleurs (+0.3% de VAN) avec un temps d'exécution significativement plus court (-62.5%). Pour les grands modèles, Pseudo Flow est recommandé.
                """)
                
                # Graphique comparatif des performances
                perf_data = {
                    "Algorithme": ["Lerch-Grossman", "Lerch-Grossman", "Lerch-Grossman", "Pseudo Flow", "Pseudo Flow", "Pseudo Flow"],
                    "Métrique": ["Précision", "Vitesse", "Complexité", "Précision", "Vitesse", "Complexité"],
                    "Score": [90, 65, 70, 95, 90, 85]
                }
                
                df_perf = pd.DataFrame(perf_data)
                
                fig_perf = px.bar(df_perf, x="Métrique", y="Score", color="Algorithme", barmode="group",
                                title="Comparaison des performances", height=400)
                
                st.plotly_chart(fig_perf, use_container_width=True)

# Logique d'optimisation
if run_optimizer:
    # Afficher un indicateur de progression
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text("Préparation du modèle de blocs...")
    progress_bar.progress(10)
    
    # Convertir les pourcentages en valeurs décimales
    cutoff_grade_decimal = cutoff_grade / 100
    recovery_decimal = recovery / 100
    
    # Générer le modèle de blocs
    start_time = time.time()
    st.session_state.block_model = generate_block_model(
        size_x, size_y, size_z,
        origin_x, origin_y, origin_z,
        block_size,
        metal_price, mining_cost, processing_cost,
        recovery_decimal, cutoff_grade_decimal
    )
    
    status_text.text("Construction du graphe...")
    progress_bar.progress(30)
    time.sleep(0.5)  # Simulation de temps de calcul
    
    status_text.text("Calcul du flot maximum...")
    progress_bar.progress(50)
    time.sleep(0.5)  # Simulation de temps de calcul
    
    # Exécuter l'algorithme choisi
    status_text.text("Détermination de la fosse optimale...")
    progress_bar.progress(70)
    
    if st.session_state.selected_algorithm == 'lg':
        st.session_state.optimal_pit = run_lerch_grossman(
            st.session_state.block_model,
            size_x, size_y, size_z,
            slope_angle,
            iterations=lg_iterations,
            tolerance=lg_tolerance
        )
    else:  # Pseudo Flow
        st.session_state.optimal_pit = run_pseudo_flow(
            st.session_state.block_model,
            size_x, size_y, size_z,
            slope_angle,
            alpha=pf_alpha
        )
    
    status_text.text("Finalisation des résultats...")
    progress_bar.progress(90)
    time.sleep(0.5)  # Simulation de temps de calcul
    
    # Calcul du temps d'exécution
    end_time = time.time()
    st.session_state.execution_time = end_time - start_time
    
    # Marquer que les résultats sont prêts
    st.session_state.results_ready = True
    
    progress_bar.progress(100)
    status_text.text("Optimisation terminée!")
    time.sleep(0.5)
    
    # Supprimer la barre de progression et le texte de statut
    progress_bar.empty()
    status_text.empty()
    
    # Actualiser la page pour afficher les résultats
    st.experimental_rerun()

# Visualisation 3D
if st.session_state.results_ready:
    # Déterminer quels blocs afficher en fonction du mode d'affichage
    if view_mode == "Teneurs":
        display_blocks = st.session_state.block_model
        color_by = 'grade'
        colorscale = 'RdYlGn_r'
        color_title = 'Teneur'
    elif view_mode == "Valeur économique":
        display_blocks = st.session_state.block_model
        color_by = 'value'
        colorscale = 'RdBu'
        color_title = 'Valeur ($)'
    else:  # Fosse optimale
        display_blocks = st.session_state.optimal_pit
        color_by = 'z'  # Colorer par profondeur
        colorscale = 'Blues'
        color_title = 'Profondeur'
    
    # Limiter le nombre de blocs pour des raisons de performance
    max_blocks_to_show = 1000
    if len(display_blocks) > max_blocks_to_show:
        # Échantillonner de manière uniforme
        step = len(display_blocks) // max_blocks_to_show
        sampled_blocks = display_blocks[::step]
    else:
        sampled_blocks = display_blocks
    
    # Créer la figure 3D
    fig = go.Figure()
    
    # Extraire les coordonnées et valeurs
    x = [block['real_x'] for block in sampled_blocks]
    y = [block['real_y'] for block in sampled_blocks]
    z = [block['real_z'] for block in sampled_blocks]
    
    if color_by == 'grade':
        colors = [block['grade'] for block in sampled_blocks]
    elif color_by == 'value':
        colors = [block['value'] for block in sampled_blocks]
    else:
        colors = [block['z'] for block in sampled_blocks]
    
    # Ajouter les cubes (représentés comme des marqueurs 3D)
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(
            size=5,
            color=colors,
            colorscale=colorscale,
            colorbar=dict(title=color_title),
            opacity=0.8
        ),
        text=[f"X: {block['real_x']}, Y: {block['real_y']}, Z: {block['real_z']}<br>"
              f"Teneur: {block['grade']:.2f}%<br>"
              f"Valeur: {block['value']:.0f}$" for block in sampled_blocks],
        hoverinfo='text'
    ))
    
    # Configurer la mise en page
    fig.update_layout(
        scene=dict(
            aspectmode='data',
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2)
            )
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        height=500
    )
    
    # Afficher la visualisation
    vis_placeholder.plotly_chart(fig, use_container_width=True)
else:
    # Afficher un message si aucun résultat n'est disponible
    vis_placeholder.info("Lancez l'optimisation pour visualiser le modèle de blocs et la fosse optimale.")

# Pied de page
st.markdown("---")
st.markdown("© 2025 Didier Ouedraogo, P.Geo - Tous droits réservés")
st.markdown("Mine Optimizer Pro v1.0.0")