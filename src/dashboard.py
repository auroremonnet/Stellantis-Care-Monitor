import streamlit as st
import pandas as pd
import altair as alt
import time
import numpy as np

# --- CONFIGURATION & STYLE STELLANTIS ULTIME ---
st.set_page_config(
    page_title="Stellantis Engineering Lab",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Fond Application */
    .stApp { background-color: #f4f7f6; }
    
    /* STYLE DES CARTES DE CALCUL (LIVE DEBUG) */
    .calc-card {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin-bottom: 20px;
        border-top: 5px solid #ccc; /* Couleur par défaut */
    }
    
    .calc-header {
        font-size: 18px;
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 15px;
        border-bottom: 1px solid #eee;
        padding-bottom: 10px;
    }
    
    .calc-row {
        display: flex;
        justify-content: space-between;
        margin-bottom: 8px;
        font-family: 'Segoe UI', sans-serif;
    }
    
    .calc-label { color: #7f8c8d; font-size: 14px; font-weight: 600; }
    .calc-value { color: #2c3e50; font-size: 14px; font-weight: bold; font-family: 'Courier New', monospace; }
    
    .score-impact-pos { background-color: #d4edda; color: #155724; padding: 2px 8px; border-radius: 4px; font-weight: bold; font-size: 12px; }
    .score-impact-neg { background-color: #f8d7da; color: #721c24; padding: 2px 8px; border-radius: 4px; font-weight: bold; font-size: 12px; }
    .score-impact-neu { background-color: #e2e3e5; color: #383d41; padding: 2px 8px; border-radius: 4px; font-weight: bold; font-size: 12px; }

    /* BOX MATHS (Page 1) */
    .math-box {
        background-color: #ffffff;
        border: 1px solid #dcdcdc;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .math-val { font-size: 28px; font-weight: 700; color: #2c3e50; }
    .math-title { font-size: 11px; color: #888; text-transform: uppercase; letter-spacing: 1px; }

    /* Sidebar */
    section[data-testid="stSidebar"] { background-color: #0b1e3b; }
    section[data-testid="stSidebar"] * { color: #ecf0f1 !important; }
</style>
""", unsafe_allow_html=True)

# --- CHARGEMENT DATA ---
def load_data():
    files = ["session_data.csv", "../session_data.csv", "src/session_data.csv"]
    for f in files:
        try:
            df = pd.read_csv(f)
            if not df.empty: 
                if 'Score_Geo' not in df.columns: df['Score_Geo'] = 0
                if 'Score_CNN' not in df.columns: df['Score_CNN'] = 0
                return df
        except: continue
    return pd.DataFrame()

# --- SIDEBAR NAV ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/b/b2/Stellantis.svg", width=180)
    st.markdown("### CARE MONITOR ")
    st.divider()
    nav = st.radio("MODULE", [
        "1. Cockpit de Fusion (Live)",
        "2. FACS & Biométrie (Détails)",
    ])
    st.divider()
    df = load_data()
    if not df.empty:
        st.markdown("STATUS CAPTEURS")
        c1, c2 = st.columns(2)
        c1.metric("FPS", int(df['FPS'].mean()))
        c2.metric("Temp", f"{df.iloc[-1]['Temperature_Habitacle']:.0f}°")
    else: st.error("OFFLINE")

# =========================================================
# PAGE 1 : COCKPIT DE FUSION
# =========================================================
if nav == "1. Cockpit de Fusion (Live)":
    st.title("Cockpit de Supervision")
    st.markdown("Visualisation de la boucle de décision algorithmique.")
    
    placeholder = st.empty()
    while True:
        df = load_data()
        if df.empty: time.sleep(1); continue
        
        with placeholder.container():
            last = df.iloc[-1]
            
            # 1. ÉQUATION VISUELLE
            st.subheader("Calcul du Score en Temps Réel")
            c1, c2, c3, c4, c5 = st.columns([2, 0.5, 2, 0.5, 2])
            
            with c1:
                st.markdown(f"""
                <div class="math-box" style="border-top: 4px solid #3498db;">
                    <div class="math-title">GÉOMÉTRIE (POINTS)</div>
                    <div class="math-val">{last['Score_Geo']:.2f}</div>
                    <div style="font-size:12px; color:#666;">Landmarks 3D</div>
                </div>""", unsafe_allow_html=True)
            with c2: st.markdown("<h2 style='text-align:center; color:#ccc'>+</h2>", unsafe_allow_html=True)
            with c3:
                st.markdown(f"""
                <div class="math-box" style="border-top: 4px solid #e67e22;">
                    <div class="math-title">EMOTION (CNN)</div>
                    <div class="math-val">{last['Score_CNN']:.2f}</div>
                    <div style="font-size:12px; color:#666;">Deep Learning</div>
                </div>""", unsafe_allow_html=True)
            with c4: st.markdown("<h2 style='text-align:center; color:#ccc'>=</h2>", unsafe_allow_html=True)
            with c5:
                color = "#27ae60" if last['Score_Total'] > 0 else "#c0392b"
                st.markdown(f"""
                <div class="math-box" style="border-top: 4px solid {color}; background: #fff;">
                    <div class="math-title">INDEX FINAL</div>
                    <div class="math-val" style="color:{color}">{last['Score_Total']:.2f}</div>
                    <div style="font-size:12px; color:#666;">Décision</div>
                </div>""", unsafe_allow_html=True)
            
            st.divider()

            # 2. GRAPHIQUE
            st.subheader("Dynamique de Régulation")
            data_viz = df.tail(200).melt('Time_Relatif', value_vars=['Score_Geo', 'Score_CNN'], var_name='Source', value_name='Score')
            chart = alt.Chart(data_viz).mark_area(opacity=0.6).encode(
                x=alt.X('Time_Relatif', title='Temps (s)'),
                y='Score', color=alt.Color('Source', scale=alt.Scale(range=['#3498db', '#e67e22']))
            ).properties(height=300)
            st.altair_chart(chart, use_container_width=True)

        time.sleep(1)

# =========================================================
# PAGE 2 : FACS & BIOMÉTRIE (LE COEUR DE L'ANALYSE)
# =========================================================
elif nav == "2. FACS & Biométrie (Détails)":
    st.title("Analyse & Calculs")
    st.markdown("Débogueur temps réel : Affichage des règles logiques déclenchées pour chaque zone.")
    
    if df.empty: st.warning("En attente de données..."); st.stop()
    last = df.iloc[-1]
    
    # --- HELPER POUR AFFICHER UNE CARTE DE CALCUL ---
    def render_calc_card(title, observed_state, logic_rule, score_impact, color_border, icon):
        # Détermination du style d'impact
        if score_impact > 0: 
            impact_html = f"<span class='score-impact-pos'>+{score_impact} PTS (CONFORT)</span>"
        elif score_impact < 0: 
            impact_html = f"<span class='score-impact-neg'>{score_impact} PTS (INCONFORT)</span>"
        else: 
            impact_html = f"<span class='score-impact-neu'>0 PTS (NEUTRE)</span>"
            
        st.markdown(f"""
        <div class="calc-card" style="border-top-color: {color_border};">
            <div class="calc-header">{icon} {title}</div>
            <div class="calc-row">
                <span class="calc-label">État Détecté (Input) :</span>
                <span class="calc-value">{observed_state}</span>
            </div>
            <div class="calc-row">
                <span class="calc-label">Logique Appliquée :</span>
                <span class="calc-value" style="color:#d63384;">{logic_rule}</span>
            </div>
            <div style="border-top:1px dashed #eee; margin:10px 0;"></div>
            <div class="calc-row">
                <span class="calc-label">Résultat du Calcul :</span>
                {impact_html}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # --- RAFRAICHISSEMENT AUTOMATIQUE ---
    placeholder = st.empty()
    while True:
        df = load_data()
        if df.empty: time.sleep(1); continue
        last = df.iloc[-1]
        
        with placeholder.container():
            col_live, col_hist = st.columns([1.5, 2])
            
            with col_live:
                st.subheader("Derniers Calculs (Live)")
                
                # 1. YEUX
                # Logique simplifiée pour l'affichage basée sur geometry_analysis.py
                eye_rule = "EAR ≈ Baseline"
                eye_score = 0
                eye_color = "#95a5a6"
                if "Plisses" in last['Detail_Yeux']:
                    eye_rule = "EAR < 0.20 (l'oeil est plat) + Activation Orbicularis Oculi (AU6)(muscle Orbicularis = contracté =coin des yeux plissés) - "
                    eye_score = 0 # (Fait partie du sourire global souvent, mais ici neutre en score direct)
                    eye_color = "#2ecc71"
                elif "Ecarquilles" in last['Detail_Yeux']:
                    eye_rule = ": EAR > 0.35 (oeil presque rond) + Activation Levator Palpebrae (AU5)(paupière tirée vers le haut = blanc de l'oeil exposé)"
                    eye_color = "#f1c40f"
                
                render_calc_card("ANALYSE YEUX", last['Detail_Yeux'], eye_rule, eye_score, eye_color, "")
                
                # 2. SOURCILS
                brow_rule = "Dist ≈ Baseline"
                brow_score = 0
                brow_color = "#95a5a6"
                if "Fronces" in last['Detail_Sourcils']:
                    brow_rule = "Baseline - 8% (par rapport à la calibration) + Activation Corrugator Supercilii (AU4)(tire les sourcils vers le bas et le centre)"
                    brow_score = -6.0
                    brow_color = "#e74c3c"
                elif "Releves" in last['Detail_Sourcils']:
                    brow_rule = "Baseline + 8% (distance sourcil-oeil = augmente)+ Activation Frontalis (AU1 + AU2)(tire tout vers le haut)"
                    brow_color = "#f39c12"

                render_calc_card("ANALYSE SOURCILS", last['Detail_Sourcils'], brow_rule, brow_score, brow_color, "")
                
                # 3. BOUCHE
                mouth_rule = "Stable"
                mouth_score = 0
                mouth_color = "#95a5a6"
                if "Sourire" in last['Detail_Bouche']:
                    mouth_rule = "Corner_Y < Center_Y + Activation Zygomaticus Major (AU12)(relie le coin de la bouche à la pommette)"
                    mouth_score = +5.0
                    mouth_color = "#27ae60"
                elif "Grimace" in last['Detail_Bouche']:
                    mouth_rule = "Largeur > 112% Baseline + Activation Risorius (AU20) (coin de la bouche vers les oreilles/horizontale)"
                    mouth_score = -4.0
                    mouth_color = "#e74c3c"
                elif "Baillement" in last['Detail_Bouche']:
                    mouth_rule = "MAR > 0.35 + Jaw Drop (AU26)(ouvertule verticale)"
                    mouth_score = -5.0
                    mouth_color = "#8e44ad"

                render_calc_card("ANALYSE BOUCHE", last['Detail_Bouche'], mouth_rule, mouth_score, mouth_color, "")

            with col_hist:
                st.subheader("Historique & Tendances")
                
                # Tableaux de données propres
                st.markdown("Derniers Événements Détectés")
                st.dataframe(
                    df.tail(10)[['Timestamp', 'Detail_Sourcils', 'Detail_Bouche', 'Etat_Global']],
                    use_container_width=True,
                    hide_index=True
                )
                
                st.divider()
                st.markdown("Répartition des Causes d'Inconfort")
                
                # Graphique des causes d'inconfort
                df_inconf = df[df['Etat_Global'] == "INCONFORT"]
                if not df_inconf.empty:
                    # On combine Sourcils et Bouche pour voir le coupable principal
                    causes = pd.concat([df_inconf['Detail_Sourcils'], df_inconf['Detail_Bouche']])
                    causes = causes[causes.str.contains("Fronces|Grimace|Tension")] # Filtre mots clés
                    
                    if not causes.empty:
                        cause_counts = causes.value_counts().reset_index()
                        cause_counts.columns = ['Cause Anatomique', 'Occurrences']
                        
                        chart = alt.Chart(cause_counts).mark_arc(innerRadius=50).encode(
                            theta=alt.Theta("Occurrences", stack=True),
                            color=alt.Color("Cause Anatomique", scale=alt.Scale(scheme='reds')),
                            tooltip=["Cause Anatomique", "Occurrences"]
                        ).properties(height=250)
                        st.altair_chart(chart, use_container_width=True)
                    else:
                        st.info("Inconfort détecté par score global bas (Accumulation).")
                else:
                    st.success("Aucun inconfort détecté dans la session actuelle.")

        time.sleep(1)
