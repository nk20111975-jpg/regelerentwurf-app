import streamlit as st
import numpy as np
import control as ct
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def transfer_function_to_latex(tf):
    """Konvertiert eine control.tf Übertragungsfunktion in einen LaTeX-String."""
    def poly_to_latex(poly):
        deg = len(poly) - 1
        terms = []
        for i, coef in enumerate(poly):
            power = deg - i
            if abs(coef) < 1e-5:
                continue
            # Koeffizient formatieren
            coef_str = f"{coef:.2f}".rstrip('0').rstrip('.') if not float(coef).is_integer() else f"{int(coef)}"
            if power == 0:
                terms.append(coef_str)
            elif power == 1:
                terms.append(f"{coef_str if coef_str != '1' else ''}s")
            else:
                terms.append(f"{coef_str if coef_str != '1' else ''}s^{{{power}}}")
        return " + ".join(terms) if terms else "0"

    num = tf.num[0][0]
    den = tf.den[0][0]
    return f"\\frac{{{poly_to_latex(num)}}}{{{poly_to_latex(den)}}}"

def transfer_function_to_time_constant_latex(tf):
    """Konvertiert eine Übertragungsfunktion in die Zeitkonstantenform (LaTeX)."""
    import numpy as np
    
    # Pole und Nullstellen extrahieren
    zeros = tf.zeros()
    poles = tf.poles()
    
    # Koeffizienten der Polynome holen
    num_poly = tf.num[0][0]
    den_poly = tf.den[0][0]
    
    # Anzahl der I-Anteile (Nullstellen im Ursprung) bestimmen
    num_zeros_at_origin = sum(1 for z in zeros if abs(z) < 1e-5)
    den_poles_at_origin = sum(1 for p in poles if abs(p) < 1e-5)
    
    # Relevante Pole/Nullstellen abseits des Ursprungs filtern
    filtered_zeros = [z for z in zeros if abs(z) >= 1e-5]
    filtered_poles = [p for p in poles if abs(p) >= 1e-5]
    
    # Berechnung des Verstärkungsfaktors K der Zeitkonstantenform
    # Verwende das Verhältnis der jeweils kleinsten Koeffizienten ungleich Null
    k_num = num_poly[-1 - num_zeros_at_origin]
    k_den = den_poly[-1 - den_poles_at_origin]
    K_bode = k_num / k_den

    def build_product_string(roots):
        """Baut das Produkt der Terme (1 + T*s) bzw. (s^2/w^2 + 2d/w*s + 1) zusammen."""
        terms = []
        # Komplexe Wurzeln paarweise zusammenfassen, reelle einzeln verarbeiten
        used = set()
        for idx, r in enumerate(roots):
            if idx in used:
                continue
                
            # Reelle Wurzel
            if abs(r.imag) < 1e-5:
                T = -1.0 / r.real
                T_str = f"{T:.2f}".rstrip('0').rstrip('.') if not float(T).is_integer() else f"{int(T)}"
                # Vorzeichenkorrektur für stabile/instabile Terme
                sign = "+" if T >= 0 else "-"
                terms.append(f"(1 {sign} {abs(float(T_str))}s)")
                used.add(idx)
            else:
                # Komplex konjugiertes Paar suchen
                for jdx, other in enumerate(roots):
                    if jdx not in used and jdx != idx and abs(r.real - other.real) < 1e-5 and abs(r.imag + other.imag) < 1e-5:
                        # Berechne Kennkreisfrequenz w0 und Dämpfung d
                        w0 = np.abs(r)
                        d = -r.real / w0
                        
                        w0_sq_inv = 1.0 / (w0**2)
                        two_d_w0 = 2.0 * d / w0
                        
                        w0_str = f"{w0_sq_inv:.3f}".rstrip('0').rstrip('.')
                        d_str = f"{two_d_w0:.3f}".rstrip('0').rstrip('.')
                        
                        terms.append(f"\\left({w0_str}s^2 + {d_str}s + 1\\right)")
                        used.add(idx)
                        used.add(jdx)
                        break
        
        # Falls keine Terme übrig sind, bleibt eine 1 stehen
        return "".join(terms) if terms else "1"

    # Zähler- und Nenner-Strings generieren
    num_str = build_product_string(filtered_zeros)
    den_str = build_product_string(filtered_poles)
    
    # Zusätzliche s-Faktoren für Ursprungselemente einbauen
    if num_zeros_at_origin > 0:
        s_factor = f"s^{num_zeros_at_origin}" if num_zeros_at_origin > 1 else "s"
        num_str = s_factor + num_str
    if den_poles_at_origin > 0:
        s_factor = f"s^{den_poles_at_origin}" if den_poles_at_origin > 1 else "s"
        den_str = s_factor + den_str
        
    # K formatieren
    K_str = f"{K_bode:.2f}".rstrip('0').rstrip('.') if not float(K_bode).is_integer() else f"{int(K_bode)}"
    
    # Gesamter LaTeX-Bruch
    if K_str == "1" and num_str != "1":
        return f"\\frac{{{num_str}}}{{{den_str}}}"
    elif K_str == "-1" and num_str != "1":
        return f"-\\frac{{{num_str}}}{{{den_str}}}"
    else:
        return f"{K_str} \\cdot \\frac{{{num_str}}}{{{den_str}}}"

def regler_to_time_constant_latex(Kp, Tn, Tv, i_aktiv, d_aktiv):
    """Baut die Zeitkonstantenform des Reglers direkt aus den bekannten Parametern."""
    # Helfer zum Formatieren von Fließkommazahlen (entfernt unnötige Nullen)
    fmt = lambda val: f"{val:.2f}".rstrip('0').rstrip('.') if not float(val).is_integer() else f"{int(val)}"
    
    Kp_str = fmt(Kp)
    Tn_str = fmt(Tn)
    Tv_str = fmt(Tv)
    
    # 1. Reiner P-Regler
    if not i_aktiv and not d_aktiv:
        return f"{Kp_str}"
    
    # 2. PI-Regler: Kp * (1 + Tn*s) / (Tn*s)
    elif i_aktiv and not d_aktiv:
        return f"{Kp_str} \\cdot \\frac{{1 + {Tn_str}s}}{{{Tn_str}s}}"
    
    # 3. PD-Regler: Kp * (1 + Tv*s)
    elif not i_aktiv and d_aktiv:
        return f"{Kp_str} \\cdot (1 + {Tv_str}s)"
    
    # 4. Idealisiertes PID-Glied: Kp * (1 + Tn*s) * (1 + Tv*s) / (Tn*s)
    else:
        return f"{Kp_str} \\cdot \\frac{{(1 + {Tn_str}s)(1 + {Tv_str}s)}}{{{Tn_str}s}}"

# Konfiguration der Streamlit-Seite
st.set_page_config(page_title="Regelungstechnik Sandbox", layout="wide")
st.title("Reglerentwurf")

# ==========================================
# SIDEBAR: EINGABEPARAMETER
# ==========================================
st.sidebar.header("Systemparameter")

# Eingabe für die Regelstrecke
txt_num = st.sidebar.text_input("Strecke Zählerpolynom", "[1]")
txt_den = st.sidebar.text_input("Strecke Nennerpolynom", "[1, 2, 1]")

st.sidebar.markdown("---")
st.sidebar.header("Reglerkonfiguration")

# Aktivierung der Regleranteile
chk_i = st.sidebar.checkbox("I-Anteil", value=False)
chk_d = st.sidebar.checkbox("D-Anteil", value=False)

# Numerische Reglerparameter
Kp = st.sidebar.number_input("Proportionalbeiwert Kp", value=1.5, step=0.1)
Tn = st.sidebar.number_input("Nachstellzeit Tn (s)", value=2.0, step=0.1, min_value=1e-6) if chk_i else 1.0
Tv = st.sidebar.number_input("Vorhaltzeit Tv (s)", value=0.2, step=0.05, min_value=0.0) if chk_d else 0.0

st.sidebar.markdown("---")
phi_soll = st.sidebar.slider("Gewünschte Phasenreserve in [°]", min_value=5, max_value=90, value=50, step=1)

# ==========================================
# BERECHNUNGSKERN (BERECHNUNG DER SYSTEME)
# ==========================================
try:
    # Strings in Python-Listen konvertieren
    num_s = eval(txt_num)
    den_s = eval(txt_den)
    
    # Übertragungsfunktion der Strecke
    G = ct.tf(num_s, den_s)

    # Dynamische Reglerstruktur aufbauen
    if not chk_i and not chk_d:
        num_r, den_r = [Kp], [1]
    elif chk_i and not chk_d:
        num_r, den_r = [Kp * Tn, Kp], [Tn, 0]
    elif not chk_i and chk_d:
        num_r, den_r = [Kp * Tv, Kp], [1]
    else:
        num_r, den_r = [Kp * Tv, Kp, Kp / Tn], [1, 0]

    R = ct.tf(num_r, den_r)
    G0 = R * G               # Offener Regelkreis
    T = ct.feedback(G0, 1)   # Geschlossener Regelkreis

    # Ist-Kenngrößen ermitteln
    try:
        gm, pm, wg, wd = ct.margin(G0)
        pm_text = f"{pm:.1f}°" if (pm is not None and not np.isnan(pm)) else "Nicht definiert"
    except Exception:
        wd, pm_text = None, "Berechnungsfehler"

    # Frequenzvektor generieren
    omega = np.logspace(-2, 2, 800)
    mag, phase, omega_out = ct.bode_plot(G0, omega, plot=False)
    mag_db = 20 * np.log10(mag)
    phase_deg = np.degrees(phase)

    # Intervallsuche für Soll-Durchtrittsfrequenz (omega_D,soll)
    phase_ziel = phi_soll - 180.0
    wd_soll = None
    mag_db_soll = None

    for i in range(len(phase_deg) - 1):
        p1, p2 = phase_deg[i], phase_deg[i+1]
        w1, w2 = omega_out[i], omega_out[i+1]
        m1, m2 = mag_db[i], mag_db[i+1]

        if (p1 <= phase_ziel <= p2) or (p2 <= phase_ziel <= p1):
            if abs(p2 - p1) > 1e-4:
                wd_soll = w1 + (phase_ziel - p1) * (w2 - w1) / (p2 - p1)
                mag_db_soll = m1 + (wd_soll - w1) * (m2 - m1) / (w2 - w1)
                break

    # Hilfsgrenzen für vertikale Linien
    y_min_db, y_max_db = float(np.min(mag_db)), float(np.max(mag_db))
    y_min_deg, y_max_deg = float(np.min(phase_deg)), float(np.max(phase_deg))


    # ==========================================
    # VISUALISIERUNG: LATEX FORMELN
    # ==========================================
    st.subheader("Mathematische Beschreibung")
    formel_col1, formel_col2 = st.columns(2)
    
    with formel_col1:
        st.markdown("**Regelstrecke $G(s)$:**")
        st.latex(f"G(s) = {transfer_function_to_latex(G)}")
        st.markdown("*Zeitkonstantenform:*")
        st.latex(f"G(s) = {transfer_function_to_time_constant_latex(G)}")
        
    with formel_col2:
        st.markdown("**Regler $R(s)$:**")
        st.latex(f"R(s) = {transfer_function_to_latex(R)}")
        st.markdown("*Zeitkonstantenform:*")
        # KORREKTUR: .checked entfernt, da chk_i und chk_d bereits Booleans sind
        st.latex(f"R(s) = {regler_to_time_constant_latex(Kp, Tn, Tv, chk_i, chk_d)}")

    st.markdown("---")



    # ==========================================
    # VISUALISIERUNG: LAYOUT IN STREAMLIT
    # ==========================================
    col1, col2 = st.columns([3, 2])

    with col1:
        st.subheader(f"Bode-Diagramm (phi_ist: {pm_text})")
        
        fig_bode = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1)

        # Reihe 1: Amplitudengang
        fig_bode.add_trace(go.Scatter(x=omega_out, y=mag_db, mode='lines', name='|G0(jw)| [dB]', line=dict(color='blue', width=2)), row=1, col=1)
        if wd is not None and not np.isnan(wd):
            fig_bode.add_trace(go.Scatter(x=[wd], y=[0], mode='markers', name='Ist-w_D', marker=dict(color='blue', size=8)), row=1, col=1)
            
        if wd_soll is not None and not np.isnan(wd_soll):
            fig_bode.add_trace(go.Scatter(x=[wd_soll], y=[mag_db_soll], mode='markers+text', name='Soll-w_D (Betrag)',
                                          text=[f"  {mag_db_soll:.1f} dB"], textposition="top right", marker=dict(color='red', size=10, symbol='x')), row=1, col=1)
            fig_bode.add_shape(type="line", x0=wd_soll, y0=y_min_db, x1=wd_soll, y1=y_max_db, line=dict(color="red", width=1.5, dash="dashdot"), row=1, col=1)

        # Reihe 2: Phasengang
        fig_bode.add_trace(go.Scatter(x=omega_out, y=phase_deg, mode='lines', name='Phase G0(jw) [°]', line=dict(color='orange', width=2)), row=2, col=1)
        fig_bode.add_trace(go.Scatter(x=[omega_out[0], omega_out[-1]], y=[-180, -180], mode='lines', name='Stabilitätsgrenze (-180°)', line=dict(color='black', dash='dot')), row=2, col=1)
        
        # Horizontale Ziel-Phasenlinie
        fig_bode.add_trace(go.Scatter(x=[omega_out[0], omega_out[-1]], y=[phase_ziel, phase_ziel], mode='lines', 
                                      name=f'Ziel-Phase ({phase_ziel:.1f}°)', line=dict(color='red', width=1.5, dash='dash')), row=2, col=1)
        
        if wd_soll is not None and not np.isnan(wd_soll):
            fig_bode.add_trace(go.Scatter(x=[wd_soll], y=[phase_ziel], mode='markers+text', name='Soll-w_D (Phase)',
                                          text=[f"  w_D,soll = {wd_soll:.2f} rad/s"], textposition="top right", marker=dict(color='red', size=10, symbol='x')), row=2, col=1)
            fig_bode.add_shape(type="line", x0=wd_soll, y0=y_min_deg, x1=wd_soll, y1=y_max_deg, line=dict(color="red", width=1.5, dash="dashdot"), row=2, col=1)

        # ACHSEN-KONFIGURATION (Explizit logarithmisches Zwischengitter aktivieren)
        fig_bode.update_xaxes(
            type='log', gridcolor='lightgray', gridwidth=1,
            minor=dict(dtick='log', showgrid=True, gridcolor='gray', gridwidth=0.5),
            row=1, col=1
        )
        fig_bode.update_xaxes(
            type='log', gridcolor='lightgray', gridwidth=1, title_text="Frequenz w [rad/s]",
            minor=dict(dtick='log', showgrid=True, gridcolor='gray', gridwidth=0.5),
            row=2, col=1
        )
        
        fig_bode.update_yaxes(gridcolor='lightgray', title_text="Verstärkung [dB]", row=1, col=1)
        fig_bode.update_yaxes(gridcolor='lightgray', title_text="Phase [°]", row=2, col=1)
        
        fig_bode.update_layout(plot_bgcolor='white', height=800, margin=dict(t=20, b=20),
                               legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5))
        
        st.plotly_chart(fig_bode, use_container_width=True)

    with col2:
        st.subheader("Zeitbereich: Sprungantwort")
        fig_step = go.Figure()
        try:
            t, y = ct.step_response(T)
            fig_step.add_trace(go.Scatter(x=t, y=y, mode='lines', name='xa(t)', line=dict(color='green', width=2)))
            fig_step.add_trace(go.Scatter(x=[t[0], t[-1]], y=[1, 1], mode='lines', name='w(t)=1', line=dict(color='black', dash='dash')))
            max_overshoot = (np.max(y) - 1) * 100 if len(y) > 0 and np.max(y) > 1 else 0
            st.info(f"Maximales Überschwingen: {max_overshoot:.1f}%")
        except Exception:
            st.error("Sprungantwort für dieses System instabil!")

        fig_step.update_layout(xaxis=dict(title='Zeit t [s]', gridcolor='lightgray'),
                               yaxis=dict(title='Amplitude', gridcolor='lightgray'),
                               plot_bgcolor='white', height=400, margin=dict(t=20, b=20))
        
        st.plotly_chart(fig_step, use_container_width=True)

except Exception as e:
    st.error(f"Fehler in den mathematischen Eingabedaten: {e}")
