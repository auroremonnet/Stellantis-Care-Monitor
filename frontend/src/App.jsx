import React, { useState, useEffect, useCallback, useRef } from 'react';
import CameraView from './components/CameraView';
import { useCamera } from './hooks/useCamera';
import { api } from './services/api';
import './App.css';

// ─── Constantes perf ────────────────────────────────────────────────────────
const ANALYSIS_FPS   = 8;   // Frames envoyées au backend (évite le lag)
const ANALYSIS_MS    = 1000 / ANALYSIS_FPS;
const VLM_CHECK_MS   = 2000;

function App() {
  const { videoRef, canvasRef, captureFrame, error } = useCamera();

  // ── State ────────────────────────────────────────────────────────────────
  const [temperature,   setTemperature]   = useState(21.0);
  const [climateMode,   setClimateMode]   = useState('AUTO');
  const [annotatedImage,setAnnotatedImage]= useState(null);
  const [primaryEmotion,setPrimaryEmotion]= useState('');
  const [vlmQuestion,   setVlmQuestion]   = useState(null);
  const [isWaitingVLM,  setIsWaitingVLM]  = useState(false);
  const [lastVLMCheck,  setLastVLMCheck]  = useState(Date.now());
  const [globalState,   setGlobalState]   = useState('—');
  const [stats,         setStats]         = useState({ CONFORT: 0, NEUTRE: 0, INCONFORT: 0 });
  const [cnnEmotion,    setCnnEmotion]    = useState('—');
  const [facs, setFacs] = useState({ eyes: '—', brows: '—', mouth: '—' });
  const [scores, setScores] = useState({ geo: 0, cnn: 0, total: 0 });
  const [fps, setFps]   = useState(0);
  const [isProcessing, setIsProcessing] = useState(false);

  // Refs pour éviter les re-renders inutiles
  const temperatureRef  = useRef(temperature);
  const lastFrameRef    = useRef(0);
  const frameCountRef   = useRef(0);
  const fpsTimerRef     = useRef(Date.now());

  useEffect(() => { temperatureRef.current = temperature; }, [temperature]);

  // ── Pipeline analyse (throttlé à ANALYSIS_FPS) ──────────────────────────
  const processFrame = useCallback(async () => {
    const now = performance.now();
    if (now - lastFrameRef.current < ANALYSIS_MS) return;
    if (isProcessing) return;

    const frameData = captureFrame();
    if (!frameData) return;

    lastFrameRef.current = now;
    setIsProcessing(true);

    // Calcul FPS affiché
    frameCountRef.current++;
    const elapsed = (now - fpsTimerRef.current) / 1000;
    if (elapsed >= 1.0) {
      setFps(Math.round(frameCountRef.current / elapsed));
      frameCountRef.current = 0;
      fpsTimerRef.current   = now;
    }

    try {
      const result = await api.sendFrame(frameData, temperatureRef.current);
      if (!result) return;

      if (result.annotated_image !== undefined) setAnnotatedImage(result.annotated_image);
      if (result.primary_emotion !== undefined) setPrimaryEmotion(result.primary_emotion);
      if (result.temperature     !== undefined) setTemperature(result.temperature);
      if (result.climate_mode    !== undefined) setClimateMode(result.climate_mode);
      if (result.facs            !== undefined) setFacs(result.facs);
      if (result.stats           !== undefined) setStats(result.stats);
      if (result.state           !== undefined) setGlobalState(result.state);
      if (result.emotion         !== undefined) setCnnEmotion(result.emotion);
      if (result.scores          !== undefined) setScores(result.scores);
    } catch (err) { /* silencieux */ }
    finally { setIsProcessing(false); }
  }, [captureFrame, isProcessing]);

  // ── Boucle RAF pour la vidéo fluide + throttling API ────────────────────
  const rafRef = useRef(null);
  useEffect(() => {
    const loop = () => { processFrame(); rafRef.current = requestAnimationFrame(loop); };
    rafRef.current = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(rafRef.current);
  }, [processFrame]);

  // ── VLM check (réduit à toutes les 2s) ──────────────────────────────────
  useEffect(() => {
    const id = setInterval(async () => {
      if (isWaitingVLM) return;
      if (Date.now() - lastVLMCheck < 5000) return;
      try {
        const result = await api.checkVLM();
        if (result.question) { setVlmQuestion(result.question); setIsWaitingVLM(true); }
        else setLastVLMCheck(Date.now());
      } catch (e) {}
    }, VLM_CHECK_MS);
    return () => clearInterval(id);
  }, [isWaitingVLM, lastVLMCheck]);

  const handleVLMResponse = async (response) => {
    try { await api.sendVLMResponse(response); } catch (e) {}
    setVlmQuestion(null); setIsWaitingVLM(false); setLastVLMCheck(Date.now());
  };

  // ── Interprétation FACS ───────────────────────────────────────────────────
  const eyeInfo = (() => {
    const v = facs.eyes || '';
    if (v.includes('Plisses'))     return { icon:'😌', state:'Plissés',     color:'#16a34a', metric:'EAR < 0.20',      muscle:'Orbicularis Oculi (AU6)',           detail:'Coin des yeux contracté — sourire Duchenne ou fatigue',   score: 0   };
    if (v.includes('Ecarquilles')) return { icon:'😳', state:'Écarquillés', color:'#dc2626', metric:'EAR > 0.35',      muscle:'Levator Palpebrae (AU5)',           detail:'Blanc de l\'œil exposé — alerte ou stress thermique',     score: -2  };
    return                                { icon:'😐', state:'Neutres',     color:'#6b7280', metric:'EAR ≈ baseline',  muscle:'Aucune activation',                detail:'Ouverture oculaire nominale, état de référence calibré',  score: 0   };
  })();

  const browInfo = (() => {
    const v = facs.brows || '';
    if (v.includes('Fronces'))  return { icon:'😠', state:'Froncés',  color:'#dc2626', metric:'Dist. −8% baseline', muscle:'Corrugator Supercilii (AU4)', detail:'Glabelle plissée — inconfort, concentration ou stress fort', score: -6 };
    if (v.includes('Releves'))  return { icon:'🙄', state:'Relevés',  color:'#d97706', metric:'Dist. +8% baseline', muscle:'Frontalis (AU1 + AU2)',       detail:'Sourcils tirés vers le haut — surprise ou inconfort',       score: 0  };
    return                             { icon:'😐', state:'Stables',  color:'#6b7280', metric:'Dist. ≈ baseline',  muscle:'Aucune activation',           detail:'Distance sourcil-pupille nominale, état neutre calibré',    score: 0  };
  })();

  const mouthInfo = (() => {
    const v = facs.mouth || '';
    if (v.includes('Grand Sourire')) return { icon:'😄', state:'Grand Sourire',    color:'#16a34a', metric:'Lift actif + MAR > +0.10', muscle:'Zygomaticus Major (AU12) + ouverture', detail:'Sourire large avec dents — confort élevé confirmé',              score: +5 };
    if (v.includes('Sourire'))       return { icon:'🙂', state:'Sourire léger',    color:'#16a34a', metric:'Corner_Y < Center_Y',      muscle:'Zygomaticus Major (AU12)',            detail:'Coins relevés — confort thermique géométriquement confirmé',       score: +5 };
    if (v.includes('Grimace') || v.includes('Tension')) return { icon:'😬', state:'Grimace / Tension', color:'#dc2626', metric:'Largeur > 112% baseline', muscle:'Risorius (AU20)', detail:'Étirement horizontal sans élévation — inconfort thermique fort', score: -4 };
    if (v.includes('Baillement'))    return { icon:'🥱', state:'Baillement',       color:'#7c3aed', metric:'MAR > 0.35',              muscle:'Jaw Drop (AU26)',                     detail:'Ouverture verticale max — somnolence ou inconfort important',      score: -5 };
    if (v.includes('Parle'))         return { icon:'🗣️', state:'Parle',           color:'#2563eb', metric:'MAR > baseline+0.10',     muscle:'Dépresseur labii (AU17)',             detail:'Ouverture modérée sans marqueur émotionnel',                       score: 0  };
    return                                   { icon:'😐', state:'Fermée / Neutre', color:'#6b7280', metric:'MAR ≈ baseline',          muscle:'Aucune activation',                  detail:'Bouche au repos, état de référence calibré',                       score: 0  };
  })();

  // ── Mapping CNN ───────────────────────────────────────────────────────────
  const emoMap = {
    happy:       { label:'Heureux / Serein',   icon:'😊', color:'#16a34a', confort:'CONFORT',    pts:'+3.0' },
    neutral:     { label:'Neutre',             icon:'😐', color:'#6b7280', confort:'NEUTRE',     pts:'0'    },
    sad:         { label:'Triste / Préoccupé', icon:'😔', color:'#2563eb', confort:'INCONFORT',  pts:'−3.0' },
    angry:       { label:'En colère',          icon:'😠', color:'#dc2626', confort:'INCONFORT',  pts:'−3.0' },
    fear:        { label:'Inquiet / Stressé',  icon:'😨', color:'#7c3aed', confort:'INCONFORT',  pts:'−3.0' },
    surprise:    { label:'Surpris',            icon:'😲', color:'#d97706', confort:'CONFORT',    pts:'+3.0' },
    disgust:     { label:'Inconfort / Rejet',  icon:'😒', color:'#b45309', confort:'INCONFORT',  pts:'−3.0' },
    calibration: { label:'Calibration...',     icon:'⏳', color:'#9ca3af', confort:'—',          pts:'0'    },
  };
  const emo = emoMap[cnnEmotion] || { label: cnnEmotion, icon:'—', color:'#9ca3af', confort:'—', pts:'0' };

  // ── Dominant 30s ──────────────────────────────────────────────────────────
  const dominant = Object.entries(stats).reduce((m,[k,v]) => v>m[1]?[k,v]:m, ['—',-1]);
  const stateColor = { CONFORT:'#16a34a', INCONFORT:'#dc2626', NEUTRE:'#6b7280', '—':'#9ca3af' };

  // Score géo calculé depuis les infos FACS (affichage)
  const displayGeoScore = eyeInfo.score + browInfo.score + mouthInfo.score;
  const displayTotal    = scores.total || displayGeoScore;

  return (
    <div className="app">
      {/* ── HEADER ─────────────────────────────────────────────────────── */}
      <header className="app-header">
        <div className="header-content">
          <div className="logo-space">
            <img src="/Stellantis.png" alt="Stellantis" />
          </div>
          <h1 className="app-title">CARE</h1>
          <div className="header-right">
            <div className="hdr-pills">
              <div className="hdr-pill">
                <span className="hdr-pill-dot" style={{ background: isProcessing ? '#f59e0b' : '#16a34a' }} />
                <span>{fps} FPS</span>
              </div>
              <div className="hdr-pill temp-pill">
                <span className="hdr-pill-icon">🌡️</span>
                <span className="hdr-pill-bold">{temperature.toFixed(1)}°C</span>
                <span className="hdr-pill-sep">·</span>
                <span>{climateMode}</span>
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* ── CONTENU ────────────────────────────────────────────────────── */}
      <div className="app-content">
        <CameraView
          videoRef={videoRef} canvasRef={canvasRef}
          annotatedImage={annotatedImage} emotion={cnnEmotion}
          error={error} vlmQuestion={vlmQuestion}
          onVLMResponse={handleVLMResponse}
          temperature={temperature} primaryEmotion={primaryEmotion}
        />

        {/* ══ 3 PANNEAUX ══════════════════════════════════════════════ */}
        <div className="panels">

          {/* ── PANNEAU 1 : GÉOMÉTRIE FACS ──────────────────────────── */}
          <div className="panel">
            <div className="panel-hd">
              <span className="panel-hd-icon">📐</span>
              <div className="panel-hd-text">
                <div className="panel-hd-title">ANALYSE GÉOMÉTRIQUE — FACS</div>
                <div className="panel-hd-sub">Facial Action Coding System · Landmarks MediaPipe 3D · Normalisé inter-oculaire</div>
              </div>
              <div className="panel-hd-score" style={{ color: displayGeoScore >= 0 ? '#16a34a' : '#dc2626' }}>
                <span className="panel-hd-score-lbl">Score Géo</span>
                <span className="panel-hd-score-val">{displayGeoScore > 0 ? '+' : ''}{displayGeoScore}</span>
              </div>
            </div>

            <div className="facs-list">
              {[
                { zone:'YEUX',     metric:'Eye Aspect Ratio (EAR)',      info: eyeInfo  },
                { zone:'SOURCILS', metric:'Distance Sourcil-Pupille',    info: browInfo },
                { zone:'BOUCHE',   metric:'MAR + Smile Lift Index',      info: mouthInfo},
              ].map(({ zone, metric, info }) => (
                <div className="facs-row" key={zone} style={{ '--accent': info.color }}>
                  <div className="facs-row-l">
                    <span className="facs-emoji">{info.icon}</span>
                    <div>
                      <div className="facs-zone">{zone} <span className="facs-metric-name">— {metric}</span></div>
                      <div className="facs-state" style={{ color: info.color }}>{info.state}</div>
                    </div>
                  </div>
                  <div className="facs-row-r">
                    <div className="facs-muscle">{info.muscle}</div>
                    <div className="facs-detail">{info.detail}</div>
                    <div className="facs-metric-val">{info.metric}</div>
                  </div>
                  <div className="facs-score-badge" style={{
                    background: info.score > 0 ? '#dcfce7' : info.score < 0 ? '#fee2e2' : '#f3f4f6',
                    color:       info.score > 0 ? '#16a34a' : info.score < 0 ? '#dc2626' : '#6b7280',
                  }}>
                    {info.score > 0 ? '+' : ''}{info.score} pts
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* ── PANNEAU 2 : CNN ÉMOTIONS ────────────────────────────── */}
          <div className="panel">
            <div className="panel-hd">
              <span className="panel-hd-icon">🧠</span>
              <div className="panel-hd-text">
                <div className="panel-hd-title">ANALYSE CNN — RECONNAISSANCE ÉMOTIONNELLE</div>
                <div className="panel-hd-sub">ResNet18 fine-tuné · AffectNet · Distribution de probabilités · Inférence toutes les 3 frames</div>
              </div>
              <div className="panel-hd-score" style={{ color: emo.color }}>
                <span className="panel-hd-score-lbl">Score CNN</span>
                <span className="panel-hd-score-val">{emo.pts}</span>
              </div>
            </div>

            <div className="cnn-body">
              <div className="cnn-main-box" style={{ borderColor: emo.color }}>
                <div className="cnn-emoji">{emo.icon}</div>
                <div className="cnn-main-info">
                  <div className="cnn-main-label" style={{ color: emo.color }}>{emo.label}</div>
                  <div className="cnn-main-meta">
                    Classe brute : <code>{cnnEmotion}</code>
                    &nbsp;→&nbsp;
                    <span style={{ color: stateColor[emo.confort] || '#9ca3af', fontWeight: 600 }}>{emo.confort}</span>
                  </div>
                </div>
                <div className="cnn-pts-badge" style={{
                  background: emo.pts.startsWith('+') ? '#dcfce7' : emo.pts === '0' ? '#f3f4f6' : '#fee2e2',
                  color:       emo.pts.startsWith('+') ? '#16a34a' : emo.pts === '0' ? '#6b7280' : '#dc2626',
                }}>
                  {emo.pts} pts
                </div>
              </div>

              <div className="cnn-map-grid">
                {[
                  { emos: ['happy','surprise'],             label: '😊 happy · 😲 surprise',                              confort: 'CONFORT',   pts: '+3.0', color: '#16a34a', bg: '#dcfce7' },
                  { emos: ['neutral'],                      label: '😐 neutral',                                          confort: 'NEUTRE',    pts: '0',    color: '#6b7280', bg: '#f3f4f6' },
                  { emos: ['sad','angry','fear','disgust'],  label: '😔 sad · 😠 angry · 😨 fear · 😒 disgust',           confort: 'INCONFORT', pts: '−3.0', color: '#dc2626', bg: '#fee2e2' },
                ].map(row => {
                  const active = row.emos.includes(cnnEmotion);
                  return (
                    <div key={row.confort} className={`cnn-map-row ${active ? 'cnn-map-active' : ''}`}
                         style={active ? { background: row.bg, borderColor: row.color } : {}}>
                      <span className="cnn-map-emos">{row.label}</span>
                      <span className="cnn-map-arrow">→</span>
                      <span className="cnn-map-tag" style={{ color: row.color }}>{row.confort}</span>
                      <span className="cnn-map-pts" style={{ color: row.color }}>{row.pts} pts</span>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          {/* ── PANNEAU 3 : FUSION + RÉSULTAT ──────────────────────── */}
          <div className="panel">
            <div className="panel-hd">
              <span className="panel-hd-icon">⚡</span>
              <div className="panel-hd-text">
                <div className="panel-hd-title">FUSION MULTI-MODALE — CALCUL & RÉSULTAT</div>
                <div className="panel-hd-sub">Score Géo + Score CNN · Lissage IIR 10 frames · Seuils : ≥ 4.5 → CONFORT · ≤ −4.0 → INCONFORT</div>
              </div>
            </div>

            <div className="fusion-body">

              {/* Équation de fusion */}
              <div className="fusion-equation">
                <div className="feq-block feq-geo">
                  <div className="feq-lbl">GÉOMÉTRIE</div>
                  <div className="feq-val" style={{ color: displayGeoScore >= 0 ? '#16a34a' : '#dc2626' }}>
                    {displayGeoScore > 0 ? '+' : ''}{displayGeoScore}
                  </div>
                  <div className="feq-sub">pts</div>
                </div>
                <div className="feq-op">+</div>
                <div className="feq-block feq-cnn">
                  <div className="feq-lbl">CNN</div>
                  <div className="feq-val" style={{ color: emo.pts.startsWith('+') ? '#16a34a' : emo.pts === '0' ? '#6b7280' : '#dc2626' }}>
                    {emo.pts}
                  </div>
                  <div className="feq-sub">pts</div>
                </div>
                <div className="feq-op">=</div>
                <div className="feq-block feq-total" style={{ borderColor: stateColor[globalState] }}>
                  <div className="feq-lbl">SCORE TOTAL</div>
                  <div className="feq-val feq-total-val" style={{ color: stateColor[globalState] }}>
                    {displayTotal > 0 ? '+' : ''}{typeof displayTotal === 'number' ? displayTotal.toFixed(1) : displayTotal}
                  </div>
                  <div className="feq-sub feq-state" style={{ color: stateColor[globalState] }}>{globalState}</div>
                </div>
              </div>

              {/* Seuils visuels */}
              <div className="fusion-thresholds">
                <div className="ft-bar">
                  <div className="ft-zone ft-inconf">INCONFORT<br/>≤ −4.0</div>
                  <div className="ft-zone ft-neutre">NEUTRE<br/>−4.0 → +4.5</div>
                  <div className="ft-zone ft-confort">CONFORT<br/>≥ +4.5</div>
                </div>
                <div className="ft-cursor-wrap">
                  <div className="ft-cursor" style={{
                    left: `${Math.min(98, Math.max(2, ((Math.max(-10, Math.min(10, displayTotal)) + 10) / 20) * 100))}%`,
                    background: stateColor[globalState]
                  }} />
                </div>
              </div>

              {/* Stats 30s */}
              <div className="fusion-stats-title">MOYENNE GLISSANTE — 30 DERNIÈRES SECONDES</div>

              {[
                { key:'CONFORT',   color:'#16a34a', icon:'✅' },
                { key:'NEUTRE',    color:'#6b7280', icon:'⚖️' },
                { key:'INCONFORT', color:'#dc2626', icon:'🌡️' },
              ].map(({ key, color, icon }) => (
                <div className="fusion-bar-row" key={key}>
                  <span className="fusion-bar-icon">{icon}</span>
                  <span className="fusion-bar-lbl" style={{ color }}>{key}</span>
                  <div className="fusion-bar-track">
                    <div className="fusion-bar-fill" style={{ width:`${stats[key]}%`, background: color }} />
                  </div>
                  <span className="fusion-bar-pct">{stats[key]}%</span>
                </div>
              ))}

              {/* Dominant */}
              <div className="fusion-dominant" style={{ borderColor: stateColor[dominant[0]] }}>
                <div className="fusion-dom-left">
                  <div className="fusion-dom-lbl">ÉTAT DOMINANT (30s)</div>
                  <div className="fusion-dom-val" style={{ color: stateColor[dominant[0]] }}>
                    {dominant[0]}
                    <span className="fusion-dom-pct"> — {dominant[1]}%</span>
                  </div>
                </div>
                <div className="fusion-dom-action">
                  <div className="fusion-dom-action-lbl">ACTION CLIMATISATION</div>
                  <div className="fusion-dom-action-val">
                    {dominant[0] === 'CONFORT'   && '❄️ ECO · Cible 21°C'}
                    {dominant[0] === 'INCONFORT' && '🌬️ MODE MAX · Cible 18°C'}
                    {dominant[0] === 'NEUTRE'    && '🔄 STANDARD · Cible 22.5°C'}
                    {dominant[0] === '—'         && '⏳ En attente...'}
                  </div>
                </div>
              </div>

            </div>
          </div>

        </div>
      </div>
    </div>
  );
}

export default App;
