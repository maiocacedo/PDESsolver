import { useState, useCallback, useEffect, useRef } from "react";

const API = "http://localhost:8000";

const C = {
  bg:      "#0f1117",
  panel:   "#1a1d27",
  border:  "#2e3347",
  accent:  "#4fc3f7",
  text:    "#c9d1e0",
  textDim: "#6b7591",
  error:   "#f48fb1",
  success: "#81c784",
};

function validateNPdes(val) {
  const n = parseInt(val);
  if (isNaN(n) || !Number.isInteger(n) || n < 1 || n > 8) return null;
  return n;
}

const inputStyle = (err = false) => ({
  width: "100%", background: C.bg,
  border: `1px solid ${err ? C.error : C.border}`,
  borderRadius: 6, padding: "7px 10px",
  color: C.text, fontSize: 12,
  fontFamily: "'JetBrains Mono', monospace",
  outline: "none", boxSizing: "border-box",
  textAlign: "center", transition: "border-color 0.15s",
});

const selectStyle = {
  background: C.bg, border: `1px solid ${C.border}`,
  borderRadius: 6, padding: "7px 10px",
  color: C.text, fontSize: 12,
  outline: "none", cursor: "pointer", textAlign: "center",
};

const sectionStyle = {
  fontSize: 10, fontWeight: 700, letterSpacing: "0.12em",
  color: C.textDim, textTransform: "uppercase",
  borderBottom: `1px solid ${C.border}`,
  paddingBottom: 6, marginBottom: 12, marginTop: 4,
  textAlign: "center",
};

const labelStyle = {
  fontSize: 10, fontWeight: 600, letterSpacing: "0.08em",
  color: C.textDim, textTransform: "uppercase",
  textAlign: "center", display: "block", marginBottom: 4,
};

const tagStyle = {
  display: "inline-block", fontSize: 10, padding: "2px 8px",
  borderRadius: 4, fontFamily: "'JetBrains Mono', monospace",
  fontWeight: 600, background: C.accent + "22", color: C.accent,
  border: `1px solid ${C.accent}44`,
};

function Sec({ children }) { return <div style={sectionStyle}>{children}</div>; }
function Lbl({ children }) { return <label style={labelStyle}>{children}</label>; }

function Inp({ value, onChange, placeholder, err = false, ...rest }) {
  return (
    <input value={value} onChange={e => onChange(e.target.value)}
           placeholder={placeholder} style={inputStyle(err)} {...rest} />
  );
}

function Sel({ value, onChange, options }) {
  return (
    <select value={value} onChange={e => onChange(e.target.value)}
            style={{ ...selectStyle, width: "100%" }}>
      {options.map(([v, l]) => <option key={v} value={v}>{l}</option>)}
    </select>
  );
}

function Field({ label, children }) {
  return (
    <div style={{ flex: 1, minWidth: 0 }}>
      {label && <Lbl>{label}</Lbl>}
      {children}
    </div>
  );
}

function Row({ children, gap = 10, mb = 10 }) {
  return (
    <div style={{ display: "flex", gap, marginBottom: mb, alignItems: "flex-start" }}>
      {children}
    </div>
  );
}

function BCRow({ label, bdVal, bdOnChange, funcVal, funcOnChange }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 7 }}>
      <span style={{
        fontSize: 11, color: C.textDim, whiteSpace: "nowrap",
        minWidth: 138, textAlign: "right",
      }}>{label}</span>
      <select value={bdVal} onChange={e => bdOnChange(e.target.value)}
              style={{ ...selectStyle, width: 115, flexShrink: 0 }}>
        <option value="Dirichlet">Dirichlet</option>
        <option value="Neumann">Neumann</option>
      </select>
      <input value={funcVal} onChange={e => funcOnChange(e.target.value)}
             placeholder="0"
             style={{ ...inputStyle(), flex: 1, textAlign: "center" }} />
    </div>
  );
}

const defaultPDE = () => ({
  equation: "", ic: "0",
  west_bd: "Dirichlet", west_val: "1",
  east_bd: "Neumann",   east_val: "0",
  south_bd: "Neumann",  south_val: "0",
  north_bd: "Neumann",  north_val: "0",
});

function PDERow({ idx, pde, onChange, is2d }) {
  const letra = String.fromCharCode(65 + idx);
  const set   = (k, v) => onChange(idx, k, v);
  return (
    <div style={{
      background: C.bg, border: `1px solid ${C.border}`,
      borderRadius: 8, padding: 14, marginBottom: 10,
    }}>
      {/* Cabeçalho + equação na mesma linha */}
      <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 10 }}>
        <div style={tagStyle}>PDE {idx + 1}</div>
        <span style={{
          fontFamily: "'JetBrains Mono', monospace",
          fontSize: 12, color: C.textDim, whiteSpace: "nowrap",
        }}>d{letra}/dt =</span>
        <input value={pde.equation} onChange={e => set("equation", e.target.value)}
               placeholder={`0.1*d2${letra}/dx2 - 0.5*${letra}`}
               style={{ ...inputStyle(), flex: 1, textAlign: "left" }} />
      </div>
      <Row mb={10}>
        <Field label="Condição Inicial">
          <Inp value={pde.ic} onChange={v => set("ic", v)} placeholder="Ex: sin(pi*x)" />
        </Field>
      </Row>
      <BCRow label="Cond. Front. Oeste"
             bdVal={pde.west_bd}  bdOnChange={v => set("west_bd", v)}
             funcVal={pde.west_val} funcOnChange={v => set("west_val", v)} />
      <BCRow label="Cond. Front. Leste"
             bdVal={pde.east_bd}  bdOnChange={v => set("east_bd", v)}
             funcVal={pde.east_val} funcOnChange={v => set("east_val", v)} />
      {is2d && <>
        <BCRow label="Cond. Front. Sul"
               bdVal={pde.south_bd}  bdOnChange={v => set("south_bd", v)}
               funcVal={pde.south_val} funcOnChange={v => set("south_val", v)} />
        <BCRow label="Cond. Front. Norte"
               bdVal={pde.north_bd}  bdOnChange={v => set("north_bd", v)}
               funcVal={pde.north_val} funcOnChange={v => set("north_val", v)} />
      </>}
    </div>
  );
}

function RadioItem({ value, current, onChange, label }) {
  const active = value === current;
  return (
    <label style={{
      display: "flex", alignItems: "center", gap: 8,
      cursor: "pointer", fontSize: 12,
      color: active ? C.accent : C.text, padding: "4px 0",
    }}>
      <div onClick={() => onChange(value)} style={{
        width: 15, height: 15, borderRadius: "50%", flexShrink: 0,
        border: `2px solid ${active ? C.accent : C.border}`,
        background: active ? C.accent : "transparent",
        transition: "all 0.15s",
      }} />
      <span onClick={() => onChange(value)}>{label}</span>
    </label>
  );
}

// ── Animação ──────────────────────────────────────────────────────────────────
function AnimPlayer({ frames, tf }) {
  const [frame,   setFrame]   = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed,   setSpeed]   = useState(80);
  const intervalRef = useRef(null);

  useEffect(() => {
    setFrame(0);
    setPlaying(false);
  }, [frames]);

  useEffect(() => {
    clearInterval(intervalRef.current);
    if (playing) {
      intervalRef.current = setInterval(() => {
        setFrame(f => {
          if (f >= frames.length - 1) { setPlaying(false); return f; }
          return f + 1;
        });
      }, speed);
    }
    return () => clearInterval(intervalRef.current);
  }, [playing, speed, frames.length]);

  const t = frames.length > 1
    ? (tf * frame / (frames.length - 1)).toFixed(3)
    : "0.000";

  return (
    <div style={{ display: "flex", flexDirection: "column",
                  alignItems: "center", gap: 12, width: "100%" }}>
      <img src={`data:image/png;base64,${frames[frame]}`}
           alt={`frame ${frame}`}
           style={{ maxWidth: "100%", maxHeight: "calc(100vh - 260px)",
                    objectFit: "contain", borderRadius: 10,
                    boxShadow: "0 4px 40px #00000055" }} />

      {/* Controles */}
      <div style={{
        display: "flex", alignItems: "center", gap: 12,
        background: C.panel, border: `1px solid ${C.border}`,
        borderRadius: 10, padding: "10px 18px", width: "100%",
        maxWidth: 600,
      }}>
        <button onClick={() => setFrame(0)}
                style={animBtn()}>⏮</button>
        <button onClick={() => setFrame(f => Math.max(0, f - 1))}
                style={animBtn()}>◀</button>
        <button onClick={() => setPlaying(p => !p)}
                style={animBtn(true)}>
          {playing ? "⏸" : "▶"}
        </button>
        <button onClick={() => setFrame(f => Math.min(frames.length - 1, f + 1))}
                style={animBtn()}>▶</button>
        <button onClick={() => setFrame(frames.length - 1)}
                style={animBtn()}>⏭</button>

        {/* Slider de frame */}
        <input type="range" min={0} max={frames.length - 1} value={frame}
               onChange={e => { setPlaying(false); setFrame(parseInt(e.target.value)); }}
               style={{ flex: 1, accentColor: C.accent }} />

        <span style={{ fontSize: 11, color: C.textDim,
                       fontFamily: "'JetBrains Mono', monospace",
                       whiteSpace: "nowrap" }}>
          t = {t}
        </span>

        {/* Velocidade */}
        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <span style={{ fontSize: 10, color: C.textDim }}>vel</span>
          <select value={speed} onChange={e => setSpeed(parseInt(e.target.value))}
                  style={{ ...selectStyle, width: 70, padding: "4px 6px" }}>
            <option value={200}>0.5×</option>
            <option value={100}>1×</option>
            <option value={80}>1.5×</option>
            <option value={50}>2×</option>
            <option value={25}>4×</option>
          </select>
        </div>
      </div>
    </div>
  );
}

function animBtn(primary = false) {
  return {
    background: primary ? C.accent + "22" : "transparent",
    border: `1px solid ${primary ? C.accent : C.border}`,
    borderRadius: 6, padding: "4px 10px",
    color: primary ? C.accent : C.textDim,
    fontSize: 14, cursor: "pointer",
  };
}

// ── Downloads ─────────────────────────────────────────────────────────────────
function downloadImage(b64) {
  const a = document.createElement("a");
  a.href = `data:image/png;base64,${b64}`;
  a.download = "pdesolver_resultado.png";
  a.click();
}

function downloadJSON(payload, results) {
  const blob = new Blob(
    [JSON.stringify({ configuracao: payload,
                      timestamp: new Date().toISOString(),
                      resultados: results }, null, 2)],
    { type: "application/json" }
  );
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "pdesolver_dados.json";
  a.click();
}

async function downloadXLSX(payload) {
  const res  = await fetch(`${API}/download_xlsx`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await res.json();
  if (!data.ok) { alert("Erro ao gerar XLSX: " + data.error); return; }
  const a = document.createElement("a");
  a.href = `data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,${data.xlsx}`;
  a.download = data.filename || "pdesolver_dados.xlsx";
  a.click();
}

function dlBtn(color) {
  return {
    background: "transparent",
    border: `1px solid ${color || "#6b7591"}`,
    borderRadius: 6, padding: "4px 10px",
    color: color || "#6b7591", fontSize: 11,
    cursor: "pointer", transition: "all 0.15s",
  };
}

// ── App ───────────────────────────────────────────────────────────────────────
export default function App() {
  const [nPdesRaw,   setNPdesRaw]   = useState("1");
  const [pdes,       setPdes]       = useState([defaultPDE()]);
  const [dimension,  setDimension]  = useState("1D");
  const [nx,         setNx]         = useState("21");
  const [ny,         setNy]         = useState("21");
  const [tf,         setTf]         = useState("1.0");
  const [nt,         setNt]         = useState("200");
  const [tol,        setTol]        = useState("1e-6");
  const [method,     setMethod]     = useState("bdf2");
  const [discMethod, setDiscMethod] = useState("central");
  const [funcIdx,    setFuncIdx]    = useState("0");
  const [plotMode,   setPlotMode]   = useState("final");

  const [loading,    setLoading]    = useState(false);
  const [reploting,  setReploting]  = useState(false);
  const [image,      setImage]      = useState(null);
  const [frames,     setFrames]     = useState(null);  // animação
  const [funcs,      setFuncs]      = useState([]);
  const [error,      setError]      = useState(null);
  const [elapsed,    setElapsed]    = useState(null);
  const [lastPayload,setLastPayload]= useState(null);
  const [rawResults, setRawResults] = useState(null);
  const [hasSolved,  setHasSolved]  = useState(false);

  const is2d       = dimension === "2D";
  const nPdesValid = validateNPdes(nPdesRaw);
  const nPdesError = nPdesRaw !== "" && nPdesValid === null;
  const isAnim     = plotMode === "animation";

  const plotModes1d = [
    ["final",     "Perfil final"],
    ["xt",        "Mapa x × t"],
    ["profiles",  "Perfis sobrepostos"],
    ["animation", "Animação"],
  ];
  const plotModes2d = [
    ["heatmap2d", "Heatmap 2D"],
    ["surface2d", "Superfície 3D"],
    ["animation", "Animação"],
  ];

  const updatePDE = useCallback((idx, key, val) => {
    setPdes(prev => {
      const next = [...prev]; next[idx] = { ...next[idx], [key]: val }; return next;
    });
  }, []);

  const handleNPdesChange = val => {
    setNPdesRaw(val);
    const n = validateNPdes(val);
    if (n === null) return;
    setPdes(prev => {
      if (n > prev.length)
        return [...prev, ...Array(n - prev.length).fill(null).map(defaultPDE)];
      return prev.slice(0, n);
    });
  };

  // Replot automático via cache ao mudar modo/função
  useEffect(() => {
    if (!hasSolved || loading) return;

    if (isAnim) {
      // Busca frames de animação
      const fetchAnim = async () => {
        setReploting(true);
        setFrames(null);
        setImage(null);
        try {
          const res  = await fetch(`${API}/animate`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              func_idx: parseInt(funcIdx) || 0,
              tf: parseFloat(tf),
              n_frames: 60,
            }),
          });
          const data = await res.json();
          if (data.ok) setFrames(data.frames);
        } catch (_) {}
        setReploting(false);
      };
      fetchAnim();
    } else {
      // Replot estático
      const replot = async () => {
        setReploting(true);
        setFrames(null);
        try {
          const res  = await fetch(`${API}/replot`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              func_idx:  parseInt(funcIdx) || 0,
              plot_mode: plotMode,
            }),
          });
          const data = await res.json();
          if (data.ok) setImage(data.image);
        } catch (_) {}
        setReploting(false);
      };
      replot();
    }
  }, [plotMode, funcIdx]); // eslint-disable-line

  const handleSolve = async () => {
    if (loading) return;
    setLoading(true);
    setError(null);
    setImage(null);
    setFrames(null);
    setRawResults(null);
    const t0 = Date.now();

    const payload = {
      pdes: pdes.map((p, i) => ({ ...p, func: String.fromCharCode(65 + i) })),
      dimension, nx: parseInt(nx), ny: parseInt(ny),
      tf: parseFloat(tf), nt: parseInt(nt), tol: parseFloat(tol),
      method, disc_method: discMethod,
      func_idx: parseInt(funcIdx) || 0,
      plot_mode: isAnim ? "final" : plotMode,  // /solve não faz anim
    };
    setLastPayload(payload);

    try {
      const res  = await fetch(`${API}/solve`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      if (!data.ok) throw new Error(data.error || "Erro desconhecido");

      setFuncs(data.funcs || []);
      setRawResults(data);
      setElapsed(((Date.now() - t0) / 1000).toFixed(2));
      setHasSolved(true);

      if (isAnim) {
        // Busca animação logo após resolver
        const ar = await fetch(`${API}/animate`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            func_idx: parseInt(funcIdx) || 0,
            tf: parseFloat(tf), n_frames: 60,
          }),
        });
        const ad = await ar.json();
        if (ad.ok) setFrames(ad.frames);
      } else {
        setImage(data.image);
      }
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const btnDisabled = loading || nPdesValid === null;

  return (
    <div style={{
      display: "flex", height: "100vh", background: C.bg,
      fontFamily: "'IBM Plex Sans', sans-serif", color: C.text,
      overflow: "hidden",
    }}>

      {/* ── Painel esquerdo ── */}
      <div style={{
        width: 520, minWidth: 440, background: C.panel,
        borderRight: `1px solid ${C.border}`,
        display: "flex", flexDirection: "column", overflow: "hidden",
      }}>

        {/* Header */}
        <div style={{
          padding: "16px 22px 12px", borderBottom: `1px solid ${C.border}`,
          background: "#13161f",
        }}>
          <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
            <div style={{
              width: 34, height: 34, borderRadius: 9, flexShrink: 0,
              background: `linear-gradient(135deg, ${C.accent}, #7c4dff)`,
              display: "flex", alignItems: "center",
              justifyContent: "center", fontSize: 18, color: "white",
            }}>∂</div>
            <div>
              <div style={{ fontSize: 17, fontWeight: 700, color: "#e2e8f4",
                            letterSpacing: "-0.02em" }}>PDESsolver</div>
              <div style={{ fontSize: 10, color: C.textDim,
                            letterSpacing: "0.07em" }}>SIMULADOR DE EDPs</div>
            </div>
          </div>
        </div>

        {/* Scroll */}
        <div style={{
          flex: 1, overflowY: "auto", padding: "16px 22px",
          scrollbarWidth: "thin", scrollbarColor: `${C.border} transparent`,
        }}>
          <Sec>Malha</Sec>
          <Row>
            <Field label="Dimensão">
              <Sel value={dimension} onChange={setDimension}
                   options={[["1D","1D — Uma dimensão"],["2D","2D — Duas dimensões"]]} />
            </Field>
            <Field label="Nº de PDEs (1–8)">
              <Inp value={nPdesRaw} onChange={handleNPdesChange}
                   placeholder="1" err={nPdesError} />
              {nPdesError && (
                <div style={{ fontSize: 10, color: C.error, marginTop: 3,
                              textAlign: "center" }}>Insira um inteiro de 1 a 8</div>
              )}
            </Field>
          </Row>
          <Row>
            <Field label="nx (pontos em x)">
              <Inp value={nx} onChange={setNx} placeholder="21" />
            </Field>
            {is2d && (
              <Field label="ny (pontos em y)">
                <Inp value={ny} onChange={setNy} placeholder="21" />
              </Field>
            )}
          </Row>

          <Sec>Tempo</Sec>
          <Row>
            <Field label="Tempo final (tf)">
              <Inp value={tf} onChange={setTf} placeholder="1.0" />
            </Field>
            <Field label="Nº de passos (nt)">
              <Inp value={nt} onChange={setNt} placeholder="200" />
            </Field>
            <Field label="Tolerância">
              <Inp value={tol} onChange={setTol} placeholder="1e-6" />
            </Field>
          </Row>

          <Sec>Método</Sec>
          <Row>
            <Field label="Solver">
              <Sel value={method} onChange={setMethod}
                   options={[
                     ["bdf2","BDF2 (recomendado)"],
                     ["CN","Crank-Nicolson"],
                     ["RKF","RKF45 (GPU)"],
                   ]} />
            </Field>
            <Field label="Discretização">
              <Sel value={discMethod} onChange={setDiscMethod}
                   options={[
                     ["central","Central 2ª ordem"],
                     ["backward","Backward / Upwind"],
                     ["forward","Forward"],
                   ]} />
            </Field>
          </Row>

          {nPdesValid !== null && (
            <>
              <Sec>Equações</Sec>
              {pdes.map((pde, i) => (
                <PDERow key={i} idx={i} pde={pde}
                        onChange={updatePDE} is2d={is2d} />
              ))}
            </>
          )}

          <Sec>Visualização</Sec>
          {funcs.length > 0 && (
            <Row mb={10}>
              <Field label="Função">
                <Sel value={funcIdx} onChange={setFuncIdx}
                     options={funcs.map((f, i) => [String(i), f])} />
              </Field>
            </Row>
          )}
          <div style={{ marginBottom: 4 }}>
            <Lbl>Modo de exibição</Lbl>
            <div style={{
              background: C.bg, border: `1px solid ${C.border}`,
              borderRadius: 8, padding: "10px 14px",
            }}>
              {(is2d ? plotModes2d : plotModes1d).map(([v, l]) => (
                <RadioItem key={v} value={v} current={plotMode}
                           onChange={setPlotMode} label={l} />
              ))}
            </div>
          </div>
          {hasSolved && reploting && (
            <div style={{ fontSize: 11, color: C.textDim,
                          textAlign: "center", marginTop: 6 }}>
              Atualizando visualização…
            </div>
          )}
        </div>

        {/* Footer */}
        <div style={{
          padding: "14px 22px", borderTop: `1px solid ${C.border}`,
          background: "#13161f",
        }}>
          <button onClick={handleSolve} disabled={btnDisabled} style={{
            width: "100%", padding: "12px 0",
            background: btnDisabled
              ? C.border
              : `linear-gradient(135deg, ${C.accent}, #7c4dff)`,
            border: "none", borderRadius: 8, color: "white",
            fontSize: 13, fontWeight: 700, letterSpacing: "0.04em",
            cursor: btnDisabled ? "not-allowed" : "pointer",
            transition: "all 0.2s",
            display: "flex", alignItems: "center",
            justifyContent: "center", gap: 8,
          }}>
            {loading
              ? <><span style={{ animation: "spin 1s linear infinite",
                                 display: "inline-block" }}>⟳</span> Resolvendo…</>
              : "▶  Resolver"}
          </button>
          {error && (
            <div style={{
              marginTop: 10, padding: "8px 12px",
              background: C.error + "18", border: `1px solid ${C.error}44`,
              borderRadius: 6, fontSize: 11, color: C.error,
              fontFamily: "'JetBrains Mono', monospace", wordBreak: "break-all",
            }}>{error}</div>
          )}
        </div>
      </div>

      {/* ── Painel direito ── */}
      <div style={{ flex: 1, display: "flex", flexDirection: "column",
                    overflow: "hidden" }}>

        {/* Status bar */}
        <div style={{
          padding: "10px 24px", borderBottom: `1px solid ${C.border}`,
          background: C.panel, display: "flex", alignItems: "center",
          justifyContent: "space-between",
        }}>
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <div style={{
              width: 7, height: 7, borderRadius: "50%",
              background: (image || frames) ? C.success
                          : loading ? C.accent : C.border,
              boxShadow: (image || frames) ? `0 0 6px ${C.success}` : "none",
              transition: "all 0.3s",
            }} />
            <span style={{ fontSize: 11, color: C.textDim }}>
              {loading    ? "Processando simulação…"
               : reploting ? "Atualizando visualização…"
               : (image || frames) ? `Simulação concluída em ${elapsed}s`
               : "Aguardando configuração"}
            </span>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            {funcs.map((f, i) => <div key={i} style={tagStyle}>{f}</div>)}
            {image && (
              <button onClick={() => downloadImage(image)}
                      style={dlBtn()}>↓ PNG</button>
            )}
            {rawResults && (
              <button onClick={() => downloadJSON(lastPayload, rawResults)}
                      style={dlBtn()}>↓ JSON</button>
            )}
            {hasSolved && (
              <button onClick={() => downloadXLSX(lastPayload)}
                      style={dlBtn(C.success)}>↓ XLSX</button>
            )}
          </div>
        </div>

        {/* Resultado */}
        <div style={{
          flex: 1, display: "flex", flexDirection: "column",
          alignItems: "center", justifyContent: "center",
          padding: 24, overflow: "hidden",
        }}>
          {(loading || reploting) && (
            <div style={{ textAlign: "center", color: C.textDim }}>
              <div style={{ fontSize: 48, marginBottom: 16,
                            animation: "pulse 1.5s ease-in-out infinite" }}>∂</div>
              <div style={{ fontSize: 13 }}>
                {loading ? "Compilando e resolvendo…" : "Preparando visualização…"}
              </div>
            </div>
          )}
          {!loading && !reploting && !image && !frames && (
            <div style={{ textAlign: "center", color: C.textDim }}>
              <div style={{ fontSize: 52, marginBottom: 14, opacity: 0.2 }}>∂u/∂t</div>
              <div style={{ fontSize: 13, maxWidth: 300, lineHeight: 1.7 }}>
                Configure as equações no painel esquerdo e clique em{" "}
                <strong style={{ color: C.accent }}>Resolver</strong>
              </div>
            </div>
          )}
          {!loading && !reploting && frames && (
            <AnimPlayer frames={frames} tf={parseFloat(tf)} />
          )}
          {!loading && !reploting && image && !frames && (
            <img src={`data:image/png;base64,${image}`} alt="resultado"
                 style={{
                   maxWidth: "100%", maxHeight: "100%",
                   objectFit: "contain", borderRadius: 10,
                   boxShadow: "0 4px 40px #00000055",
                 }} />
          )}
        </div>
      </div>

      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { background: ${C.bg}; }
        input:focus, select:focus { border-color: ${C.accent} !important; outline: none; }
        ::-webkit-scrollbar { width: 4px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: ${C.border}; border-radius: 2px; }
        @keyframes spin { to { transform: rotate(360deg); } }
        @keyframes pulse {
          0%, 100% { opacity: 0.3; transform: scale(1); }
          50%       { opacity: 0.8; transform: scale(1.05); }
        }
        button:hover:not(:disabled) { opacity: 0.82; }
      `}</style>
    </div>
  );
}