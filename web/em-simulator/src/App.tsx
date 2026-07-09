import { Activity, Atom, Binary, Download, Gauge, GitBranch, Layers3, Play, Sigma } from 'lucide-react';
import { useEffect, useMemo, useRef, useState } from 'react';
import {
  type CouplingKind,
  type FieldSample,
  type ScenarioConfig,
  faradayTensor,
  simulateScenario,
} from './sim/faraday';

const scenarioLabels: Record<CouplingKind, string> = {
  toroidal_pulse: 'Toroidal pulse',
  braided_pair: 'Braided pair',
  boundary_sheaf: 'Boundary sheaf',
};

const defaultConfig: ScenarioConfig = {
  kind: 'toroidal_pulse',
  grid: 44,
  steps: 16,
  coupling: 0.72,
  phase: 0.28,
  separation: 0.55,
};

function App() {
  const [config, setConfig] = useState<ScenarioConfig>(defaultConfig);
  const [frameIndex, setFrameIndex] = useState(0);
  const [playing, setPlaying] = useState(true);
  const dataset = useMemo(() => simulateScenario(config), [config]);
  const frame = dataset.frames[Math.min(frameIndex, dataset.frames.length - 1)] ?? dataset.frames[0];
  const peakSample = useMemo(() => {
    return frame.samples.reduce((best, sample) => (sample.energyDensity > best.energyDensity ? sample : best), frame.samples[0]);
  }, [frame.samples]);
  const tensor = useMemo(() => faradayTensor(peakSample.electric, peakSample.magnetic), [peakSample]);

  useEffect(() => {
    setFrameIndex((current) => Math.min(current, dataset.frames.length - 1));
  }, [dataset.frames.length]);

  useEffect(() => {
    if (!playing) return undefined;
    const timer = window.setInterval(() => {
      setFrameIndex((current) => (current + 1) % dataset.frames.length);
    }, 280);
    return () => window.clearInterval(timer);
  }, [dataset.frames.length, playing]);

  const setNumber = (key: keyof ScenarioConfig) => (event: React.ChangeEvent<HTMLInputElement>) => {
    setConfig((current) => ({ ...current, [key]: Number(event.target.value) }));
  };

  const exportDataset = () => {
    const blob = new Blob([JSON.stringify({ config, ...dataset }, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = 'electromagnetic-field-data-simulator.json';
    anchor.click();
    URL.revokeObjectURL(url);
  };

  return (
    <main className="shell">
      <header className="topbar">
        <div>
          <p className="eyebrow">Aether-Lang research surface</p>
          <h1>Electromagnetic Field Data Simulator</h1>
        </div>
        <div className="topbarActions">
          <button className="iconButton" type="button" onClick={() => setPlaying((value) => !value)} aria-label={playing ? 'Pause animation' : 'Play animation'}>
            <Play size={18} fill={playing ? 'currentColor' : 'none'} />
          </button>
          <button className="commandButton" type="button" onClick={exportDataset}>
            <Download size={17} />
            Export JSON
          </button>
        </div>
      </header>

      <section className="workspace" aria-label="Simulator workspace">
        <aside className="controls" aria-label="Scenario controls">
          <div className="controlGroup">
            <label htmlFor="scenario">Scenario</label>
            <select
              id="scenario"
              value={config.kind}
              onChange={(event) => setConfig((current) => ({ ...current, kind: event.target.value as CouplingKind }))}
            >
              {Object.entries(scenarioLabels).map(([value, label]) => (
                <option key={value} value={value}>
                  {label}
                </option>
              ))}
            </select>
          </div>

          <Slider label="Coupling" min={0.05} max={1} step={0.01} value={config.coupling} onChange={setNumber('coupling')} />
          <Slider label="Phase" min={0} max={1} step={0.01} value={config.phase} onChange={setNumber('phase')} />
          <Slider label="Separation" min={0.08} max={0.9} step={0.01} value={config.separation} onChange={setNumber('separation')} />
          <Slider label="Grid" min={16} max={72} step={2} value={config.grid} onChange={setNumber('grid')} />
          <Slider label="Frames" min={4} max={48} step={1} value={config.steps} onChange={setNumber('steps')} />
          <Slider label="Frame" min={0} max={dataset.frames.length - 1} step={1} value={frameIndex} onChange={(event) => setFrameIndex(Number(event.target.value))} />
        </aside>

        <section className="visualColumn">
          <FieldCanvas samples={frame.samples} grid={config.grid} frameTime={frame.t} />
          <div className="metricGrid" aria-label="Simulation summary">
            <Metric icon={<Sigma size={18} />} label="Energy density" value={dataset.summary.energyTotal.toFixed(5)} />
            <Metric icon={<Activity size={18} />} label="Coupling index" value={dataset.summary.couplingIndex.toFixed(5)} />
            <Metric icon={<GitBranch size={18} />} label="Betti proxy" value={`β0 ${dataset.summary.topology.betti0} / β1 ${dataset.summary.topology.betti1}`} />
            <Metric icon={<Gauge size={18} />} label="Maxwell residual" value={dataset.summary.maxwellResidual.divergenceE.toFixed(4)} />
          </div>
        </section>

        <aside className="analysisPanel" aria-label="Faraday tensor and topology analysis">
          <section className="readout">
            <div className="readoutTitle">
              <Atom size={18} />
              Faraday tensor at peak energy
            </div>
            <div className="tensorGrid">
              {tensor.flatMap((row, rowIndex) =>
                row.map((value, colIndex) => (
                  <span key={`${rowIndex}-${colIndex}`} className={Math.abs(value) < 1e-9 ? 'zeroCell' : ''}>
                    {value.toFixed(3)}
                  </span>
                )),
              )}
            </div>
          </section>

          <section className="readout">
            <div className="readoutTitle">
              <Binary size={18} />
              Lorentz invariants
            </div>
            <dl className="definitionList">
              <dt>B² - E²</dt>
              <dd>{peakSample.invariant.magneticMinusElectric.toFixed(6)}</dd>
              <dt>E · B</dt>
              <dd>{peakSample.invariant.pseudoscalar.toFixed(6)}</dd>
              <dt>Peak position</dt>
              <dd>{`${peakSample.x.toFixed(2)}, ${peakSample.y.toFixed(2)}`}</dd>
            </dl>
          </section>

          <section className="readout">
            <div className="readoutTitle">
              <Layers3 size={18} />
              Topological field graph
            </div>
            <dl className="definitionList">
              <dt>Active vertices</dt>
              <dd>{dataset.summary.topology.activeVertices}</dd>
              <dt>Active edges</dt>
              <dd>{dataset.summary.topology.activeEdges}</dd>
              <dt>Energy threshold</dt>
              <dd>{dataset.summary.topology.threshold.toFixed(5)}</dd>
            </dl>
          </section>
        </aside>
      </section>
    </main>
  );
}

function Slider({
  label,
  min,
  max,
  step,
  value,
  onChange,
}: {
  label: string;
  min: number;
  max: number;
  step: number;
  value: number;
  onChange: (event: React.ChangeEvent<HTMLInputElement>) => void;
}) {
  return (
    <div className="controlGroup">
      <div className="labelRow">
        <label>{label}</label>
        <span>{Number(value).toFixed(step < 1 ? 2 : 0)}</span>
      </div>
      <input type="range" min={min} max={max} step={step} value={value} onChange={onChange} aria-label={label} />
    </div>
  );
}

function Metric({ icon, label, value }: { icon: React.ReactNode; label: string; value: string }) {
  return (
    <div className="metric">
      <div className="metricIcon">{icon}</div>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function FieldCanvas({ samples, grid, frameTime }: { samples: FieldSample[]; grid: number; frameTime: number }) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const context = canvas.getContext('2d');
    if (!context) return;
    const pixelRatio = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = Math.floor(rect.width * pixelRatio);
    canvas.height = Math.floor(rect.height * pixelRatio);
    context.scale(pixelRatio, pixelRatio);
    drawField(context, samples, grid, rect.width, rect.height);
  }, [samples, grid, frameTime]);

  return (
    <div className="canvasWrap">
      <canvas ref={canvasRef} aria-label="Electromagnetic energy field visualization" />
      <div className="canvasHud">
        <span>t = {frameTime.toFixed(2)}</span>
        <span>{grid}×{grid} lattice</span>
      </div>
    </div>
  );
}

function drawField(context: CanvasRenderingContext2D, samples: FieldSample[], grid: number, width: number, height: number) {
  context.clearRect(0, 0, width, height);
  context.fillStyle = '#f7f4ee';
  context.fillRect(0, 0, width, height);
  const cellW = width / grid;
  const cellH = height / grid;
  const maxEnergy = Math.max(...samples.map((sample) => sample.energyDensity), 1e-9);

  samples.forEach((sample, index) => {
    const row = Math.floor(index / grid);
    const col = index % grid;
    const intensity = Math.min(1, sample.energyDensity / maxEnergy);
    const hue = 198 - 170 * intensity;
    context.fillStyle = `hsl(${hue} 72% ${92 - 44 * intensity}%)`;
    context.fillRect(col * cellW, row * cellH, Math.ceil(cellW), Math.ceil(cellH));
  });

  context.strokeStyle = 'rgba(18, 37, 44, 0.58)';
  context.lineWidth = 1.25;
  const stride = Math.max(2, Math.floor(grid / 14));
  for (let row = 0; row < grid; row += stride) {
    for (let col = 0; col < grid; col += stride) {
      const sample = samples[row * grid + col];
      const cx = col * cellW + cellW / 2;
      const cy = row * cellH + cellH / 2;
      const scale = 10 + 12 * Math.min(1, sample.energyDensity / maxEnergy);
      context.beginPath();
      context.moveTo(cx, cy);
      context.lineTo(cx + sample.poynting.x * scale, cy - sample.poynting.y * scale);
      context.stroke();
    }
  }
}

export default App;
