import tasks from "./data.json";

export type ExperimentTask = (typeof tasks)[number];
export type ExperimentMode = ExperimentTask["modes"][number];
export type GroupPosition = [number, number];

const number = (value: number) => value.toLocaleString("en-US");
const remaining = (count: number, total: number) => count === total ? "Full"
  : count === 0 ? "Deleted" : `${Math.round(count / total * 100)}% remaining`;

function Arrow() {
  return <svg width="16" height="16" viewBox="0 0 24 24" fill="none" aria-hidden="true">
    <path d="M5 12h14m-6-6 6 6-6 6" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
  </svg>;
}

type Props = {
  data: ExperimentTask;
  mode: ExperimentMode;
  selectedGroup: GroupPosition | null;
  hovered: GroupPosition | null;
  onHoveredChange: (position: GroupPosition | null) => void;
  onSelectGroup: (position: GroupPosition) => void;
  scales?: number[][];
  showDetails?: boolean;
  className?: string;
};

/** Shared site map: both the live explorer and process animation render these exact squares. */
export function NeuronsMap({
  data, mode, selectedGroup, hovered, onHoveredChange, onSelectGroup,
  scales, showDetails = true, className = "",
}: Props) {
  const activeGroup = hovered ?? selectedGroup;
  const [layer, group] = activeGroup ?? [0, 0];

  return <section className={`map-card ${className}`} aria-labelledby="map-title">
    <div className="section-heading"><h2 id="map-title">Neurons retained</h2>
      <span className="legend" title="Brightness combines Fisher-estimated task importance and the fraction of neurons retained; it does not measure activation frequency."><i /> Removed <span className="ramp" /> Neuron importance</span></div>
    <div className="map-scroll">
      <svg className="heatmap" viewBox="0 0 940 398" role="group" aria-label="Ten Fisher-based groups per layer, ordered from lowest to highest average neuron ID, with importance-adjusted opacity"
        onMouseLeave={() => onHoveredChange(null)}>
        {mode.opacity.flatMap((column, l) => column.map((opacity, r) => {
          const x = l * 36 + 4;
          const y = r * 36 + 5;
          const scale = scales?.[l]?.[r] ?? 1;
          const transform = scale === 1 ? undefined
            : `translate(${x + 14.5} ${y + 14.5}) scale(${scale}) translate(${-x - 14.5} ${-y - 14.5})`;
          return <g key={`${l}:${r}`} className="neuron-group" transform={transform}>
            <rect x={x} y={y} width="29" height="29" rx="4" fill="#f0f3f7" />
            <rect x={x} y={y} width="29" height="29" rx="4" fill="#3b82f6"
              className="neuron-node" fillOpacity={opacity} tabIndex={0} role="button"
              aria-pressed={selectedGroup?.[0] === l && selectedGroup?.[1] === r}
              aria-label={`Layer ${l + 1}/${data.layers}, Group ${r + 1}: ${remaining(mode.remaining[l][r], data.groups[l][r].size)}`}
              onMouseEnter={() => onHoveredChange([l, r])} onFocus={() => onHoveredChange([l, r])}
              onClick={() => onSelectGroup([l, r])}
              onKeyDown={e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); onSelectGroup([l, r]); } }}
              onBlur={() => onHoveredChange(null)}>
              <title>{`Layer ${l + 1}/${data.layers} · Group ${r + 1}/10 · ${remaining(mode.remaining[l][r], data.groups[l][r].size)} · mean importance ${Math.round(data.groups[l][r].percentile)}%`}</title>
            </rect>
            <rect className="node-hover" x={x + 1} y={y + 1} width="27" height="27" rx="3" />
          </g>;
        }))}
        {[0, 4, 9, 14, 19, 25].map(l => <text key={l} x={l * 36 + 19} y="382" textAnchor="middle">{String(l + 1).padStart(2, "0")}</text>)}
      </svg>
    </div>
    <div className="map-key"><span className="layer-direction">Layers <Arrow /></span></div>
    {showDetails && <div className="map-detail" aria-live="polite">
      {activeGroup ? <><strong>Layer {layer + 1}/{data.layers} · Group {group + 1}/10</strong>
        <span>{remaining(mode.remaining[layer][group], data.groups[layer][group].size)}</span>
        <span>Mean importance {Math.round(data.groups[layer][group].percentile)}%</span>
      </>
        : <span>Model: Gemma 2 2B <b>–</b> {data.layers} layers, each with attention + MLP <b>–</b> {number(data.total / data.layers)} MLP neurons per layer, summarized in 10 groups of ~921 neurons</span>}
    </div>}
  </section>;
}
