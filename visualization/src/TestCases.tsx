import type { ExperimentMode, ExperimentTask } from "./NeuronsMap";

type Props = {
  data: ExperimentTask;
  mode: ExperimentMode;
  index: number;
  previewIndex: number | null;
  onPreviewChange: (index: number | null) => void;
  onSelect: (index: number) => void;
  visibility?: number[];
  scales?: number[];
  interactive?: boolean;
  className?: string;
};

type GridProps = Pick<Props, "data" | "mode" | "index" | "visibility" | "scales" | "interactive" | "onPreviewChange" | "onSelect">;

/** The saved green/red outcome grid, shared by the explorer and process animation. */
export function TestCaseGrid({
  data, mode, index, visibility, scales, interactive = true, onPreviewChange, onSelect,
}: GridProps) {
  return <div className="results-panel">
    <div className="section-heading"><h2>Test cases</h2><span>{mode.correct} / {data.samples.length} correct</span></div>
    <div className="result-grid" role="group" aria-label="Saved test results"
      onMouseLeave={interactive ? () => onPreviewChange(null) : undefined}
      onBlur={interactive ? e => {
        if (!e.currentTarget.contains(e.relatedTarget)) onPreviewChange(null);
      } : undefined}>
      {data.samples.map((entry, i) => {
        const correct = entry.answers[mode.key as keyof typeof entry.answers].correct;
        const animated = visibility !== undefined;
        const style = animated ? {
          opacity: visibility[i] ?? 0,
          transform: `scale(${scales?.[i] ?? 0.72})`,
        } : undefined;
        return <button key={entry.id} className={correct ? "pass" : "fail"}
          style={style}
          aria-label={`${data.item} ${i + 1}: ${correct ? "correct" : "incorrect"}`}
          aria-pressed={index === i}
          onMouseEnter={interactive ? () => onPreviewChange(i) : undefined}
          onFocus={interactive ? () => onPreviewChange(i) : undefined}
          onClick={interactive ? () => { onSelect(i); onPreviewChange(null); } : undefined}>
          {String(i + 1).padStart(2, "0")}
        </button>;
      })}
    </div>
    <div className="result-legend"><span><i className="pass" />Correct</span><span><i className="fail" />Incorrect</span></div>
  </div>;
}

/** Shared saved test-case panel used by the site and the process animation. */
export function TestCases({
  data, mode, index, previewIndex, onPreviewChange, onSelect,
  visibility, scales, interactive = true, className = "",
}: Props) {
  const activeIndex = previewIndex ?? index;
  const sample = data.samples[activeIndex];
  const answer = sample.answers[mode.key as keyof typeof sample.answers];

  return <section className={`replay ${className}`} aria-label="Explore saved answers">
    <TestCaseGrid data={data} mode={mode} index={index} visibility={visibility} scales={scales}
      interactive={interactive} onPreviewChange={onPreviewChange} onSelect={onSelect} />
    <div className="answer-panel" aria-live="polite">
      <div className="section-heading"><h2>{data.item} {String(activeIndex + 1).padStart(2, "0")} / {data.samples.length}</h2><span>{mode.label}</span></div>
      <p className="transcript" key={sample.id}>“{sample.transcript}”</p>
      <div className="answer" key={`${sample.id}:${mode.key}`}>
        <div className="answer-label"><span>VERBATIM SAVED RESPONSE</span></div>
        <pre className="raw-response">{answer.text}</pre>
        {answer.truncated && <p className="expected">Stopped at the original {data.maxTokens}-token limit.</p>}
        <p className={`verdict ${answer.correct ? "correct" : "incorrect"}`}>
          <span>{answer.correct ? "✓" : <i className="mismatch-arrow">↳</i>}</span>{answer.correct ? "Matches expected answer" : "Doesn’t match expected answer"}
        </p>
        {!answer.correct && <div className="expected-answer">
          <div className="answer-label">EXPECTED RESPONSE</div>
          <pre className="raw-response">{JSON.stringify(sample.expected)}</pre>
        </div>}
      </div>
    </div>
  </section>;
}
