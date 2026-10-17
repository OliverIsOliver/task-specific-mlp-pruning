import { useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import tasks from "./data.json";
import { NeuronsMap } from "./NeuronsMap";
import { TestCases } from "./TestCases";
import "./style.css";

const number = (value: number) => value.toLocaleString("en-US");

function DownloadIcon() {
  return <svg width="16" height="16" viewBox="0 -960 960 960" fill="currentColor" aria-hidden="true">
    <path d="M480-320 280-520l56-58 104 104v-326h80v326l104-104 56 58-200 200ZM240-160q-33 0-56.5-23.5T160-240v-120h80v120h480v-120h80v120q0 33-23.5 56.5T720-160H240Z" />
  </svg>;
}

function GitHubLink() {
  return <a className="github-link" href="https://github.com/oliveramaher/task-specific-mlp-pruning" target="_blank" rel="noreferrer">
        <svg width="17" height="17" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
          <path d="M12 .75a11.25 11.25 0 0 0-3.558 21.923c.563.104.77-.244.77-.542 0-.268-.01-.978-.015-1.92-3.13.68-3.79-1.51-3.79-1.51-.512-1.3-1.25-1.646-1.25-1.646-1.022-.7.078-.686.078-.686 1.13.08 1.725 1.16 1.725 1.16 1.004 1.722 2.634 1.225 3.277.937.102-.728.393-1.225.714-1.507-2.499-.284-5.126-1.25-5.126-5.565 0-1.23.44-2.234 1.16-3.022-.117-.285-.503-1.43.11-2.98 0 0 .945-.303 3.094 1.154a10.78 10.78 0 0 1 5.624 0c2.148-1.457 3.09-1.154 3.09-1.154.615 1.55.23 2.695.113 2.98.722.788 1.157 1.792 1.157 3.022 0 4.325-2.632 5.278-5.14 5.556.404.35.765 1.042.765 2.1 0 1.516-.014 2.739-.014 3.11 0 .3.203.651.774.54A11.25 11.25 0 0 0 12 .75Z" />
        </svg><span>GitHub</span>
      </a>;
}

function App() {
  const datasetDialog = useRef<HTMLDialogElement>(null);
  const taskMenu = useRef<HTMLDetailsElement>(null);
  useEffect(() => {
    const closeMenu = (e: PointerEvent) => {
      if (taskMenu.current && !taskMenu.current.contains(e.target as Node)) taskMenu.current.open = false;
    };
    document.addEventListener("pointerdown", closeMenu);
    return () => document.removeEventListener("pointerdown", closeMenu);
  }, []);
  const [dataset, setDataset] = useState({ label: "", text: "" });
  const [taskIndex, setTaskIndex] = useState(0);
  const [selected, setSelected] = useState(0);
  const [index, setIndex] = useState(0);
  const [previewIndex, setPreviewIndex] = useState<number | null>(null);
  const [hovered, setHovered] = useState<[number, number] | null>(null);
  const [selectedGroup, setSelectedGroup] = useState<[number, number] | null>(null);
  const data = tasks[taskIndex];
  const modes = data.modes;
  const mode = modes[selected];
  const delta = mode.correct - modes[0].correct;
  function chooseTask(i: number) {
    if (taskMenu.current) taskMenu.current.open = false;
    setTaskIndex(i);
    setIndex(0);
    setPreviewIndex(null);
    setHovered(null);
    setSelectedGroup(null);
  }
  async function openDataset(path: string, label: string) {
    const response = await fetch(path);
    setDataset({ label, text: await response.text() });
    datasetDialog.current?.showModal();
  }

  return <main>
    <header>
      <a className="wordmark" href="./"><span className="mark"><i /><i /><i /><i /></span>pruning</a>
      <nav className="task-tabs" aria-label="Evaluation task">
        {tasks.map((task, i) => <button key={task.title} aria-pressed={taskIndex === i}
          onClick={() => chooseTask(i)}>{task.tab}</button>)}
      </nav>
      <details className="task-menu" ref={taskMenu}
        onKeyDown={e => { if (e.key === "Escape" && taskMenu.current) { taskMenu.current.open = false; taskMenu.current.querySelector("summary")?.focus(); } }}
        onBlur={e => { if (!e.currentTarget.contains(e.relatedTarget)) e.currentTarget.open = false; }}>
        <summary aria-label={`Choose task: ${data.tab}`}>{data.tab}
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="m6 9 6 6 6-6" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" /></svg>
        </summary>
        <nav className="task-options" aria-label="Task choices">
          {tasks.map((task, i) => <button key={task.title} aria-pressed={taskIndex === i} onClick={() => chooseTask(i)}>
            {task.tab}<span aria-hidden="true">{taskIndex === i ? "✓" : ""}</span>
          </button>)}
        </nav>
      </details>
      <GitHubLink />
    </header>

    <section className="experiment" aria-labelledby="experiment-title">
      <div className="experiment-heading">
        <div><h1 id="experiment-title">{data.title}</h1>
          <div className="dataset-links">
            <a href={data.datasets.training} onClick={e => { e.preventDefault(); void openDataset(data.datasets.training, `${data.tab} · Training data`); }}>Training data <DownloadIcon /></a>
            <a href={data.datasets.testing} onClick={e => { e.preventDefault(); void openDataset(data.datasets.testing, `${data.tab} · Testing data`); }}>Testing data <DownloadIcon /></a>
          </div>
        </div>
        <div className="tabs" role="group" aria-label="Model retained">
          {modes.map((item, i) => <button key={item.key} aria-pressed={selected === i}
            onClick={() => setSelected(i)}>{item.label}</button>)}
        </div>
      </div>
      <div className="stats" aria-label="Saved experiment results" aria-live="polite">
        <div><span>MLP retained</span><strong>{100 - mode.cut}<small>%</small></strong></div>
        <div><span>Neurons removed</span><strong>{number(mode.removed)}</strong>
          <p>of {number(data.total)}</p></div>
        <div><span>Correct answers</span><strong>{mode.correct}<small> / {data.samples.length}</small></strong>
          <p>{selected === 0 ? "Full-model baseline" : `${delta > 0 ? "+" : ""}${delta} vs full model`}</p></div>
      </div>
    </section>

    <NeuronsMap data={data} mode={mode} selectedGroup={selectedGroup} hovered={hovered}
      onHoveredChange={setHovered} onSelectGroup={setSelectedGroup} />

    <TestCases data={data} mode={mode} index={index} previewIndex={previewIndex}
      onPreviewChange={setPreviewIndex} onSelect={setIndex} />


    <footer className="mobile-footer"><GitHubLink /></footer>

    <dialog ref={datasetDialog} className="dataset-dialog" aria-labelledby="dataset-title">
      <form method="dialog"><h2 id="dataset-title">{dataset.label}</h2><button aria-label="Close dataset">×</button></form>
      <pre>{dataset.text}</pre>
    </dialog>
  </main>;
}

createRoot(document.getElementById("root")!).render(<App />);
