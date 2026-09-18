from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
import traceback
from tkinter import filedialog, messagebox, ttk

from .config import CourtPipelineConfig, JailPipelineConfig
from .pipelines import run_court_pipeline, run_jail_pipeline
from .sites import COURT_SITES, JAIL_SITES


class ChargeKeyApp(tk.Tk):
    """Small desktop wrapper around the tested jail/court pipeline modules."""

    def __init__(self):
        super().__init__()
        self.title("Charge Key Automation")
        self.geometry("880x730")
        self.minsize(800, 650)
        self.running = False
        self.last_output: str | None = None
        self.events: queue.Queue[tuple[str, str]] = queue.Queue()
        self._build()
        self.after(100, self._poll)

    def _build(self) -> None:
        frame = ttk.Frame(self, padding=20)
        frame.pack(fill="both", expand=True)
        frame.columnconfigure(1, weight=1)

        self.data_var = tk.StringVar()
        self.output_var = tk.StringVar()
        self.workflow_var = tk.StringVar(value="Jail")
        self.site_var = tk.StringVar(value="San Francisco")
        self.year_var = tk.StringVar(value="2026")
        self.sim_var = tk.StringVar(value="0.85")
        self.fuzzy_var = tk.StringVar(value="90")
        self.decision_var = tk.StringVar(value="0.85")
        self.margin_var = tk.StringVar(value="0.03")

        ttk.Label(frame, text="Charge Key Automation", font=("Arial", 18, "bold")).grid(
            row=0, column=0, columnspan=3, sticky="w", pady=(0, 16)
        )
        self._path_row(frame, 1, "Data folder", self.data_var, self._choose_data)
        self._path_row(frame, 2, "Output folder", self.output_var, self._choose_output)

        ttk.Label(frame, text="Workflow").grid(row=3, column=0, sticky="w", pady=5)
        workflow = ttk.Combobox(
            frame,
            textvariable=self.workflow_var,
            values=("Jail", "Court"),
            state="readonly",
        )
        workflow.grid(row=3, column=1, columnspan=2, sticky="ew", pady=5)
        workflow.bind("<<ComboboxSelected>>", self._workflow_changed)

        ttk.Label(frame, text="Jurisdiction").grid(row=4, column=0, sticky="w", pady=5)
        self.site_combo = ttk.Combobox(
            frame, textvariable=self.site_var, values=JAIL_SITES, state="readonly"
        )
        self.site_combo.grid(row=4, column=1, columnspan=2, sticky="ew", pady=5)
        self._entry_row(frame, 5, "Target year", self.year_var)
        self._entry_row(frame, 6, "Embedding threshold", self.sim_var)
        self._entry_row(frame, 7, "Fuzzy threshold", self.fuzzy_var)

        self.decision_label = ttk.Label(frame, text="Reranking threshold")
        self.decision_label.grid(row=8, column=0, sticky="w", pady=5)
        ttk.Entry(frame, textvariable=self.decision_var).grid(
            row=8, column=1, columnspan=2, sticky="ew", pady=5, padx=(8, 0)
        )
        self.margin_label = ttk.Label(frame, text="Minimum margin")
        self.margin_label.grid(row=9, column=0, sticky="w", pady=5)
        ttk.Entry(frame, textvariable=self.margin_var).grid(
            row=9, column=1, columnspan=2, sticky="ew", pady=5, padx=(8, 0)
        )

        controls = ttk.Frame(frame)
        controls.grid(row=10, column=0, columnspan=3, sticky="ew", pady=(14, 8))
        self.run_btn = ttk.Button(controls, text="Run classification", command=self._start)
        self.run_btn.pack(side="left")
        self.open_btn = ttk.Button(
            controls, text="Open output", state="disabled", command=self._open_output
        )
        self.open_btn.pack(side="left", padx=8)

        self.progress = ttk.Progressbar(frame, mode="indeterminate")
        self.progress.grid(row=11, column=0, columnspan=3, sticky="ew", pady=4)
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(frame, textvariable=self.status_var).grid(
            row=12, column=0, columnspan=3, sticky="w", pady=(4, 8)
        )

        self.log = tk.Text(frame, height=16, wrap="word", state="disabled")
        self.log.grid(row=13, column=0, columnspan=3, sticky="nsew")
        frame.rowconfigure(13, weight=1)

    def _workflow_changed(self, _event=None) -> None:
        if self.workflow_var.get() == "Court":
            self.site_combo.configure(values=COURT_SITES)
            if self.site_var.get() not in COURT_SITES:
                self.site_var.set(COURT_SITES[0])
            self.decision_label.configure(text="Agreement threshold")
            self.margin_label.configure(text="Probability threshold")
            self.decision_var.set("0.80")
            self.margin_var.set("0.85")
        else:
            self.site_combo.configure(values=JAIL_SITES)
            if self.site_var.get() not in JAIL_SITES:
                self.site_var.set("San Francisco")
            self.decision_label.configure(text="Reranking threshold")
            self.margin_label.configure(text="Minimum margin")
            self.decision_var.set("0.85")
            self.margin_var.set("0.03")

    def _path_row(self, parent, row, label, variable, command) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=5)
        ttk.Entry(parent, textvariable=variable).grid(
            row=row, column=1, sticky="ew", pady=5, padx=(8, 8)
        )
        ttk.Button(parent, text="Browse", command=command).grid(row=row, column=2, pady=5)

    def _entry_row(self, parent, row, label, variable) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=5)
        ttk.Entry(parent, textvariable=variable).grid(
            row=row, column=1, columnspan=2, sticky="ew", pady=5, padx=(8, 0)
        )

    def _choose_data(self) -> None:
        folder = filedialog.askdirectory()
        if folder:
            self.data_var.set(folder)
            if not self.output_var.get():
                self.output_var.set(os.path.join(os.path.dirname(folder), "outputs"))

    def _choose_output(self) -> None:
        folder = filedialog.askdirectory()
        if folder:
            self.output_var.set(folder)

    def _build_config(self) -> JailPipelineConfig | CourtPipelineConfig:
        common = dict(
            site_name=self.site_var.get(),
            target_year=self.year_var.get(),
            similarity_threshold=float(self.sim_var.get()),
            fuzzy_threshold=int(self.fuzzy_var.get()),
        )
        if self.workflow_var.get() == "Court":
            config = CourtPipelineConfig(
                **common,
                agreement_threshold=float(self.decision_var.get()),
                probability_threshold=float(self.margin_var.get()),
            )
        else:
            config = JailPipelineConfig(
                **common,
                rerank_threshold=float(self.decision_var.get()),
                margin_threshold=float(self.margin_var.get()),
            )
        config.validate()
        return config

    def _start(self) -> None:
        if self.running:
            return
        try:
            config = self._build_config()
        except Exception as exc:
            messagebox.showerror("Invalid settings", str(exc))
            return
        if not os.path.isdir(self.data_var.get()):
            messagebox.showerror("Invalid data folder", "Select a valid data folder.")
            return
        if not self.output_var.get().strip():
            messagebox.showerror("Missing output folder", "Select an output folder.")
            return

        self.running = True
        self.run_btn.configure(state="disabled")
        self.open_btn.configure(state="disabled")
        self.progress.start(10)
        self.status_var.set(f"{self.workflow_var.get()} classification running...")
        self._append(f"Starting {self.workflow_var.get().lower()} classification...\n")
        threading.Thread(target=self._worker, args=(config,), daemon=True).start()

    def _worker(self, config: JailPipelineConfig | CourtPipelineConfig) -> None:
        try:
            if isinstance(config, CourtPipelineConfig):
                summary = run_court_pipeline(self.data_var.get(), self.output_var.get(), config)
            else:
                summary = run_jail_pipeline(self.data_var.get(), self.output_var.get(), config)
            self.events.put(("done", str(summary.output_path)))
        except Exception:
            self.events.put(("error", traceback.format_exc()))

    def _poll(self) -> None:
        try:
            while True:
                kind, value = self.events.get_nowait()
                if kind == "done":
                    self._finished(value)
                elif kind == "error":
                    self._failed(value)
        except queue.Empty:
            pass
        self.after(100, self._poll)

    def _finished(self, output_path: str) -> None:
        self.running = False
        self.progress.stop()
        self.run_btn.configure(state="normal")
        self.status_var.set("Classification complete.")
        self.last_output = output_path
        if os.path.exists(output_path):
            self.open_btn.configure(state="normal")
        self._append(f"Completed successfully.\nOutput: {output_path}\n")

    def _failed(self, error: str) -> None:
        self.running = False
        self.progress.stop()
        self.run_btn.configure(state="normal")
        self.status_var.set("Pipeline failed.")
        self._append(f"Pipeline error:\n{error}\n")
        messagebox.showerror(
            "Pipeline error", error.splitlines()[-1] if error else "Unknown error"
        )

    def _open_output(self) -> None:
        if not self.last_output or not os.path.exists(self.last_output):
            messagebox.showerror("File not found", "The generated output file could not be found.")
            return
        if sys.platform.startswith("win"):
            os.startfile(self.last_output)
        elif sys.platform == "darwin":
            subprocess.call(["open", self.last_output])
        else:
            subprocess.call(["xdg-open", self.last_output])

    def _append(self, text: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", text)
        self.log.see("end")
        self.log.configure(state="disabled")


def main() -> None:
    ChargeKeyApp().mainloop()


if __name__ == "__main__":
    main()
