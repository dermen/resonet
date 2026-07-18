"""
Tkinter control UI for resonet live processing.

- Entry fields to configure live_distributor settings
- Launch / Stop buttons to start/stop the psana2 distributor via srun
- ZMQ PULL socket receives results from GPU worker daemons
- Scrolling console displays timestamp : resolution estimates
- Running throughput counter

Launch:
  resonet.psana.control_ui
  # or
  python -m resonet.utils.psana.control_ui

The UI binds a PULL socket on --ui-port (default 5600).
GPU workers must be started with:
  --result-host <this-machine> --result-port 5600
"""
import json
import os
import signal
import subprocess
import sys
import time
import tkinter as tk
from tkinter import ttk, scrolledtext
from collections import deque
from datetime import datetime


class ResonetControlUI:
    def __init__(self, root, ui_port=5600):
        self.root = root
        self.root.title("Resonet Live Monitor")
        self.root.geometry("780x700")

        self.ui_port = ui_port
        self.proc = None  # subprocess for live_distributor
        self.zmq_ctx = None
        self.zmq_sock = None
        self.running = False

        # Throughput tracking
        self.result_times = deque(maxlen=5000)
        self.total_results = 0

        self._build_settings_frame()
        self._build_control_frame()
        self._build_console_frame()
        self._build_status_bar()
        self._setup_zmq()

        # Start polling ZMQ for results
        self._poll_results()

        # Clean up on close
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── Settings panel ───────────────────────────────────────────────────

    def _build_settings_frame(self):
        frame = ttk.LabelFrame(self.root, text="Distributor Settings", padding=8)
        frame.pack(fill=tk.X, padx=8, pady=(8, 4))

        self.fields = {}
        settings = [
            ("exp",        "Experiment",       "mfxl1234",     12),
            ("run",        "Run",              "42",            6),
            ("hosts",      "Worker hosts",     "sdfampere032 sdfampere033", 30),
            ("nwork",      "Workers/host",     "4",             4),
            ("wavelen",    "Wavelength (A)",   "1.24",          8),
            ("detdist",    "Det dist (mm)",    "150",           8),
            ("pixsize",    "Pixel size (mm)",  "0.075",         8),
            ("center_mm",  "Center mm (f,s)",  "155.5 163.5",  14),
            ("ds_factor",  "DS factor",        "4",             4),
            ("det_name",   "Detector name",    "jungfrau",     12),
            ("port_base",  "Port base",        "5550",          6),
        ]

        for i, (key, label, default, width) in enumerate(settings):
            row, col = divmod(i, 3)
            lbl = ttk.Label(frame, text=label + ":")
            lbl.grid(row=row, column=col * 2, sticky=tk.E, padx=(8, 2), pady=2)
            var = tk.StringVar(value=default)
            ent = ttk.Entry(frame, textvariable=var, width=width)
            ent.grid(row=row, column=col * 2 + 1, sticky=tk.W, padx=(0, 12), pady=2)
            self.fields[key] = var

        # Extra options
        row_extra = (len(settings) // 3) + 1
        self.live_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(frame, text="Live mode", variable=self.live_var
                        ).grid(row=row_extra, column=0, columnspan=2, sticky=tk.W, padx=8)

        self.max_events_var = tk.StringVar(value="")
        ttk.Label(frame, text="Max events:").grid(
            row=row_extra, column=2, sticky=tk.E, padx=(8, 2))
        ttk.Entry(frame, textvariable=self.max_events_var, width=8).grid(
            row=row_extra, column=3, sticky=tk.W)

        # srun prefix (for custom allocation flags)
        row_srun = row_extra + 1
        self.srun_var = tk.StringVar(value="srun -n 20")
        ttk.Label(frame, text="srun cmd:").grid(
            row=row_srun, column=0, sticky=tk.E, padx=(8, 2))
        ttk.Entry(frame, textvariable=self.srun_var, width=50).grid(
            row=row_srun, column=1, columnspan=5, sticky=tk.W, padx=(0, 8), pady=2)

    # ── Control buttons ──────────────────────────────────────────────────

    def _build_control_frame(self):
        frame = ttk.Frame(self.root, padding=4)
        frame.pack(fill=tk.X, padx=8)

        self.launch_btn = ttk.Button(frame, text="Launch", command=self._launch)
        self.launch_btn.pack(side=tk.LEFT, padx=4)

        self.stop_btn = ttk.Button(frame, text="Stop", command=self._stop,
                                   state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=4)

        self.clear_btn = ttk.Button(frame, text="Clear", command=self._clear_console)
        self.clear_btn.pack(side=tk.LEFT, padx=4)

        self.cmd_label = ttk.Label(frame, text="", foreground="gray")
        self.cmd_label.pack(side=tk.LEFT, padx=12, fill=tk.X, expand=True)

    # ── Console ──────────────────────────────────────────────────────────

    def _build_console_frame(self):
        frame = ttk.LabelFrame(self.root, text="Results", padding=4)
        frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        self.console = scrolledtext.ScrolledText(
            frame, wrap=tk.WORD, font=("Courier", 10),
            state=tk.DISABLED, bg="#1e1e1e", fg="#d4d4d4",
            insertbackground="white")
        self.console.pack(fill=tk.BOTH, expand=True)

        # Color tags
        self.console.tag_config("info", foreground="#d4d4d4")
        self.console.tag_config("good", foreground="#4ec9b0")   # green — good reso
        self.console.tag_config("warn", foreground="#dcdcaa")    # yellow — medium
        self.console.tag_config("bad", foreground="#f44747")     # red — poor/error
        self.console.tag_config("system", foreground="#569cd6")  # blue — system msgs

    # ── Status bar ───────────────────────────────────────────────────────

    def _build_status_bar(self):
        frame = ttk.Frame(self.root, padding=2)
        frame.pack(fill=tk.X, padx=8, pady=(0, 4))

        self.status_label = ttk.Label(frame, text="Idle", foreground="gray")
        self.status_label.pack(side=tk.LEFT)

        self.rate_label = ttk.Label(frame, text="", foreground="gray")
        self.rate_label.pack(side=tk.RIGHT)

        self.count_label = ttk.Label(frame, text="0 results", foreground="gray")
        self.count_label.pack(side=tk.RIGHT, padx=16)

    # ── ZMQ setup ────────────────────────────────────────────────────────

    def _setup_zmq(self):
        import zmq
        self.zmq_ctx = zmq.Context()
        self.zmq_sock = self.zmq_ctx.socket(zmq.PULL)
        self.zmq_sock.bind(f"tcp://*:{self.ui_port}")
        self._log(f"Listening for results on port {self.ui_port}", tag="system")

    # ── Launch / Stop ────────────────────────────────────────────────────

    def _build_cmd(self):
        """Build the srun + live_distributor command from UI fields."""
        f = {k: v.get().strip() for k, v in self.fields.items()}

        srun_cmd = self.srun_var.get().strip()

        cmd = srun_cmd.split()
        cmd += ["resonet.psana.live_distributor"]
        cmd += ["--exp", f["exp"]]
        cmd += ["--run", f["run"]]
        cmd += ["--hosts"] + f["hosts"].split()
        cmd += ["--nwork-per-host", f["nwork"]]
        cmd += ["--wavelen", f["wavelen"]]
        cmd += ["--detdist", f["detdist"]]
        cmd += ["--pixsize", f["pixsize"]]
        cmd += ["--center-mm"] + f["center_mm"].split()
        cmd += ["--ds-factor", f["ds_factor"]]
        cmd += ["--det-name", f["det_name"]]
        cmd += ["--port-base", f["port_base"]]

        if self.live_var.get():
            cmd += ["--live"]

        max_ev = self.max_events_var.get().strip()
        if max_ev:
            cmd += ["--max-events", max_ev]

        return cmd

    def _launch(self):
        if self.proc is not None:
            self._log("Already running! Stop first.", tag="warn")
            return

        cmd = self._build_cmd()
        cmd_str = " ".join(cmd)
        self._log(f"Launching: {cmd_str}", tag="system")
        self.cmd_label.config(text=cmd_str[:80] + ("..." if len(cmd_str) > 80 else ""))

        try:
            self.proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                preexec_fn=os.setsid)
            self.running = True
            self.launch_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.NORMAL)
            self.status_label.config(text="Running", foreground="green")
            self._log(f"PID {self.proc.pid}", tag="system")
            self._poll_subprocess()
        except Exception as e:
            self._log(f"Launch failed: {e}", tag="bad")

    def _stop(self):
        if self.proc is not None:
            self._log("Sending SIGTERM...", tag="system")
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
            self.proc = None
            self.running = False
            self.launch_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)
            self.status_label.config(text="Stopped", foreground="orange")

    # ── Polling loops ────────────────────────────────────────────────────

    def _poll_results(self):
        """Non-blocking poll of ZMQ PULL socket for worker results."""
        import zmq
        batch = 0
        while batch < 50:  # process up to 50 messages per tick
            try:
                msg = self.zmq_sock.recv_json(flags=zmq.NOBLOCK)
            except zmq.Again:
                break

            now = time.time()
            self.result_times.append(now)
            self.total_results += 1
            batch += 1

            ts = msg.get("timestamp", 0)
            reso = msg.get("resolution", -1)
            evt = msg.get("event", "?")
            node = msg.get("node", "?")
            gpu = msg.get("gpu", "?")
            status = msg.get("status", "?")

            # Format timestamp
            if ts > 1e15:  # nanoseconds
                ts_str = f"{ts / 1e9:.3f}"
            elif ts > 1e9:  # seconds (epoch)
                ts_str = datetime.fromtimestamp(ts).strftime("%H:%M:%S.%f")[:-3]
            else:
                ts_str = str(ts)

            # Color by resolution quality
            if status != "OK":
                tag = "bad"
                line = f"evt {evt}  ts={ts_str}  ERROR: {status}  [{node}:gpu{gpu}]"
            elif reso < 0:
                tag = "bad"
                line = f"evt {evt}  ts={ts_str}  reso=FAIL  [{node}:gpu{gpu}]"
            elif reso < 3.0:
                tag = "good"
                line = f"evt {evt}  ts={ts_str}  reso={reso:.2f} A  [{node}:gpu{gpu}]"
            elif reso < 5.0:
                tag = "warn"
                line = f"evt {evt}  ts={ts_str}  reso={reso:.2f} A  [{node}:gpu{gpu}]"
            else:
                tag = "info"
                line = f"evt {evt}  ts={ts_str}  reso={reso:.2f} A  [{node}:gpu{gpu}]"

            self._log(line, tag=tag)

        # Update rate display
        if self.result_times:
            now = time.time()
            recent = [t for t in self.result_times if now - t < 5.0]
            if len(recent) > 1:
                rate = len(recent) / (now - recent[0])
                self.rate_label.config(text=f"{rate:.0f} ev/sec")
            self.count_label.config(text=f"{self.total_results} results")

        # Schedule next poll (20ms = 50Hz UI update)
        self.root.after(20, self._poll_results)

    def _poll_subprocess(self):
        """Check if the subprocess has exited."""
        if self.proc is None:
            return
        ret = self.proc.poll()
        if ret is not None:
            # Read remaining stdout
            remaining = self.proc.stdout.read()
            if remaining:
                for line in remaining.decode(errors="replace").splitlines():
                    self._log(line, tag="system")
            self._log(f"Process exited with code {ret}", tag="system")
            self.proc = None
            self.running = False
            self.launch_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)
            self.status_label.config(text="Exited", foreground="gray")
        else:
            # Read any available stdout without blocking
            import select
            while select.select([self.proc.stdout], [], [], 0)[0]:
                line = self.proc.stdout.readline()
                if not line:
                    break
                self._log(line.decode(errors="replace").rstrip(), tag="system")
            self.root.after(500, self._poll_subprocess)

    # ── Helpers ──────────────────────────────────────────────────────────

    def _log(self, text, tag="info"):
        self.console.config(state=tk.NORMAL)
        self.console.insert(tk.END, text + "\n", tag)
        self.console.see(tk.END)
        self.console.config(state=tk.DISABLED)

    def _clear_console(self):
        self.console.config(state=tk.NORMAL)
        self.console.delete("1.0", tk.END)
        self.console.config(state=tk.DISABLED)
        self.total_results = 0
        self.result_times.clear()
        self.count_label.config(text="0 results")
        self.rate_label.config(text="")

    def _on_close(self):
        self._stop()
        if self.zmq_sock:
            self.zmq_sock.close()
        if self.zmq_ctx:
            self.zmq_ctx.term()
        self.root.destroy()


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Resonet live processing control UI")
    parser.add_argument("--ui-port", type=int, default=5600,
                        help="Port to receive results from workers (default 5600)")
    args = parser.parse_args()

    root = tk.Tk()
    app = ResonetControlUI(root, ui_port=args.ui_port)
    root.mainloop()


if __name__ == "__main__":
    main()
